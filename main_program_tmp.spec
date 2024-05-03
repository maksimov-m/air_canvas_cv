# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_program_tmp.py'],
      pathex=['/usr/lib/python3.9/site-packages/cv2/python-3.9'],
      binaries=[],
      datas=[('C:/Users/maks6/PycharmProjects/git/Air-Canvas-with-ML/env/Lib/site-packages/cv2', 'cv2/'), ('C:/Users/maks6/anaconda3/Lib/site-packages/mediapipe', 'mediapipe/'), ('C:/Users/maks6/PycharmProjects/git/Air-Canvas-with-ML/env/Lib/site-packages/customtkinter', 'customtkinter/'), ('C:/Users/maks6/PycharmProjects/git/Air-Canvas-with-ML/env/Lib/site-packages/tk', 'tk/')],
      hiddenimports=['mediapipe', 'mediapipe.python._framework_bindings', 'darkdetect','typing-extensions',  'packaging', 'tkinter.ttk', 'tkinter.font', 'tkinter.filedialog', 'customtkinter', 'platform', 'ctypes.wintypes', 'ctypes', 'packaging'],
      hookspath=[],
      hooksconfig={},
      runtime_hooks=[],
      excludes=[],
      noarchive=False,
      optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='customtkinter_prob',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_program_tmp',
)
