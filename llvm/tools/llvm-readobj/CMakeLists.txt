set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  DebugInfoDWARF
  Demangle
  Object
  BinaryFormat
  Support
  DebugInfoCodeView
  DebugInfoMSF
  DebugInfoPDB
  )

add_llvm_tool(llvm-readobj
  ARMWinEHPrinter.cpp
  COFFDumper.cpp
  COFFImportDumper.cpp
  ELFDumper.cpp
  llvm-readobj.cpp
  MachODumper.cpp
  ObjDumper.cpp
  WasmDumper.cpp
  Win64EHDumper.cpp
  WindowsResourceDumper.cpp
  XCOFFDumper.cpp
  )

add_llvm_tool_symlink(llvm-readelf llvm-readobj)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(readelf llvm-readobj)
endif()
