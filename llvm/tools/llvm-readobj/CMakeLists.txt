set(LLVM_LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  DebugInfoDWARF
  Demangle
  Object
  Option
  Support
  TargetParser
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ReadobjOptsTableGen)

# ELFDumper.cpp defines createError and checkHashTable that conflict with
# identically-named functions from other dumper files in unity builds.
set_source_files_properties(
  ELFDumper.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
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
  DEPENDS
  ReadobjOptsTableGen
  GENERATE_DRIVER
  )

setup_host_tool(llvm-readobj LLVM_READOBJ llvm_readobj_exe llvm_readobj_target)

add_llvm_tool_symlink(llvm-readelf llvm-readobj)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(readelf llvm-readobj)
endif()
