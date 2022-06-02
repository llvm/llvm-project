set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  BinaryFormat
  DebugInfoDWARF
  DebugInfoPDB
  Demangle
  MC
  MCDisassembler
  Object
  Option
  Support
  Symbolize
  )

set(LLVM_TARGET_DEFINITIONS ObjdumpOpts.td)
tablegen(LLVM ObjdumpOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ObjdumpOptsTableGen)

set(LLVM_TARGET_DEFINITIONS OtoolOpts.td)
tablegen(LLVM OtoolOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(OtoolOptsTableGen)

add_llvm_tool(llvm-objdump
  llvm-objdump.cpp
  SourcePrinter.cpp
  COFFDump.cpp
  ELFDump.cpp
  MachODump.cpp
  OffloadDump.cpp
  WasmDump.cpp
  XCOFFDump.cpp
  DEPENDS
  ObjdumpOptsTableGen
  OtoolOptsTableGen
  )

if(LLVM_HAVE_LIBXAR)
  target_link_libraries(llvm-objdump PRIVATE ${XAR_LIB})
endif()

add_llvm_tool_symlink(llvm-otool llvm-objdump)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(objdump llvm-objdump)
endif()

if(LLVM_INSTALL_CCTOOLS_SYMLINKS)
  add_llvm_tool_symlink(otool llvm-otool)
endif()
