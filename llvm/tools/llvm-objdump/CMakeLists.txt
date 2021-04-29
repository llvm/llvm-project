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

add_llvm_tool(llvm-objdump
  llvm-objdump.cpp
  COFFDump.cpp
  ELFDump.cpp
  MachODump.cpp
  WasmDump.cpp
  XCOFFDump.cpp
  DEPENDS
  ObjdumpOptsTableGen
  )

if(HAVE_LIBXAR)
  target_link_libraries(llvm-objdump PRIVATE ${XAR_LIB})
endif()

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(objdump llvm-objdump)
endif()
