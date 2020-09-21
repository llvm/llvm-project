set(LLVM_LINK_COMPONENTS
  Object
  Option
  Support
  MC
  )

set(LLVM_TARGET_DEFINITIONS ObjcopyOpts.td)
tablegen(LLVM ObjcopyOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ObjcopyOptsTableGen)

set(LLVM_TARGET_DEFINITIONS InstallNameToolOpts.td)
tablegen(LLVM InstallNameToolOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(InstallNameToolOptsTableGen)

set(LLVM_TARGET_DEFINITIONS BitcodeStripOpts.td)
tablegen(LLVM BitcodeStripOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(BitcodeStripOptsTableGen)

set(LLVM_TARGET_DEFINITIONS StripOpts.td)
tablegen(LLVM StripOpts.inc -gen-opt-parser-defs)
add_public_tablegen_target(StripOptsTableGen)

add_llvm_tool(llvm-objcopy
  Buffer.cpp
  CopyConfig.cpp
  llvm-objcopy.cpp
  COFF/COFFObjcopy.cpp
  COFF/Object.cpp
  COFF/Reader.cpp
  COFF/Writer.cpp
  ELF/ELFConfig.cpp
  ELF/ELFObjcopy.cpp
  ELF/Object.cpp
  MachO/MachOObjcopy.cpp
  MachO/MachOReader.cpp
  MachO/MachOWriter.cpp
  MachO/MachOLayoutBuilder.cpp
  MachO/Object.cpp
  wasm/Object.cpp
  wasm/Reader.cpp
  wasm/Writer.cpp
  wasm/WasmObjcopy.cpp
  DEPENDS
  ObjcopyOptsTableGen
  InstallNameToolOptsTableGen
  StripOptsTableGen
  )

add_llvm_tool_symlink(llvm-install-name-tool llvm-objcopy)
add_llvm_tool_symlink(llvm-bitcode-strip llvm-objcopy)
add_llvm_tool_symlink(llvm-strip llvm-objcopy)

if(LLVM_INSTALL_BINUTILS_SYMLINKS)
  add_llvm_tool_symlink(objcopy llvm-objcopy)
  add_llvm_tool_symlink(strip llvm-objcopy)
endif()

if(LLVM_INSTALL_CCTOOLS_SYMLINKS)
  add_llvm_tool_symlink(install_name_tool llvm-install-name-tool)
  add_llvm_tool_symlink(bitcode_strip llvm-bitcode-strip)
endif()
