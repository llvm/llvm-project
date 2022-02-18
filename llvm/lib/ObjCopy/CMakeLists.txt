add_llvm_component_library(LLVMObjCopy
  Archive.cpp
  ObjCopy.cpp
  ConfigManager.cpp
  COFF/COFFObjcopy.cpp
  COFF/Object.cpp
  COFF/Reader.cpp
  COFF/Writer.cpp
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

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Object

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Object
  Support
  MC
  )
