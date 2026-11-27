add_llvm_component_library(LLVMDebugInfoCVPack
  COFFEdit.cpp
  FunctionDebug.cpp
  IndexRemap.cpp
  Materialize.cpp
  PackFile.cpp
  PackIndex.cpp
  Producer.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/CVPack

  LINK_COMPONENTS
  BinaryFormat
  DebugInfoCodeView
  Object
  Support
  )
