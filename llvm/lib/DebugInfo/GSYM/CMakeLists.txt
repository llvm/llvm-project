add_llvm_component_library(LLVMDebugInfoGSYM
  DwarfTransformer.cpp
  Header.cpp
  HeaderV2.cpp
  FileWriter.cpp
  FunctionInfo.cpp
  GlobalData.cpp
  GsymCreator.cpp
  GsymCreatorV1.cpp
  GsymCreatorV2.cpp
  GsymContext.cpp
  GsymReader.cpp
  GsymReaderV1.cpp
  GsymReaderV2.cpp
  InlineInfo.cpp
  LineTable.cpp
  LookupResult.cpp
  CallSiteInfo.cpp
  MergedFunctionsInfo.cpp
  ObjectFileTransformer.cpp
  ExtractRanges.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/GSYM
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo

  DEPENDS
  LLVMMC

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  MC
  Object
  Support
  TargetParser
  DebugInfoDWARF
  )
