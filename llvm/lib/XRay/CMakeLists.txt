add_llvm_library(LLVMXRay
  FDRRecords.cpp
  FDRTraceWriter.cpp
  FileHeaderReader.cpp
  InstrumentationMap.cpp
  Profile.cpp
  RecordInitializer.cpp
  Trace.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ADT
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/XRay

  DEPENDS
  LLVMSupport
  LLVMObject

  LINK_LIBS
  LLVMSupport
  LLVMObject
  )
