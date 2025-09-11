add_llvm_component_library(LLVMSupportLSP 
  Protocol.cpp
  Transport.cpp
  Logging.cpp

  DEPENDS
  LLVMSupport
)
