add_llvm_component_library(LLVMSupportLSP 
  Protocol.cpp
  Transport.cpp
  Logging.cpp

  LINK_COMPONENTS
  Support
)
