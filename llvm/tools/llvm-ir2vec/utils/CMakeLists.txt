add_llvm_library(emb_utils STATIC
  utils.cpp
)

target_link_libraries(emb_utils PRIVATE
  LLVMAnalysis LLVMCore LLVMSupport LLVMIRReader 
  LLVMAsmParser LLVMPasses LLVMDemangle
)