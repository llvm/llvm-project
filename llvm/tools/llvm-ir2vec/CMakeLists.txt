set(LLVM_LINK_COMPONENTS
  Analysis Core Demangle IRReader Support CodeGen MIRParser
  AllTargetsAsmParsers AllTargetsCodeGens AllTargetsDescs 
  AllTargetsInfos TargetParser
)

# Static utility library
add_library(ir2vec_utils STATIC utils/utils.cpp)

target_include_directories(ir2vec_utils 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/utils  # PUBLIC so consumers can find utils.h
  PRIVATE ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(ir2vec_utils PRIVATE
  LLVMAnalysis LLVMCore LLVMSupport LLVMIRReader 
  LLVMAsmParser LLVMPasses LLVMDemangle
)

target_compile_definitions(ir2vec_utils PRIVATE ${LLVM_DEFINITIONS})

# Required for Python shared library
set_target_properties(ir2vec_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Main executable
add_llvm_tool(llvm-ir2vec llvm-ir2vec.cpp DEPENDS intrinsics_gen)
target_link_libraries(llvm-ir2vec PRIVATE ir2vec_utils)
