set(LLVM_LINK_COMPONENTS
  # Machine IR components (for -mode=mir)
  MIRParser
  
  # Target initialization (required for MIR parsing)
  AllTargetsAsmParsers
  AllTargetsCodeGens
  )

# Add the utils subdirectory for the library
add_subdirectory(utils)

# Main executable
add_llvm_tool(llvm-ir2vec
  llvm-ir2vec.cpp
  
  DEPENDS
  intrinsics_gen
)

target_include_directories(llvm-ir2vec PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utils)
target_link_libraries(llvm-ir2vec PRIVATE emb_utils)
