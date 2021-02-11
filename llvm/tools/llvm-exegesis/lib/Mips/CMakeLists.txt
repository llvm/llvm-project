include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/Mips
  ${LLVM_BINARY_DIR}/lib/Target/Mips
  )

add_library(LLVMExegesisMips
  STATIC
  Target.cpp
  )

llvm_update_compile_flags(LLVMExegesisMips)
llvm_map_components_to_libnames(libs
  Mips
  Exegesis
  )

target_link_libraries(LLVMExegesisMips ${libs})
set_target_properties(LLVMExegesisMips PROPERTIES FOLDER "Libraries")
