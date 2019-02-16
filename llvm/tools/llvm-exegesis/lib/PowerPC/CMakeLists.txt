include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC
  ${LLVM_BINARY_DIR}/lib/Target/PowerPC
  )

add_library(LLVMExegesisPowerPC
  STATIC
  Target.cpp
  )

llvm_update_compile_flags(LLVMExegesisPowerPC)
llvm_map_components_to_libnames(libs
  PowerPC
  Exegesis
  )

target_link_libraries(LLVMExegesisPowerPC ${libs})
set_target_properties(LLVMExegesisPowerPC PROPERTIES FOLDER "Libraries")
