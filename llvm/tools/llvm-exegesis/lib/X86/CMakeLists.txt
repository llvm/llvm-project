add_library(LLVMExegesisX86
  STATIC
  Target.cpp
  )

llvm_update_compile_flags(LLVMExegesisX86)
llvm_map_components_to_libnames(libs
  X86
  Exegesis
  )

target_link_libraries(LLVMExegesisX86 ${libs})
set_target_properties(LLVMExegesisX86 PROPERTIES FOLDER "Libraries")
