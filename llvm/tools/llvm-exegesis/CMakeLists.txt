set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Support
  )

add_llvm_tool(llvm-exegesis
  DISABLE_LLVM_LINK_LLVM_DYLIB
  llvm-exegesis.cpp

  DEPENDS
  intrinsics_gen
  )

# Has side effect of defining LLVM_EXEGESIS_TARGETS
add_subdirectory(lib)

# Link all enabled exegesis targets
set(libs)
foreach(t ${LLVM_EXEGESIS_TARGETS})
  string(STRIP ${t} t)
  list(APPEND libs "LLVMExegesis${t}")
endforeach()

target_link_libraries(llvm-exegesis PRIVATE
  LLVMExegesis
  ${libs}
  )
