set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
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

# Register the native target (we don't yet support -march)
if (LLVM_EXEGESIS_TARGETS MATCHES "${LLVM_NATIVE_ARCH}")
  set(LLVM_EXEGESIS_NATIVE_ARCH "${LLVM_NATIVE_ARCH}")
  set_source_files_properties(llvm-exegesis.cpp PROPERTIES COMPILE_FLAGS "-DLLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET=Initialize${LLVM_EXEGESIS_NATIVE_ARCH}ExegesisTarget")
endif()

# Link the native exegesis targets
set(libs)
foreach(t ${LLVM_EXEGESIS_TARGETS})
  string(STRIP ${t} t)
  list(APPEND libs "LLVMExegesis${t}")
endforeach()

target_link_libraries(llvm-exegesis PRIVATE
  LLVMExegesis
  ${libs}
  )
