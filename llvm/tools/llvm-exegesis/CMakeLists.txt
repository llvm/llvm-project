set(LLVM_LINK_COMPONENTS
  MC
  MCParser
  Support
  native
  )

add_llvm_tool(llvm-exegesis
  DISABLE_LLVM_LINK_LLVM_DYLIB
  llvm-exegesis.cpp

  DEPENDS
  intrinsics_gen
  )

add_subdirectory(lib)

# Link the native exegesis target if compiled and on the right host.
if ((LLVM_TARGETS_TO_BUILD MATCHES "${LLVM_NATIVE_ARCH}") AND (LLVM_EXEGESIS_TARGETS MATCHES "${LLVM_NATIVE_ARCH}"))
  set(LLVM_EXEGESIS_NATIVE_ARCH "${LLVM_NATIVE_ARCH}")
endif()

if (LLVM_EXEGESIS_NATIVE_ARCH)
  set(LLVM_EXEGESIS_NATIVE_TARGET "LLVMExegesis${LLVM_EXEGESIS_NATIVE_ARCH}")
  set_source_files_properties(llvm-exegesis.cpp PROPERTIES COMPILE_FLAGS "-DLLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET=Initialize${LLVM_EXEGESIS_NATIVE_ARCH}ExegesisTarget")
endif()

target_link_libraries(llvm-exegesis PRIVATE
  LLVMExegesis
  ${LLVM_EXEGESIS_NATIVE_TARGET}
  )
