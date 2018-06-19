
set(LLVM_LINK_COMPONENTS
  Support
  native
  )

add_llvm_tool(llvm-exegesis
  llvm-exegesis.cpp
  )

add_subdirectory(lib)

# Link the native exegesis target if compiled and on the right host.
if ((LLVM_TARGETS_TO_BUILD MATCHES "X86") AND ("${LLVM_NATIVE_ARCH}" STREQUAL "X86"))
  set(LLVM_EXEGESIS_NATIVE_ARCH "X86")
endif()

if (LLVM_EXEGESIS_NATIVE_ARCH)
  set(LLVM_EXEGESIS_NATIVE_TARGET "LLVMExegesis${LLVM_EXEGESIS_NATIVE_ARCH}")
  set_source_files_properties(llvm-exegesis.cpp PROPERTIES COMPILE_FLAGS "-DLLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET=Initialize${LLVM_EXEGESIS_NATIVE_ARCH}ExegesisTarget")
endif()

target_link_libraries(llvm-exegesis PRIVATE
  LLVMExegesis
  ${LLVM_EXEGESIS_NATIVE_TARGET}
  )
