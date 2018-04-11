set(LLVM_LINK_COMPONENTS
  Support
  native
  )

add_llvm_tool(llvm-exegesis
  llvm-exegesis.cpp
  )

add_subdirectory(lib)
target_link_libraries(llvm-exegesis PRIVATE LLVMExegesis)

if(LLVM_ENABLE_LIBPFM AND HAVE_LIBPFM)
  target_link_libraries(llvm-exegesis PRIVATE pfm)
endif()
