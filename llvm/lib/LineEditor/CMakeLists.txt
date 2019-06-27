if(HAVE_LIBEDIT)
  set(link_libs edit)
endif()

add_llvm_library(LLVMLineEditor
  LineEditor.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LineEditor

  LINK_LIBS
  LLVMSupport
  ${link_libs}
  )
