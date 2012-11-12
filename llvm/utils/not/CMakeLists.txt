add_llvm_utility(not
  not.cpp
  )

target_link_libraries(not LLVMSupport)
if( MINGW )
  target_link_libraries(not imagehlp psapi)
endif( MINGW )
if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
  target_link_libraries(not pthread)
endif()
