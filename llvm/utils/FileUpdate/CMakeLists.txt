add_llvm_utility(FileUpdate
  FileUpdate.cpp
  )

target_link_libraries(FileUpdate LLVMSupport)
if( MINGW )
  target_link_libraries(FileUpdate imagehlp psapi)
endif( MINGW )
if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
  target_link_libraries(FileUpdate pthread)
endif()
