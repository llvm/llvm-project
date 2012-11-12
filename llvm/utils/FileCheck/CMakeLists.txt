add_llvm_utility(FileCheck
  FileCheck.cpp
  )

target_link_libraries(FileCheck LLVMSupport)
if( MINGW )
  target_link_libraries(FileCheck imagehlp psapi)
endif( MINGW )
if( LLVM_ENABLE_THREADS AND HAVE_LIBPTHREAD )
  target_link_libraries(FileCheck pthread)
endif()
