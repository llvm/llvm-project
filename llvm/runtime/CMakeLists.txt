if( NOT LLVM_BUILD_RUNTIME )
  set(EXCLUDE_FROM_ALL ON)
endif()

add_subdirectory(libprofile)
