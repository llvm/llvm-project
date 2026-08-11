# Building shared libraries requires PIC objects.
if(LLVM_ENABLE_PIC)

  set(LLVM_LINK_COMPONENTS
    Remarks
    )

  set(SOURCES
    libremarks.cpp
    )

  if (NOT (BUILD_SHARED_LIBS OR LLVM_LINK_LLVM_DYLIB))
    set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/Remarks.exports)
  endif()

  add_llvm_library(Remarks SHARED INSTALL_WITH_TOOLCHAIN ${SOURCES})

  if (LLVM_INTEGRATED_CRT_ALLOC AND MSVC)
    # Make sure we search LLVMSupport first, before the CRT libs
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -INCLUDE:malloc")
  endif()
  
  install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/Remarks.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/llvm-c"
    COMPONENT Remarks)

  if (APPLE)
    llvm_set_macho_current_version(Remarks ${LLVM_VERSION_MAJOR})
  endif()

endif()
