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

  if (NOT BUILD_SHARED_LIBS AND NOT LLVM_LINK_LLVM_DYLIB)
    # libremarks.cpp only defines LLVMRemarkVersion. Without forcing the
    # inclusion of all libLLVMRemarks.a members, the linker omits the C API
    # implementations because nothing references them.
    if (APPLE)
      target_link_options(Remarks PRIVATE
        "LINKER:-force_load,$<TARGET_LINKER_FILE:LLVMRemarks>")
    elseif(MSVC)
      target_link_options(Remarks PRIVATE
        "LINKER:/WHOLEARCHIVE:$<TARGET_LINKER_FILE:LLVMRemarks>")
    else()
      target_link_options(Remarks PRIVATE
        "LINKER:--whole-archive"
        "LINKER:$<TARGET_LINKER_FILE:LLVMRemarks>"
        "LINKER:--no-whole-archive")
    endif()
  endif()

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
