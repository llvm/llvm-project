# Building shared libraries requires PIC objects.
if(LLVM_ENABLE_PIC)

  set(LLVM_LINK_COMPONENTS
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsDisassemblers
    AllTargetsInfos
    BitReader
    Core
    CodeGen
    LTO
    MC
    MCDisassembler
    Support
    Target
    TargetParser
    )

  set(SOURCES
    LTODisassembler.cpp
    lto.cpp
    )

  set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/lto.exports)

  if(CMAKE_SYSTEM_NAME STREQUAL "AIX")
      set(LTO_LIBRARY_TYPE MODULE)
      set(LTO_LIBRARY_NAME libLTO)
    else()
      set(LTO_LIBRARY_TYPE SHARED)
      set(LTO_LIBRARY_NAME LTO)
  endif()

  add_llvm_library(${LTO_LIBRARY_NAME} ${LTO_LIBRARY_TYPE} INSTALL_WITH_TOOLCHAIN
      ${SOURCES} DEPENDS intrinsics_gen)

  install(FILES ${LLVM_MAIN_INCLUDE_DIR}/llvm-c/lto.h
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/llvm-c"
    COMPONENT LTO)

  if (APPLE)
    set(LTO_VERSION ${LLVM_VERSION_MAJOR})
    if(LLVM_LTO_VERSION_OFFSET)
      math(EXPR LTO_VERSION "${LLVM_VERSION_MAJOR} + ${LLVM_LTO_VERSION_OFFSET}")
    endif()
    set_property(TARGET LTO APPEND_STRING PROPERTY
                LINK_FLAGS
                " -compatibility_version 1 -current_version ${LTO_VERSION}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
    if(LLVM_USE_SANITIZER)
      execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=lib
                      OUTPUT_VARIABLE clang_lib_dir)
      string(STRIP ${clang_lib_dir} clang_lib_dir)
      if (clang_lib_dir MATCHES ".*xctoolchain.*")
        message(WARNING "Sanitized libLTO (with LLVM_USE_SANITIZER) is unsupported on Darwin when compiling "
                        "with a system sanitizer library. Please use just-built sanitizers.")
      endif()
    endif()
  endif()

endif()
