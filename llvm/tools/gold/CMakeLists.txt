set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/gold.exports)

if( LLVM_ENABLE_PIC AND LLVM_BINUTILS_INCDIR )
  include_directories( ${LLVM_BINUTILS_INCDIR} )

  # Because off_t is used in the public API, the largefile parts are required for
  # ABI compatibility.
  add_definitions( -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64 )

  set(LLVM_LINK_COMPONENTS
     ${LLVM_TARGETS_TO_BUILD}
     Linker
     LTO
     BitWriter
     IPO
     )

  add_llvm_loadable_module(LLVMgold
    gold-plugin.cpp
    )

endif()

