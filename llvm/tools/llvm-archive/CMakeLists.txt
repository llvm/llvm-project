# This tool creates a static library from the LLVM libraries.

option(LLVM_BUILD_LLVM_ARCHIVE "Build libllvm static library" OFF)

if(LLVM_BUILD_LLVM_ARCHIVE)
  set(LLVM_ARCHIVE_COMPONENTS "all" CACHE STRING "Libraries to include in libLLVM archive")
  
  llvm_map_components_to_libnames(LIB_NAMES ${LLVM_ARCHIVE_COMPONENTS})

  list(REMOVE_ITEM LIB_NAMES "LLVMTableGen")

  list(REMOVE_DUPLICATES LIB_NAMES)
  # Link against the object libraries instead of static libraries.
  list(TRANSFORM LIB_NAMES PREPEND "obj.")

  add_llvm_library(LLVM_static STATIC
    LINK_LIBS ${LIB_NAMES}
    OUTPUT_NAME "LLVM"
    NO_EXPORT
    )
endif()


