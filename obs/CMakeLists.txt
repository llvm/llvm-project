include(CMakeDependentOption)
include(GNUInstallDirs)

# Make sure that our source directory is on the current cmake module path so
# that we can include cmake files from this directory.
list(INSERT CMAKE_MODULE_PATH 0
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  "${LLVM_COMMON_CMAKE_UTILS}/Modules"
  )

include(CMakeParseArguments)
include(AddOBS)


add_subdirectory(obs-ir)