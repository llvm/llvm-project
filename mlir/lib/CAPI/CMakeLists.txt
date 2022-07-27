# For upstream, we accumulate all libraries into the MLIR_CAPI_LIBRARIES
# property via a custom wrapper function. This is then used to create an
# aggregate below.
set_property(GLOBAL APPEND PROPERTY MLIR_CAPI_LIBRARIES)
function(add_mlir_upstream_c_api_library name)
  add_mlir_public_c_api_library(${name} ${ARGN})
  set_property(GLOBAL APPEND PROPERTY MLIR_CAPI_LIBRARIES ${name})
endfunction()

add_subdirectory(Debug)
add_subdirectory(Dialect)
add_subdirectory(Conversion)
add_subdirectory(Interfaces)
add_subdirectory(IR)
add_subdirectory(RegisterEverything)
add_subdirectory(Transforms)

# Only enable the ExecutionEngine if the native target is configured in.
if(TARGET ${LLVM_NATIVE_ARCH})
  add_subdirectory(ExecutionEngine)
endif()

# Build the optional CAPI dylib.
if(MLIR_BUILD_MLIR_C_DYLIB)
  message(STATUS "Building MLIR-C dylib")
  get_property(_capi_libraries GLOBAL PROPERTY MLIR_CAPI_LIBRARIES)
  add_mlir_aggregate(MLIR-C
    SHARED
    EMBED_LIBS
      ${_capi_libraries}
  )
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(MLIR-C PRIVATE "-Wl,-exclude-libs,ALL")
  else()
    if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
      message(STATUS "MLIR-C on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
    endif()
  endif()
endif()

