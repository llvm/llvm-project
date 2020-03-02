# MLIR project.
set(MLIR_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include ) # --src-root
set(MLIR_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include ) # --includedir
set(MLIR_TABLEGEN_EXE mlir-tblgen)

set(MLIR_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(AddMLIR)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(mlir-headers)
set_target_properties(mlir-headers PROPERTIES FOLDER "Misc")
add_custom_target(mlir-doc)

# Build the CUDA conversions and run according tests if the NVPTX backend
# is available
if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(MLIR_CUDA_CONVERSIONS_ENABLED 1)
else()
  set(MLIR_CUDA_CONVERSIONS_ENABLED 0)
endif()
# TODO: we should use a config.h file like LLVM does
add_definitions(-DMLIR_CUDA_CONVERSIONS_ENABLED=${MLIR_CUDA_CONVERSIONS_ENABLED})

set(MLIR_CUDA_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir CUDA runner")
set(MLIR_VULKAN_RUNNER_ENABLED 0 CACHE BOOL "Enable building the mlir Vulkan runner")

include_directories( "include")
include_directories( ${MLIR_INCLUDE_DIR})

add_subdirectory(include/mlir)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(test)

if( LLVM_INCLUDE_EXAMPLES )
  add_subdirectory(examples)
endif()

option(MLIR_INCLUDE_DOCS "Generate build targets for the MLIR docs."
  ${LLVM_INCLUDE_DOCS})
if (MLIR_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(DIRECTORY include/mlir include/mlir-c
    DESTINATION include
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "LICENSE.TXT"
    )

  install(DIRECTORY ${MLIR_INCLUDE_DIR}/mlir ${MLIR_INCLUDE_DIR}/mlir-c
    DESTINATION include
    COMPONENT mlir-headers
    FILES_MATCHING
    PATTERN "*.def"
    PATTERN "*.h"
    PATTERN "*.gen"
    PATTERN "*.inc"
    PATTERN "*.td"
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "config.h" EXCLUDE
    )

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-mlir-headers
                             DEPENDS mlir-headers
                             COMPONENT mlir-headers)
  endif()
endif()

add_subdirectory(cmake/modules)
