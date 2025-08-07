add_mlir_library(MLIRTargetLLVM
  ModuleToObject.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVM

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRReader
  Linker
  MC
  Passes
  Support
  Target
  LINK_LIBS PUBLIC
  MLIRExecutionEngineUtils
  MLIRTargetLLVMIRExport
)

if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

add_mlir_dialect_library(MLIRNVVMTarget
  NVVM/Target.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/LLVMIR

  LINK_COMPONENTS
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRNVVMToLLVMIRTranslation
  )

if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  # Find the CUDA toolkit.
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
    # Get the CUDA toolkit path. The path is needed for detecting `libdevice.bc`.
    # These extra steps are needed because of a bug on CMake.
    # See: https://gitlab.kitware.com/cmake/cmake/-/issues/24858
    # TODO: Bump the MLIR CMake version to 3.26.4 and switch to
    # ${CUDAToolkit_LIBRARY_ROOT}
    if(NOT DEFINED CUDAToolkit_LIBRARY_ROOT)
      get_filename_component(MLIR_CUDAToolkit_ROOT ${CUDAToolkit_BIN_DIR}
                             DIRECTORY ABSOLUTE)
    else()
      set(MLIR_CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_ROOT})
    endif()

    # Add the `nvptxcompiler` library.
    if(MLIR_ENABLE_NVPTXCOMPILER)
      # Find the `nvptxcompiler` library.
      # TODO: Bump the MLIR CMake version to 3.25 and use `CUDA::nvptxcompiler_static`.
      find_library(MLIR_NVPTXCOMPILER_LIB_PATH nvptxcompiler_static
                  PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)

      # Fail if `nvptxcompiler_static` couldn't be found.
      if(MLIR_NVPTXCOMPILER_LIB_PATH STREQUAL "MLIR_NVPTXCOMPILER_LIB_PATH-NOTFOUND")
        message(FATAL_ERROR
                "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
      endif()

      add_library(MLIR_NVPTXCOMPILER_LIB STATIC IMPORTED GLOBAL)
      # Downstream projects can modify this path and use it in CMake. For example:
      # add_library(MLIR_NVPTXCOMPILER_LIB STATIC IMPORTED GLOBAL)
      # set_property(TARGET MLIR_NVPTXCOMPILER_LIB PROPERTY IMPORTED_LOCATION ${...})  
      # where `...` is to be replaced with the path to the library.
      set_property(TARGET MLIR_NVPTXCOMPILER_LIB PROPERTY IMPORTED_LOCATION ${MLIR_NVPTXCOMPILER_LIB_PATH})
      # Link against `nvptxcompiler_static`. TODO: use `CUDA::nvptxcompiler_static`.
      target_link_libraries(MLIRNVVMTarget PRIVATE MLIR_NVPTXCOMPILER_LIB)
      target_include_directories(obj.MLIRNVVMTarget PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

      # Add the `nvfatbin` library.
      find_library(MLIR_NVFATBIN_LIB_PATH nvfatbin_static
                  PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)
      # Fail if `nvfatbin_static` couldn't be found.
      if(MLIR_NVFATBIN_LIB_PATH STREQUAL "MLIR_NVFATBIN_LIB_PATH-NOTFOUND")
        message(FATAL_ERROR
                "Requested using the static `nvptxcompiler` library which requires the \
                'nvfatbin` library, but it couldn't be found.")
      endif()

      add_library(MLIR_NVFATBIN_LIB STATIC IMPORTED GLOBAL)
      set_property(TARGET MLIR_NVFATBIN_LIB PROPERTY IMPORTED_LOCATION ${MLIR_NVFATBIN_LIB_PATH})  
      target_link_libraries(MLIRNVVMTarget PRIVATE MLIR_NVFATBIN_LIB)
    endif()
  else()
    # Fail if `MLIR_ENABLE_NVPTXCOMPILER` is enabled and the toolkit couldn't be found.
    if(MLIR_ENABLE_NVPTXCOMPILER)
      message(FATAL_ERROR
              "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
    endif()
  endif()
  message(VERBOSE "MLIR default CUDA toolkit path: ${MLIR_CUDAToolkit_ROOT}")

  # Define the `CUDAToolkit` path.
  target_compile_definitions(obj.MLIRNVVMTarget
    PRIVATE
    __DEFAULT_CUDATOOLKIT_PATH__="${MLIR_CUDAToolkit_ROOT}"
  )
endif()


function(embed_binary_to_src file output_file symbol)
    file(READ ${file} filedata HEX)
    # Convert hex data for C compatibility
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
    # Write data to output file
    file(WRITE ${output_file} "const unsigned char ${symbol}[] = {${filedata}};\nconst int ${symbol}_size = sizeof(${symbol});\n")
endfunction()

set(MLIR_NVVM_EMBED_LIBDEVICE 0 CACHE BOOL "Embed CUDA libdevice.bc in the binary at build time instead of looking it up at runtime")
if (MLIR_NVVM_EMBED_LIBDEVICE)
  if (NOT MLIR_NVVM_LIBDEVICE_PATH)
    if(CUDAToolkit_FOUND)
      find_file(MLIR_NVVM_LIBDEVICE_PATH libdevice.10.bc
                PATHS ${CUDAToolkit_LIBRARY_ROOT}
                PATH_SUFFIXES "nvvm/libdevice" NO_DEFAULT_PATH REQUIRED)
    else()
      message(FATAL_ERROR
              "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
    endif()
  endif()
  
  embed_binary_to_src(${MLIR_NVVM_LIBDEVICE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/libdevice_embedded.c _mlir_embedded_libdevice)
  add_mlir_library(MLIRNVVMLibdevice
    ${CMAKE_CURRENT_BINARY_DIR}/libdevice_embedded.c
  )
  target_link_libraries(MLIRNVVMTarget PRIVATE MLIRNVVMLibdevice)
  target_compile_definitions(obj.MLIRNVVMTarget
    PRIVATE
    MLIR_NVVM_EMBED_LIBDEVICE=1
  )
else()
  target_compile_definitions(obj.MLIRNVVMTarget
    PRIVATE
    MLIR_NVVM_EMBED_LIBDEVICE=0
  )
endif()


if (MLIR_ENABLE_ROCM_CONVERSIONS)
  set(AMDGPU_LIBS
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

add_mlir_dialect_library(MLIRROCDLTarget
  ROCDL/Target.cpp
  ROCDL/Utils.cpp

  OBJECT

  LINK_COMPONENTS
  FrontendOffloading
  MCParser
  ${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRROCDLToLLVMIRTranslation
  )

if(MLIR_ENABLE_ROCM_CONVERSIONS)
  if (DEFINED ROCM_PATH)
    set(DEFAULT_ROCM_PATH "${ROCM_PATH}" CACHE PATH "Fallback path to search for ROCm installs")
  elseif(DEFINED ENV{ROCM_PATH})
    set(DEFAULT_ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Fallback path to search for ROCm installs")
  else()
    if (WIN32)
      # Avoid setting an UNIX path for Windows.
      # TODO: Eventually migrate to FindHIP once it becomes a part of CMake.
      set(DEFAULT_ROCM_PATH "" CACHE PATH "Fallback path to search for ROCm installs")
    else()
      set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
    endif()
  endif()
  message(VERBOSE "MLIR Default ROCM toolkit path: ${DEFAULT_ROCM_PATH}")

  target_compile_definitions(obj.MLIRROCDLTarget
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
  )
endif()

if ("SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(SPIRV_LIBS
    SPIRVCodeGen

  )
endif()

add_mlir_dialect_library(MLIRXeVMTarget
  XeVM/Target.cpp

  OBJECT

  LINK_COMPONENTS
  ${SPIRV_LIBS}

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRExecutionEngineUtils
  MLIRSupport
  MLIRGPUDialect
  MLIRTargetLLVM
  MLIRXeVMToLLVMIRTranslation
)
