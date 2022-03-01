set(LLVM_OPTIONAL_SOURCES
  mlir-vulkan-runner.cpp
  vulkan-runtime-wrappers.cpp
  VulkanRuntime.cpp
  VulkanRuntime.h
  )

if (MLIR_ENABLE_VULKAN_RUNNER)
  message(STATUS "Building the Vulkan runner")

  # At first try "FindVulkan" from:
  # https://cmake.org/cmake/help/v3.7/module/FindVulkan.html
  if (NOT CMAKE_VERSION VERSION_LESS 3.7.0)
    find_package(Vulkan)
  endif()

  # If Vulkan is not found try a path specified by VULKAN_SDK.
  if (NOT Vulkan_FOUND)
    if ("$ENV{VULKAN_SDK}" STREQUAL "")
      message(FATAL_ERROR "Please use at least CMAKE 3.7.0 or provide "
                          "VULKAN_SDK path as an environment variable")
    endif()

    find_library(Vulkan_LIBRARY vulkan HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
    if (Vulkan_LIBRARY)
      set(Vulkan_FOUND ON)
      set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
      message(STATUS "Found Vulkan: " ${Vulkan_LIBRARY})
    endif()
  endif()

  if (NOT Vulkan_FOUND)
    message(FATAL_ERROR "Cannot find Vulkan library")
  endif()

  add_llvm_library(vulkan-runtime-wrappers SHARED
    vulkan-runtime-wrappers.cpp
    VulkanRuntime.cpp
  )

  target_include_directories(vulkan-runtime-wrappers
    PUBLIC
    ${Vulkan_INCLUDE_DIR}
  )

  # *IMPORTANT*: This library cannot depend on LLVM libraries. Otherwise,
  # it may cause LLVM version conflict when used together with other shared
  # libraries depending on LLVM. Notably, Mesa, who implements Vulkan
  # drivers on Linux, depends on the system libLLVM.so.
  target_link_libraries(vulkan-runtime-wrappers
    PUBLIC
    ${Vulkan_LIBRARY}
  )

  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  set(LIBS
    ${conversion_libs}
    MLIRAnalysis
    MLIRArithmetic
    MLIRExecutionEngine
    MLIRFunc
    MLIRGPUOps
    MLIRIR
    MLIRJitRunner
    MLIRLLVMIR
    MLIRLLVMCommonConversion
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRef
    MLIRMemRefToLLVM
    MLIRParser
    MLIRSPIRV
    MLIRSPIRVTransforms
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    MLIRTranslation
    ${Vulkan_LIBRARY}
  )

  # Manually expand the target library, since our MLIR libraries
  # aren't plugged into the LLVM dependency tracking. If we don't
  # do this then we can't insert the CodeGen library after ourselves
  llvm_expand_pseudo_components(TARGET_LIBS AllTargetsCodeGens)
  # Prepend LLVM in front of every target, this is how the library
  # are named with CMake
  SET(targets_to_link)
  FOREACH(t ${TARGET_LIBS})
    LIST(APPEND targets_to_link "LLVM${t}")
  ENDFOREACH(t)

  add_llvm_tool(mlir-vulkan-runner
    mlir-vulkan-runner.cpp

    DEPENDS
    vulkan-runtime-wrappers
  )
  llvm_update_compile_flags(mlir-vulkan-runner)
  target_link_libraries(mlir-vulkan-runner PRIVATE ${LIBS})

endif()
