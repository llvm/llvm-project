add_custom_target(OffloadUnitTests)
set_target_properties(OffloadUnitTests PROPERTIES FOLDER "Tests/UnitTests")

if (CMAKE_CROSSCOMPILING)
  # TODO: It is possible that LLVM_GTEST_RUN_UNDER defines an emulator or
  #       ssh remote command invocation; for this case provide an option to
  #       enable unittests.
  message(STATUS "Offload unittests disabled because we are cross-compiling")
  return ()
endif ()

if (NOT TARGET llvm_gtest)
  message(WARNING "Offload unittests disabled due to GTest being unavailable; "
                  "Try LLVM_INSTALL_GTEST=ON for the LLVM build")
  return ()
endif ()

set(OFFLOAD_UNITTESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR})

function(add_offload_test_device_code test_filename test_name)
  set(SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${test_filename})
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

  # Try to build with support for NVPTX devices.
  if("cuda" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      get_filename_component(cuda_path "${CUDAToolkit_BIN_DIR}" DIRECTORY ABSOLUTE)
    endif()
    check_cxx_compiler_flag(
      "--target=nvptx64-nvidia-cuda -march=native --cuda-path=${cuda_path}" PLATFORM_HAS_NVPTX)

    if(OFFLOAD_TESTS_FORCE_NVPTX_ARCH)
      set(nvptx_arch "${OFFLOAD_TESTS_FORCE_NVPTX_ARCH}")
    elseif(PLATFORM_HAS_NVPTX)
      set(nvptx_arch "native")
    endif()

    if(nvptx_arch AND CUDAToolkit_FOUND)
      set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.nvptx64.bin")
      add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_CXX_COMPILER}
        -I${OFFLOAD_UNITTESTS_DIR}
        --target=nvptx64-nvidia-cuda -march=${nvptx_arch}
        -nogpulib --cuda-path=${cuda_path} -flto ${ARGN}
        ${SRC_PATH} -o ${output_file}
        DEPENDS ${SRC_PATH}
      )
      add_custom_target(${test_name}.nvptx64 DEPENDS ${output_file})
    endif()
  endif()

  # Try to build with support for AMDGPU devices.
  if("amdgpu" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
    check_cxx_compiler_flag("--target=amdgcn-amd-amdhsa -mcpu=native" PLATFORM_HAS_AMDGPU)

    if(OFFLOAD_TESTS_FORCE_AMDGPU_ARCH)
      set(amdgpu_arch "${OFFLOAD_TESTS_FORCE_AMDGPU_ARCH}")
    elseif(PLATFORM_HAS_AMDGPU)
      set(amdgpu_arch "native")
    endif()

    if(amdgpu_arch)
      set(output_file "${CMAKE_CURRENT_BINARY_DIR}/${test_name}.amdgpu.bin")
      add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_CXX_COMPILER}
        -I${OFFLOAD_UNITTESTS_DIR}
        --target=amdgcn-amd-amdhsa -mcpu=${amdgpu_arch}
        -nogpulib -flto ${ARGN} ${SRC_PATH} -o ${output_file}
        DEPENDS ${SRC_PATH}
      )
      add_custom_target(${test_name}.amdgpu DEPENDS ${output_file})
    endif()
  endif()

  # Create a single dependency target for the device code.
  add_custom_target(${test_name}.bin)
  if(TARGET ${test_name}.amdgpu)
    add_dependencies(${test_name}.bin ${test_name}.amdgpu)
  endif()
  if(TARGET ${test_name}.nvptx64)
    add_dependencies(${test_name}.bin ${test_name}.nvptx64)
  endif()
endfunction()

function(add_offload_unittest test_dirname)
  set(target_name "${test_dirname}.unittests")

  list(TRANSFORM ARGN PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE files)

  add_unittest(OffloadUnitTests "${target_name}"
    ${CMAKE_CURRENT_SOURCE_DIR}/common/Environment.cpp
    ${files})
  add_dependencies(${target_name} ${PLUGINS_TEST_COMMON} offload_device_binaries)
  target_compile_definitions(${target_name} PRIVATE DEVICE_CODE_PATH="${OFFLOAD_TEST_DEVICE_CODE_PATH}")
  target_link_libraries(${target_name} PRIVATE ${PLUGINS_TEST_COMMON})
  target_include_directories(${target_name} PRIVATE ${PLUGINS_TEST_INCLUDE})
endfunction()

function(add_conformance_test test_name)
  set(target_name "${test_name}.conformance")

  list(TRANSFORM ARGN PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/" OUTPUT_VARIABLE files)

  add_executable(${target_name} ${files})
  add_dependencies(${target_name} conformance_device_binaries)
  target_compile_definitions(${target_name}
    PRIVATE DEVICE_BINARY_DIR="${OFFLOAD_CONFORMANCE_DEVICE_BINARY_DIR}")
  target_link_libraries(${target_name} PRIVATE ${PLUGINS_TEST_COMMON} libc)
  set_target_properties(${target_name} PROPERTIES EXCLUDE_FROM_ALL TRUE)

  add_custom_target(offload.conformance.${test_name}
    COMMAND $<TARGET_FILE:${target_name}>
    DEPENDS ${target_name})
  add_dependencies(offload.conformance offload.conformance.${test_name})
endfunction()

set(OFFLOAD_TESTS_FORCE_NVPTX_ARCH "" CACHE STRING
  "Force building of NVPTX device code for Offload unit tests with the given arch, e.g. sm_61")
set(OFFLOAD_TESTS_FORCE_AMDGPU_ARCH "" CACHE STRING
  "Force building of AMDGPU device code for Offload unit tests with the given arch, e.g. gfx1030")

add_subdirectory(OffloadAPI)
add_subdirectory(Conformance)
