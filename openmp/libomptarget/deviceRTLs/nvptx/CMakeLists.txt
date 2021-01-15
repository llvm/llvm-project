##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build the NVPTX (CUDA) Device RTL if the CUDA tools are available
#
##===----------------------------------------------------------------------===##

get_filename_component(devicertl_base_directory
  ${CMAKE_CURRENT_SOURCE_DIR}
  DIRECTORY)
set(devicertl_common_directory
  ${devicertl_base_directory}/common)
set(devicertl_nvptx_directory
  ${devicertl_base_directory}/nvptx)

if(LIBOMPTARGET_DEP_CUDA_FOUND)
  # Build library support for the highest compute capability the system supports
  # and always build support for sm_35 by default
  if (${LIBOMPTARGET_DEP_CUDA_ARCH} EQUAL 35)
    set(default_capabilities 35)
  else()
      set(default_capabilities "35,${LIBOMPTARGET_DEP_CUDA_ARCH}")
  endif()

  if (DEFINED LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY)
    set(default_capabilities ${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY})
    libomptarget_warning_say("LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITY is deprecated, please use LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES")
  endif()
  set(LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES ${default_capabilities} CACHE STRING
    "List of CUDA Compute Capabilities to be used to compile the NVPTX device RTL.")
  string(REPLACE "," ";" nvptx_sm_list ${LIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES})

  foreach(sm ${nvptx_sm_list})
    set(CUDA_ARCH ${CUDA_ARCH} -gencode arch=compute_${sm},code=sm_${sm})
  endforeach()

  # Override default MAX_SM in src/target_impl.h if requested
  if (DEFINED LIBOMPTARGET_NVPTX_MAX_SM)
    set(MAX_SM_DEFINITION "-DMAX_SM=${LIBOMPTARGET_NVPTX_MAX_SM}")
  endif()

  # Activate RTL message dumps if requested by the user.
  set(LIBOMPTARGET_NVPTX_DEBUG FALSE CACHE BOOL
    "Activate NVPTX device RTL debug messages.")

  # Check if we can create an LLVM bitcode implementation of the runtime library
  # that could be inlined in the user application. For that we need to find
  # a Clang compiler capable of compiling our CUDA files to LLVM bitcode and
  # an LLVM linker.
  set(LIBOMPTARGET_NVPTX_CUDA_COMPILER "" CACHE STRING
    "Location of a CUDA compiler capable of emitting LLVM bitcode.")
  set(LIBOMPTARGET_NVPTX_BC_LINKER "" CACHE STRING
    "Location of a linker capable of linking LLVM bitcode objects.")

  include(LibomptargetNVPTXBitcodeLibrary)

  if (LIBOMPTARGET_NVPTX_BCLIB_SUPPORTED)
    libomptarget_say("Building CUDA LLVM bitcode offloading device RTL.")

    set(cuda_src_files
      ${devicertl_common_directory}/src/cancel.cu
      ${devicertl_common_directory}/src/critical.cu
      ${devicertl_common_directory}/src/data_sharing.cu
      ${devicertl_common_directory}/src/libcall.cu
      ${devicertl_common_directory}/src/loop.cu
      ${devicertl_common_directory}/src/omp_data.cu
      ${devicertl_common_directory}/src/omptarget.cu
      ${devicertl_common_directory}/src/parallel.cu
      ${devicertl_common_directory}/src/reduction.cu
      ${devicertl_common_directory}/src/support.cu
      ${devicertl_common_directory}/src/sync.cu
      ${devicertl_common_directory}/src/task.cu
      src/target_impl.cu
    )

    # Set flags for LLVM Bitcode compilation.
    set(bc_flags ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER_FLAGS}
                 -I${devicertl_base_directory}
                 -I${devicertl_nvptx_directory}/src)

    if(${LIBOMPTARGET_NVPTX_DEBUG})
      set(bc_flags ${bc_flags} -DOMPTARGET_NVPTX_DEBUG=-1)
    else()
      set(bc_flags ${bc_flags} -DOMPTARGET_NVPTX_DEBUG=0)
    endif()

    if(${LIBOMPTARGET_OMPD_SUPPORT})
      set(bc_flags ${bc_flags} -DOMPD_SUPPORT=1)
    endif()

    # Create target to build all Bitcode libraries.
    add_custom_target(omptarget-nvptx-bc)

    # Generate a Bitcode library for all the compute capabilities the user requested.
    foreach(sm ${nvptx_sm_list})
      set(cuda_arch --cuda-gpu-arch=sm_${sm})

      # Compile CUDA files to bitcode.
      set(bc_files "")
      foreach(src ${cuda_src_files})
        get_filename_component(infile ${src} ABSOLUTE)
        get_filename_component(outfile ${src} NAME)

        add_custom_command(OUTPUT ${outfile}-sm_${sm}.bc
          COMMAND ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER} ${bc_flags} ${cuda_arch} ${MAX_SM_DEFINITION}
            -c ${infile} -o ${outfile}-sm_${sm}.bc
          DEPENDS ${infile}
          IMPLICIT_DEPENDS CXX ${infile}
          COMMENT "Building LLVM bitcode ${outfile}-sm_${sm}.bc"
          VERBATIM
        )
        set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile}-sm_${sm}.bc)

        list(APPEND bc_files ${outfile}-sm_${sm}.bc)
      endforeach()

      # Link to a bitcode library.
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc
          COMMAND ${LIBOMPTARGET_NVPTX_SELECTED_BC_LINKER}
            -o ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc ${bc_files}
          DEPENDS ${bc_files}
          COMMENT "Linking LLVM bitcode libomptarget-nvptx-sm_${sm}.bc"
      )
      set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES libomptarget-nvptx-sm_${sm}.bc)

      add_custom_target(omptarget-nvptx-${sm}-bc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc)
      add_dependencies(omptarget-nvptx-bc omptarget-nvptx-${sm}-bc)

      # Copy library to destination.
      add_custom_command(TARGET omptarget-nvptx-${sm}-bc POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc
                         ${LIBOMPTARGET_LIBRARY_DIR})

      # Install bitcode library under the lib destination folder.
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc DESTINATION "${OPENMP_INSTALL_LIBDIR}")
    endforeach()
  endif()

  add_subdirectory(test)
else()
  libomptarget_say("Not building CUDA offloading device RTL: tools to build bc lib not found in the system.")
endif()
