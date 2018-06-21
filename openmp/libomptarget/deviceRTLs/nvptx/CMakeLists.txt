##===----------------------------------------------------------------------===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is dual licensed under the MIT and the University of Illinois Open
# Source Licenses. See LICENSE.txt for details.
#
##===----------------------------------------------------------------------===##
#
# Build the NVPTX (CUDA) Device RTL if the CUDA tools are available
#
##===----------------------------------------------------------------------===##

set(LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER "" CACHE STRING
  "Path to alternate NVCC host compiler to be used by the NVPTX device RTL.")

if(LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER)
  find_program(ALTERNATE_CUDA_HOST_COMPILER NAMES ${LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER})
  if(NOT ALTERNATE_CUDA_HOST_COMPILER)
    libomptarget_say("Not building CUDA offloading device RTL: invalid NVPTX alternate host compiler.")
  endif()
  set(CUDA_HOST_COMPILER ${ALTERNATE_CUDA_HOST_COMPILER} CACHE FILEPATH "" FORCE)
endif()

# We can't use clang as nvcc host preprocessor, so we attempt to replace it with
# gcc.
if(CUDA_HOST_COMPILER MATCHES clang)

  find_program(LIBOMPTARGET_NVPTX_ALTERNATE_GCC_HOST_COMPILER NAMES gcc)

  if(NOT LIBOMPTARGET_NVPTX_ALTERNATE_GCC_HOST_COMPILER)
    libomptarget_say("Not building CUDA offloading device RTL: clang is not supported as NVCC host compiler.")
    libomptarget_say("Please include gcc in your path or set LIBOMPTARGET_NVPTX_ALTERNATE_HOST_COMPILER to the full path of of valid compiler.")
    return()
  endif()
  set(CUDA_HOST_COMPILER "${LIBOMPTARGET_NVPTX_ALTERNATE_GCC_HOST_COMPILER}" CACHE FILEPATH "" FORCE)
endif()

if(LIBOMPTARGET_DEP_CUDA_FOUND)
  libomptarget_say("Building CUDA offloading device RTL.")

  # We really don't have any host code, so we don't need to care about
  # propagating host flags.
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  set(cuda_src_files
      src/cancel.cu
      src/critical.cu
      src/data_sharing.cu
      src/libcall.cu
      src/loop.cu
      src/omptarget-nvptx.cu
      src/parallel.cu
      src/reduction.cu
      src/sync.cu
      src/task.cu
  )

  set(omp_data_objects src/omp_data.cu)

  # Get the compute capability the user requested or use SM_35 by default.
  # SM_35 is what clang uses by default.
  set(default_capabilities 35)
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

  # Activate RTL message dumps if requested by the user.
  set(LIBOMPTARGET_NVPTX_DEBUG FALSE CACHE BOOL
    "Activate NVPTX device RTL debug messages.")
  if(${LIBOMPTARGET_NVPTX_DEBUG})
    set(CUDA_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1 -g --ptxas-options=-v)
  endif()

  # NVPTX runtime library has to be statically linked. Dynamic linking is not
  # yet supported by the CUDA toolchain on the device.
  set(BUILD_SHARED_LIBS OFF)
  set(CUDA_SEPARABLE_COMPILATION ON)

  cuda_add_library(omptarget-nvptx STATIC ${cuda_src_files} ${omp_data_objects}
      OPTIONS ${CUDA_ARCH} ${CUDA_DEBUG})

  # Install device RTL under the lib destination folder.
  install(TARGETS omptarget-nvptx ARCHIVE DESTINATION "${OPENMP_INSTALL_LIBDIR}")

  target_link_libraries(omptarget-nvptx ${CUDA_LIBRARIES})


  # Check if we can create an LLVM bitcode implementation of the runtime library
  # that could be inlined in the user application. For that we need to find
  # a Clang compiler capable of compiling our CUDA files to LLVM bitcode and
  # an LLVM linker.
  set(LIBOMPTARGET_NVPTX_CUDA_COMPILER "" CACHE STRING
    "Location of a CUDA compiler capable of emitting LLVM bitcode.")
  set(LIBOMPTARGET_NVPTX_BC_LINKER "" CACHE STRING
    "Location of a linker capable of linking LLVM bitcode objects.")

  include(LibomptargetNVPTXBitcodeLibrary)

  set(bclib_default FALSE)
  if (${LIBOMPTARGET_NVPTX_BCLIB_SUPPORTED})
    set(bclib_default TRUE)
  endif()
  set(LIBOMPTARGET_NVPTX_ENABLE_BCLIB ${bclib_default} CACHE BOOL
    "Enable CUDA LLVM bitcode offloading device RTL.")
  if (${LIBOMPTARGET_NVPTX_ENABLE_BCLIB})
    if (NOT ${LIBOMPTARGET_NVPTX_BCLIB_SUPPORTED})
      libomptarget_error_say("Cannot build CUDA LLVM bitcode offloading device RTL!")
    endif()
    libomptarget_say("Building CUDA LLVM bitcode offloading device RTL.")

    # Set flags for LLVM Bitcode compilation.
    set(bc_flags ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER_FLAGS} -DOMPTARGET_NVPTX_TEST=0)
    if(${LIBOMPTARGET_NVPTX_DEBUG})
      set(bc_flags ${bc_flags} -DOMPTARGET_NVPTX_DEBUG=-1)
    else()
      set(bc_flags ${bc_flags} -DOMPTARGET_NVPTX_DEBUG=0)
    endif()

    # CUDA 9 header files use the nv_weak attribute which clang is not yet prepared
    # to handle. Therefore, we use 'weak' instead. We are compiling only for the
    # device, so it should be equivalent.
    if(CUDA_VERSION_MAJOR GREATER 8)
      set(bc_flags ${bc_flags} -Dnv_weak=weak)
    endif()

    # Generate a Bitcode library for all the compute capabilities the user requested.
    foreach(sm ${nvptx_sm_list})
      set(cuda_arch --cuda-gpu-arch=sm_${sm})

      # Compile CUDA files to bitcode.
      set(bc_files "")
      foreach(src ${cuda_src_files})
        get_filename_component(infile ${src} ABSOLUTE)
        get_filename_component(outfile ${src} NAME)

        add_custom_command(OUTPUT ${outfile}-sm_${sm}.bc
          COMMAND ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER} ${bc_flags} ${cuda_arch}
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

      # Copy library to destination.
      add_custom_command(TARGET omptarget-nvptx-${sm}-bc POST_BUILD
                         COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc
                         $<TARGET_FILE_DIR:omptarget-nvptx>)

      # Install bitcode library under the lib destination folder.
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc DESTINATION "${OPENMP_INSTALL_LIBDIR}")
    endforeach()
  endif()

else()
  libomptarget_say("Not building CUDA offloading device RTL: CUDA tools not found in the system.")
endif()
