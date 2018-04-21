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
  install(TARGETS omptarget-nvptx ARCHIVE DESTINATION "lib")

  target_link_libraries(omptarget-nvptx ${CUDA_LIBRARIES})

  # Check if we can create an LLVM bitcode implementation of the runtime library
  # that could be inlined in the user implementation.
  set(LIBOMPTARGET_NVPTX_ENABLE_BCLIB FALSE CACHE BOOL
    "Enable CUDA LLVM bitcode offloading device RTL.")
  if (${LIBOMPTARGET_NVPTX_ENABLE_BCLIB})

    # Find a clang compiler capable of compiling cuda files to LLVM bitcode and
    # an LLVM linker.
    # We use the one provided by the user, attempt to use the one used to build
    # libomptarget or just fail.

    set(LIBOMPTARGET_NVPTX_CUDA_COMPILER "" CACHE STRING
      "Location of a CUDA compiler capable of emitting LLVM bitcode.")
    set(LIBOMPTARGET_NVPTX_BC_LINKER "" CACHE STRING
      "Location of a linker capable of linking LLVM bitcode objects.")

    if (NOT LIBOMPTARGET_NVPTX_CUDA_COMPILER STREQUAL "")
      set(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER ${LIBOMPTARGET_NVPTX_CUDA_COMPILER})
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
      set(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER ${CMAKE_C_COMPILER})
    else()
      libomptarget_error_say("Cannot find a CUDA compiler capable of emitting LLVM bitcode.")
      libomptarget_error_say("Please configure with flag -DLIBOMPTARGET_NVPTX_CUDA_COMPILER")
    endif()

    # Get compiler directory to try to locate a suitable linker
    get_filename_component(COMPILER_DIR ${CMAKE_C_COMPILER} DIRECTORY)
    
    if (NOT LIBOMPTARGET_NVPTX_BC_LINKER STREQUAL "")
      set(LIBOMPTARGET_NVPTX_SELECTED_BC_LINKER ${LIBOMPTARGET_NVPTX_BC_LINKER})
    elseif(${CMAKE_C_COMPILER_ID} STREQUAL "Clang" AND EXISTS "${COMPILER_DIR}/llvm-link")
      # Use llvm-link from the directory containing clang
      set(LIBOMPTARGET_NVPTX_SELECTED_BC_LINKER ${COMPILER_DIR}/llvm-link)
    else()
      libomptarget_error_say("Cannot find a linker capable of linking LLVM bitcode objects.")
      libomptarget_error_say("Please configure with flag -DLIBOMPTARGET_NVPTX_BC_LINKER")
    endif()

    if(LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER AND LIBOMPTARGET_NVPTX_SELECTED_BC_LINKER)
      libomptarget_say("Building CUDA LLVM bitcode offloading device RTL.")

      # Decide which ptx version to use. Same choices as Clang.
      if(CUDA_VERSION_MAJOR GREATER 9 OR CUDA_VERSION_MAJOR EQUAL 9)
        set(CUDA_PTX_VERSION ptx60)
      else()
        set(CUDA_PTX_VERSION ptx42)
      endif()
 
      set(BC_DEBUG -DOMPTARGET_NVPTX_DEBUG=0)
      if(${LIBOMPTARGET_NVPTX_DEBUG})
        set(BC_DEBUG -DOMPTARGET_NVPTX_DEBUG=-1)
      endif()

      # Set flags for Clang cuda compilation.  Only Clang is supported because there is
      # no other compiler capable of generating bitcode from cuda sources.
      set(CUDA_FLAGS
          -emit-llvm
          -O1
          -Xclang -target-feature
          -Xclang +${CUDA_PTX_VERSION}
          --cuda-device-only
          -DOMPTARGET_NVPTX_TEST=0
          ${BC_DEBUG}
      )

      # CUDA 9 header files use the nv_weak attribute which clang is not yet prepared
      # to handle. Therefore, we use 'weak' instead. We are compiling only for the
      # device, so it should be equivalent.
      if(CUDA_VERSION_MAJOR EQUAL 9)
        set(CUDA_FLAGS ${CUDA_FLAGS} -Dnv_weak=weak)
      endif()

      # Get the compute capability the user requested or use SM_35 by default.
      set(CUDA_ARCH "")
      foreach(sm ${nvptx_sm_list})
        set(CUDA_ARCH --cuda-gpu-arch=sm_${sm})

        # Compile cuda files to bitcode.
        set(bc_files "")
        foreach(src ${cuda_src_files})
          get_filename_component(infile ${src} ABSOLUTE)
          get_filename_component(outfile ${src} NAME)

          add_custom_command(OUTPUT ${outfile}-sm_${sm}.bc
            COMMAND ${LIBOMPTARGET_NVPTX_SELECTED_CUDA_COMPILER} ${CUDA_FLAGS} ${CUDA_ARCH} ${CUDA_INCLUDES}
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

        # Install device RTL under the lib destination folder.
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libomptarget-nvptx-sm_${sm}.bc DESTINATION "lib")
      endforeach()
    endif()
  endif()

else()
  libomptarget_say("Not building CUDA offloading device RTL: CUDA tools not found in the system.")
endif()
