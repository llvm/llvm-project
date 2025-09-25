set(LIBOMPTARGET_BUILD_DEVICERTL_BCLIB TRUE CACHE BOOL
  "Can be set to false to disable building this library.")

if (NOT LIBOMPTARGET_BUILD_DEVICERTL_BCLIB)
  message(STATUS "Not building DeviceRTL: Disabled by LIBOMPTARGET_BUILD_DEVICERTL_BCLIB")
  return()
endif()

# Check to ensure the host system is a supported host architecture.
if(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL "8")
  message(STATUS "Not building DeviceRTL: Runtime does not support 32-bit hosts")
  return()
endif()

if (LLVM_DIR)
  # Builds that use pre-installed LLVM have LLVM_DIR set.
  # A standalone or LLVM_ENABLE_RUNTIMES=openmp build takes this route
  find_program(CLANG_TOOL clang PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
elseif (LLVM_TOOL_CLANG_BUILD AND NOT CMAKE_CROSSCOMPILING AND NOT OPENMP_STANDALONE_BUILD)
  # LLVM in-tree builds may use CMake target names to discover the tools.
  # A LLVM_ENABLE_PROJECTS=openmp build takes this route
  set(CLANG_TOOL $<TARGET_FILE:clang>)
else()
  message(STATUS "Not building DeviceRTL. No appropriate clang found")
  return()
endif()

set(devicertl_base_directory ${CMAKE_CURRENT_SOURCE_DIR})
set(include_directory ${devicertl_base_directory}/include)
set(source_directory ${devicertl_base_directory}/src)

set(include_files
  ${include_directory}/Allocator.h
  ${include_directory}/Configuration.h
  ${include_directory}/Debug.h
  ${include_directory}/Interface.h
  ${include_directory}/LibC.h
  ${include_directory}/Mapping.h
  ${include_directory}/Profiling.h
  ${include_directory}/State.h
  ${include_directory}/Synchronization.h
  ${include_directory}/DeviceTypes.h
  ${include_directory}/DeviceUtils.h
  ${include_directory}/Workshare.h
)

set(src_files
  ${source_directory}/Allocator.cpp
  ${source_directory}/Configuration.cpp
  ${source_directory}/Debug.cpp
  ${source_directory}/Kernel.cpp
  ${source_directory}/LibC.cpp
  ${source_directory}/Mapping.cpp
  ${source_directory}/Misc.cpp
  ${source_directory}/Parallelism.cpp
  ${source_directory}/Profiling.cpp
  ${source_directory}/Reduction.cpp
  ${source_directory}/State.cpp
  ${source_directory}/Synchronization.cpp
  ${source_directory}/Tasking.cpp
  ${source_directory}/DeviceUtils.cpp
  ${source_directory}/Workshare.cpp
)

# We disable the slp vectorizer during the runtime optimization to avoid
# vectorized accesses to the shared state. Generally, those are "good" but
# the optimizer pipeline (esp. Attributor) does not fully support vectorized
# instructions yet and we end up missing out on way more important constant
# propagation. That said, we will run the vectorizer again after the runtime
# has been linked into the user program.
set(clang_opt_flags -O3 -mllvm -openmp-opt-disable -DSHARED_SCRATCHPAD_SIZE=512 -mllvm -vectorize-slp=false )

# If the user built with the GPU C library enabled we will use that instead.
if(${LIBOMPTARGET_GPU_LIBC_SUPPORT})
  list(APPEND clang_opt_flags -DOMPTARGET_HAS_LIBC)
endif()

# Set flags for LLVM Bitcode compilation.
set(bc_flags -c -flto -std=c++17 -fvisibility=hidden
             ${clang_opt_flags} -nogpulib -nostdlibinc
             -fno-rtti -fno-exceptions -fconvergent-functions
             -Wno-unknown-cuda-version
             -DOMPTARGET_DEVICE_RUNTIME
             -I${include_directory}
             -I${devicertl_base_directory}/../include
             -I${devicertl_base_directory}/../../libc
)

# first create an object target
function(compileDeviceRTLLibrary target_name target_triple)
  set(target_bc_flags ${ARGN})

  foreach(src ${src_files})
    get_filename_component(infile ${src} ABSOLUTE)
    get_filename_component(outfile ${src} NAME)
    set(outfile "${outfile}-${target_name}.o")
    set(depfile "${outfile}.d")

    # Passing an empty CPU to -march= suppressed target specific metadata.
    add_custom_command(OUTPUT ${outfile}
      COMMAND ${CLANG_TOOL}
      ${bc_flags}
      --target=${target_triple}
      ${target_bc_flags}
      -MD -MF ${depfile}
      ${infile} -o ${outfile}
      DEPENDS ${infile}
      DEPFILE ${depfile}
      COMMENT "Building LLVM bitcode ${outfile}"
      VERBATIM
    )
    if(TARGET clang)
      # Add a file-level dependency to ensure that clang is up-to-date.
      # By default, add_custom_command only builds clang if the
      # executable is missing.
      add_custom_command(OUTPUT ${outfile}
        DEPENDS clang
        APPEND
      )
    endif()
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${outfile})

    list(APPEND obj_files ${CMAKE_CURRENT_BINARY_DIR}/${outfile})
  endforeach()
  # Trick to combine these into a bitcode file via the linker's LTO pass. This
  # is used to provide the legacy `libomptarget-<name>.bc` files. Hack this
  # through as an executable to get it to use the relocatable link.
  add_executable(libomptarget-${target_name} ${obj_files})
  set_target_properties(libomptarget-${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${LIBOMPTARGET_LLVM_LIBRARY_INTDIR}
    LINKER_LANGUAGE CXX
    BUILD_RPATH ""
    INSTALL_RPATH ""
    RUNTIME_OUTPUT_NAME libomptarget-${target_name}.bc)
  target_compile_options(libomptarget-${target_name} PRIVATE
    "--target=${target_triple}" "-fuse-ld=lld" "-march=" "-mcpu="
    "-Wno-unused-command-line-argument")
  target_link_options(libomptarget-${target_name} PRIVATE "--target=${target_triple}"
                      "-r" "-nostdlib" "-flto" "-Wl,--lto-emit-llvm"
                      "-fuse-ld=lld" "-march=" "-mcpu=")
  install(TARGETS libomptarget-${target_name}
          PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
          DESTINATION "lib${LLVM_LIBDIR_SUFFIX}/${target_triple}")

  add_library(omptarget.${target_name}.all_objs OBJECT IMPORTED)
  set_property(TARGET omptarget.${target_name}.all_objs APPEND PROPERTY IMPORTED_OBJECTS
               ${LIBOMPTARGET_LLVM_LIBRARY_INTDIR}/libomptarget-${target_name}.bc)
  add_dependencies(omptarget.${target_name}.all_objs libomptarget-${target_name})

  # Archive all the object files generated above into a static library
  add_library(omptarget.${target_name} STATIC)
  set_target_properties(omptarget.${target_name} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${LIBOMPTARGET_LLVM_LIBRARY_INTDIR}/${target_triple}"
    ARCHIVE_OUTPUT_NAME ompdevice
    LINKER_LANGUAGE CXX
  )
  target_link_libraries(omptarget.${target_name} PRIVATE omptarget.${target_name}.all_objs)
  target_link_options(omptarget.${target_name} PRIVATE "--target=${target_triple}"
                      "-Wno-unused-command-line-argument" "-r" "-nostdlib" "-flto"
                       "-Wl,--lto-emit-llvm" "-fuse-ld=lld" "-march=" "-mcpu=")

  install(TARGETS omptarget.${target_name}
          ARCHIVE DESTINATION "lib${LLVM_LIBDIR_SUFFIX}/${target_triple}")

  if (CMAKE_EXPORT_COMPILE_COMMANDS)
    set(ide_target_name omptarget-ide-${target_name})
    add_library(${ide_target_name} STATIC EXCLUDE_FROM_ALL ${src_files})
    target_compile_options(${ide_target_name} PRIVATE
      -fvisibility=hidden --target=${target_triple}
      -nogpulib -nostdlibinc -Wno-unknown-cuda-version
    )
    target_compile_definitions(${ide_target_name} PRIVATE SHARED_SCRATCHPAD_SIZE=512)
    target_include_directories(${ide_target_name} PRIVATE
      ${include_directory}
      ${devicertl_base_directory}/../../libc
      ${devicertl_base_directory}/../include
    )
    install(TARGETS ${ide_target_name} EXCLUDE_FROM_ALL)
  endif()
endfunction()

if(NOT LLVM_TARGETS_TO_BUILD OR "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  compileDeviceRTLLibrary(amdgpu amdgcn-amd-amdhsa -Xclang -mcode-object-version=none)
endif()

if(NOT LLVM_TARGETS_TO_BUILD OR "NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  compileDeviceRTLLibrary(nvptx nvptx64-nvidia-cuda --cuda-feature=+ptx63)
endif()
