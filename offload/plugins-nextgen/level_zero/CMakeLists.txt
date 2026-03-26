# Create the library and add the default arguments.
add_target_library(omptarget.rtl.level_zero LEVEL_ZERO)

set(LEVEL_ZERO_SRC_FILES
  src/L0Context.cpp
  src/L0Device.cpp
  src/L0Kernel.cpp
  src/L0Memory.cpp
  src/L0Program.cpp
  src/L0Plugin.cpp
  src/L0Program.cpp
  src/L0Options.cpp
)

target_sources(omptarget.rtl.level_zero PRIVATE
  ${LEVEL_ZERO_SRC_FILES}
)

target_include_directories(omptarget.rtl.level_zero PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(omptarget.rtl.level_zero PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIR}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_OMP_HEADER_DIR}
)

if (EXISTS ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY} AND NOT "level_zero" IN_LIST LIBOMPTARGET_DLOPEN_PLUGINS)
  message(STATUS "Building Level Zero NG plugin linked against level_zero library")
 if(UNIX)
    target_link_libraries(omptarget.rtl.level_zero PRIVATE
        ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY})
  elseif(WIN32)
    # Full path to the Level Zero library is recognized as a linker option, so we
    # separate directory and file name
    cmake_path(GET LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY FILENAME LEVEL_ZERO_LIBRARY_NAME)
    cmake_path(GET LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY PARENT_PATH LEVEL_ZERO_LIBRARY_PATH)
    target_link_libraries(omptarget.rtl.level_zero PRIVATE ${LEVEL_ZERO_LIBRARY_NAME}
        ${LIBOMP_LIB_FILE})
    target_link_directories(omptarget.rtl.level_zero PRIVATE ${LEVEL_ZERO_LIBRARY_PATH})
  else()
    message(FATAL_ERROR "Missing platform support")
  endif()
else()
  message(STATUS "Building Level Zero NG plugin for dlopened level_zero")
  target_sources(omptarget.rtl.level_zero PRIVATE src/L0DynWrapper.cpp)
endif()

option(LIBOMPTARGET_FORCE_LEVELZERO_TESTS "Build Level Zero libomptarget tests" OFF)
if (LIBOMPTARGET_FOUND_INTELGPU_GPU OR LIBOMPTARGET_FORCE_LEVELZERO_TESTS)
  # Report to the parent scope that we are building a plugin for intelgpu
  set(LIBOMPTARGET_SYSTEM_TARGETS
      "${LIBOMPTARGET_SYSTEM_TARGETS} spirv64-intel" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.level_zero")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  message(STATUS "Not generating Level Zero tests, no supported devices detected."
                   " Use 'LIBOMPTARGET_FORCE_INTELGPU_TESTS' to override.")
endif()
