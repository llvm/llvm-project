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

cmake_path(GET LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY FILENAME LEVEL_ZERO_LIBRARY_NAME)
if (EXISTS ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY} AND NOT "level_zero" IN_LIST LIBOMPTARGET_DLOPEN_PLUGINS)
  message(STATUS "Building Level Zero NG plugin linked against level_zero library")
 if(UNIX)
    target_link_libraries(omptarget.rtl.level_zero PRIVATE
        ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY})
  elseif(WIN32)
    # Full path to the Level Zero library is recognized as a linker option, so we
    # separate directory and file name
    cmake_path(GET LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARY PARENT_PATH LEVEL_ZERO_LIBRARY_PATH)
    target_link_libraries(omptarget.rtl.level_zero PRIVATE ${LEVEL_ZERO_LIBRARY_NAME}
        ${LIBOMP_LIB_FILE})
    target_link_directories(omptarget.rtl.level_zero PRIVATE ${LEVEL_ZERO_LIBRARY_PATH})
  else()
    message(FATAL_ERROR "Missing platform support")
  endif()
else()
  message(STATUS "Building Level Zero NG plugin for dlopened level_zero")
  if(WIN32)
    cmake_path(REPLACE_EXTENSION LEVEL_ZERO_LIBRARY_NAME dll)
  endif()
  target_compile_definitions(omptarget.rtl.level_zero PRIVATE
      LEVEL_ZERO_LIBRARY="${LEVEL_ZERO_LIBRARY_NAME}")
  target_sources(omptarget.rtl.level_zero PRIVATE src/L0DynWrapper.cpp)
endif()
