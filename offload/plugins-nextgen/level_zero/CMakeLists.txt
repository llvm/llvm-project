if(NOT LIBOMPTARGET_DEP_LEVEL_ZERO_FOUND)
  return()
endif()

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
list(APPEND LEVEL_ZERO_SRC_FILES
  src/OmpWrapper.cpp
)

target_sources(omptarget.rtl.level_zero PRIVATE
  ${LEVEL_ZERO_SRC_FILES}
)

target_include_directories(omptarget.rtl.level_zero PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(omptarget.rtl.level_zero PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_DEP_LEVEL_ZERO_INCLUDE_DIRS}
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_OMP_HEADER_DIR}
)

if (EXISTS ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARIES} AND NOT "level_zero" IN_LIST LIBOMPTARGET_DLOPEN_PLUGINS)
  message(STATUS "Building Level Zero NG plugin linked against level_zero library")

  if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(omptarget.rtl.level_zero PRIVATE
                        ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARIES})
  elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # Full path to the L0 library is recognized as a linker option, so we
    # separate directory and file name
    get_filename_component(LEVEL_ZERO_LIBRARY_PATH
                           ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARIES} DIRECTORY)
    get_filename_component(LEVEL_ZERO_LIBRARY_NAME
                           ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARIES} NAME)
    target_link_libraries(omptarget.rtl.level_zero PRIVATE
                          ${LEVEL_ZERO_LIBRARY_NAME} ${LIBOMP_LIB_FILE})
    target_link_directories(omptarget.rtl.level_zero PRIVATE 
                            ${LEVEL_ZERO_LIBRARY_PATH})
    target_link_options(omptarget.rtl.level_zero PRIVATE 
                        "LINKER:-def:${CMAKE_CURRENT_SOURCE_DIR}/src/rtl.def")
    libomptarget_add_resource_file(omptarget.rtl.level_zero)
  else()
    message(FATAL_ERROR "Missing platform support")
  endif()
else()
  message(STATUS "Building Level Zero NG plugin for dlopened level_zero")
  get_filename_component(LEVEL_ZERO_LIBRARY_NAME 
                         ${LIBOMPTARGET_DEP_LEVEL_ZERO_LIBRARIES} NAME)
  if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    # Windows uses dll instead of lib files at runtime
    string(REGEX REPLACE "lib$" "dll" LEVEL_ZERO_LIBRARY_NAME 
           ${LEVEL_ZERO_LIBRARY_NAME})
  endif()
  target_compile_options(omptarget.rtl.level_zero PRIVATE 
                         "-DLEVEL_ZERO_LIBRARY=\"${LEVEL_ZERO_LIBRARY_NAME}\"")
  target_sources(omptarget.rtl.level_zero PRIVATE src/L0DynWrapper.cpp)
endif()
