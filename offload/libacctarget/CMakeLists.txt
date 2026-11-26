message(STATUS "Building OpenACC offloading runtime library libacctarget.")

remove_definitions(-DOMPT_SUPPORT=1)

add_subdirectory(API)

add_library(acctarget SHARED
  AccAPI.cpp
  AccAPIImpl.cpp
  DeviceManager.cpp
  Interface.cpp
  RuntimeState.cpp

  ${LIBOMPACCSUPPORT_SOURCES}
)
add_dependencies(acctarget AccAPITableGen)

if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs FrontendOpenMP Support)
endif()
target_link_libraries(acctarget PRIVATE ${llvm_libs})
target_include_directories(acctarget PRIVATE
  ${LIBOMPACCSUPPORT_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/API
  ${CMAKE_CURRENT_SOURCE_DIR}/../../flang/include/flang
)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(acctarget PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_BINARY_DIR}/API/acc.exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(acctarget PRIVATE
  TARGET_NAME=acctarget
  DEBUG_PREFIX="acctarget"
)


target_link_libraries(acctarget PRIVATE LLVMOffloadStatic)

target_compile_options(acctarget PRIVATE ${offload_compile_flags})
target_link_options(acctarget PRIVATE ${offload_link_flags})

# libacctarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(acctarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "${OFFLOAD_INSTALL_RPATH}"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
if(UNIX AND NOT APPLE)
  set_target_properties(acctarget PROPERTIES
                        SOVERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}"
                        VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}")
endif()
install(TARGETS acctarget LIBRARY COMPONENT offload DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/openacc.h
               ${CMAKE_CURRENT_BINARY_DIR}/API/openacc.h COPYONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/API/openacc.h
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT acctarget-headers)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/API/openacc/AccCEnums.inc
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/openacc"
  COMPONENT acctarget-headers)
add_custom_target(install-acctarget-headers
  DEPENDS AccAPITableGen
  COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --component acctarget-headers)

# The library is also installable on its own, without the rest of offload.
install(TARGETS acctarget LIBRARY COMPONENT acctarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
add_custom_target(install-acctarget
  DEPENDS acctarget
  COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --component acctarget)
add_custom_target(install-acctarget-stripped
  DEPENDS acctarget
  COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --component acctarget --strip)

# Build the openacc module from the generated Fortran interfaces so Flang finds
# it next to the other intrinsic modules.
if(RUNTIMES_FORTRAN_MODULES)
  add_library(acctarget-mod OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/include/openacc.f90)
  add_dependencies(acctarget-mod AccAPITableGen)
  target_include_directories(acctarget-mod PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/API)
  set_target_properties(acctarget-mod PROPERTIES FOLDER "OpenACC/Fortran Modules")
  flang_module_target(acctarget-mod PUBLIC)
  add_dependencies(acctarget acctarget-mod)

  install(FILES "${RUNTIMES_OUTPUT_RESOURCE_MOD_DIR}/openacc.mod"
    DESTINATION "${RUNTIMES_INSTALL_RESOURCE_MOD_PATH}"
    COMPONENT acctarget-mod
    EXCLUDE_FROM_ALL)

  if(NOT CMAKE_CONFIGURATION_TYPES)
    add_custom_target(install-acctarget-mod
      DEPENDS acctarget-mod
      COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
              --component acctarget-mod)
    add_custom_target(install-acctarget-mod-stripped
      DEPENDS acctarget-mod
      COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}"
              --component acctarget-mod --strip)
  endif()
endif()
