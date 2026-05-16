message(STATUS "Building support library for OpenMP and OpenACC.")

set(OMPACCSUPPORT_SRC
  OMPT/Callback.cpp
  Mapping.cpp
  PluginManager.cpp
  DeviceImage.cpp
  OffloadRTL.cpp
  device.cpp
  interface.cpp
)

add_llvm_library(ompaccsupport
  SHARED

  ${OMPACCSUPPORT_SRC}

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  Support

  LINK_LIBS
  PUBLIC

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)

target_include_directories(ompaccsupport PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  # TODO the link script currently exports _all_ symbols. We want to limit this
  # to what we only need so as not to litter with symbols that can potentially
  # conflict with user code. A good way to achieve this is to wrap everything we
  # need in a namespace (e.g. llvm::offload::ompaccsupport) and export
  # everything with the mangled namespace prefix.
  target_link_libraries(ompaccsupport PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
  set_property(TARGET ompaccsupport APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports)
endif()

target_compile_definitions(ompaccsupport PRIVATE
  TARGET_NAME=ompaccsupport
  DEBUG_PREFIX="ompaccsupport"
)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  target_link_libraries(ompaccsupport PUBLIC omptarget.rtl.${plugin})
endforeach()

target_compile_options(ompaccsupport PRIVATE ${offload_compile_flags})
target_link_options(ompaccsupport PRIVATE ${offload_link_flags})

set_target_properties(ompaccsupport PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS ompaccsupport LIBRARY COMPONENT ompaccsupport DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
