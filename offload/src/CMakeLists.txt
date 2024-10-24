message(STATUS "Building offloading runtime library libomptarget.")

if(LIBOMP_STANDALONE)
  set(LIBOMP ${LIBOMP_STANDALONE})
else()
  set(LIBOMP omp)
endif()

add_llvm_library(omptarget
  SHARED

  device.cpp
  interface.cpp
  omptarget.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp

  KernelLanguage/API.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  Object

  LINK_LIBS
  PUBLIC
  ${LIBOMP}

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(omptarget PRIVATE
  TARGET_NAME=omptarget
  DEBUG_PREFIX="omptarget"
)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  target_link_libraries(omptarget PRIVATE omptarget.rtl.${plugin})
endforeach()

target_compile_options(omptarget PRIVATE ${offload_compile_flags})
target_link_options(omptarget PRIVATE ${offload_link_flags})

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

#===============================================================================
# Ensure that omptarget does not contain a mixture of static and dynamically
# linked LLVM libs.
#===============================================================================
if (LLVM_LINK_LLVM_DYLIB)
  if(LLVM_AVAILABLE_LIBS)
    set(llvm_libs ${LLVM_AVAILABLE_LIBS})
  else()
    # Inside LLVM itself available libs are in a global property.
    get_property(llvm_libs GLOBAL PROPERTY LLVM_LIBS)
  endif()

  #-----------------------------------------------------------------------------
  # Helper function to recursively get the llvm targets that 'tgt' links against
  #-----------------------------------------------------------------------------
  function(get_llvm_link_targets var tgt visited)
    list(APPEND visited ${tgt})

    get_target_property(link_libs ${tgt} LINK_LIBRARIES)
    if(NOT link_libs)
      set(link_libs "")
    endif()
    get_target_property(i_link_libs ${tgt} INTERFACE_LINK_LIBRARIES)
    if(i_link_libs)
      list(APPEND link_libs ${i_link_libs})
    endif()
    if(NOT link_libs)
      return()
    endif()
    list(REMOVE_DUPLICATES link_libs)
    list(REMOVE_ITEM link_libs ${visited})

    foreach(lib ${link_libs})
      if(${lib} IN_LIST llvm_libs)
        list(APPEND rv ${lib})
      endif()
      if(TARGET ${lib})
        get_llvm_link_targets(indirect ${lib} ${visited})
        list(APPEND rv ${indirect})
      endif()
      list(REMOVE_DUPLICATES rv)
    endforeach()

    set(visited ${visited} PARENT_SCOPE)
    set(${var} ${rv} PARENT_SCOPE)
  endfunction()

  #-----------------------------------------------------------------------------
  # Check for extraneous libs
  #-----------------------------------------------------------------------------
  get_llvm_link_targets(llvm_link_targets omptarget "" 0)
  list(REMOVE_ITEM llvm_link_targets "LLVM")
  if(llvm_link_targets)
    list(JOIN llvm_link_targets " " pp_list)
    message(
      FATAL_ERROR
      "'omptarget' should only link against 'LLVM' when 'LLVM_LINK_LLVM_DYLIB' "
      "is on. Extraneous LLVM Libraries: ${pp_list}")
  endif()
endif()
