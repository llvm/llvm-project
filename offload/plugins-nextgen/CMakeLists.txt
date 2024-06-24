# Common interface to handle creating a plugin library.
set(common_dir ${CMAKE_CURRENT_SOURCE_DIR}/common)
add_subdirectory(common)
function(add_target_library target_name lib_name)
  add_llvm_library(${target_name} STATIC
    LINK_COMPONENTS
      AggressiveInstCombine
      Analysis
      BinaryFormat
      BitReader
      BitWriter
      CodeGen
      Core
      Extensions
      InstCombine
      Instrumentation
      IPO
      IRReader
      Linker
      MC
      Object
      Passes
      Remarks
      ScalarOpts
      Support
      Target
      TargetParser
      TransformUtils
      Vectorize

    NO_INSTALL_RPATH
    BUILDTREE_ONLY
  )

  llvm_update_compile_flags(${target_name})
  target_include_directories(${target_name} PUBLIC ${common_dir}/include)
  target_link_libraries(${target_name} PRIVATE
                        PluginCommon ${OPENMP_PTHREAD_LIB})

  target_compile_definitions(${target_name} PRIVATE TARGET_NAME=${lib_name})
  target_compile_definitions(${target_name} PRIVATE
                             DEBUG_PREFIX="TARGET ${lib_name} RTL")
  set_target_properties(${target_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${plugin})
    message(FATAL_ERROR "Unknown plugin target '${plugin}'")
  endif()
  add_subdirectory(${plugin})
endforeach()

# Make sure the parent scope can see the plugins that will be created.
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
