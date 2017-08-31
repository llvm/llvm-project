include(LLVMProcessSources)
include(LLVM-Config)
include(DetermineGCCCompatible)

function(llvm_update_compile_flags name)
  get_property(sources TARGET ${name} PROPERTY SOURCES)
  if("${sources}" MATCHES "\\.c(;|$)")
    set(update_src_props ON)
  endif()

  # LLVM_REQUIRES_EH is an internal flag that individual targets can use to
  # force EH
  if(LLVM_REQUIRES_EH OR LLVM_ENABLE_EH)
    if(NOT (LLVM_REQUIRES_RTTI OR LLVM_ENABLE_RTTI))
      message(AUTHOR_WARNING "Exception handling requires RTTI. Enabling RTTI for ${name}")
      set(LLVM_REQUIRES_RTTI ON)
    endif()
    if(MSVC)
      list(APPEND LLVM_COMPILE_FLAGS "/EHsc")
    endif()
  else()
    if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
      list(APPEND LLVM_COMPILE_FLAGS "-fno-exceptions")
    elseif(MSVC)
      list(APPEND LLVM_COMPILE_DEFINITIONS _HAS_EXCEPTIONS=0)
      list(APPEND LLVM_COMPILE_FLAGS "/EHs-c-")
    endif()
  endif()

  # LLVM_REQUIRES_RTTI is an internal flag that individual
  # targets can use to force RTTI
  set(LLVM_CONFIG_HAS_RTTI YES CACHE INTERNAL "")
  if(NOT (LLVM_REQUIRES_RTTI OR LLVM_ENABLE_RTTI))
    set(LLVM_CONFIG_HAS_RTTI NO CACHE INTERNAL "")
    list(APPEND LLVM_COMPILE_DEFINITIONS GTEST_HAS_RTTI=0)
    if (LLVM_COMPILER_IS_GCC_COMPATIBLE)
      list(APPEND LLVM_COMPILE_FLAGS "-fno-rtti")
    elseif (MSVC)
      list(APPEND LLVM_COMPILE_FLAGS "/GR-")
    endif ()
  elseif(MSVC)
    list(APPEND LLVM_COMPILE_FLAGS "/GR")
  endif()

  # Assume that;
  #   - LLVM_COMPILE_FLAGS is list.
  #   - PROPERTY COMPILE_FLAGS is string.
  string(REPLACE ";" " " target_compile_flags " ${LLVM_COMPILE_FLAGS}")

  if(update_src_props)
    foreach(fn ${sources})
      get_filename_component(suf ${fn} EXT)
      if("${suf}" STREQUAL ".cpp")
        set_property(SOURCE ${fn} APPEND_STRING PROPERTY
          COMPILE_FLAGS "${target_compile_flags}")
      endif()
    endforeach()
  else()
    # Update target props, since all sources are C++.
    set_property(TARGET ${name} APPEND_STRING PROPERTY
      COMPILE_FLAGS "${target_compile_flags}")
  endif()

  set_property(TARGET ${name} APPEND PROPERTY COMPILE_DEFINITIONS ${LLVM_COMPILE_DEFINITIONS})
endfunction()

function(add_llvm_symbol_exports target_name export_file)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(native_export_file "${target_name}.exports")
    add_custom_command(OUTPUT ${native_export_file}
      COMMAND sed -e "s/^/_/" < ${export_file} > ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                 LINK_FLAGS " -Wl,-exported_symbols_list,${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
  elseif(${CMAKE_SYSTEM_NAME} MATCHES "AIX")
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                 LINK_FLAGS " -Wl,-bE:${export_file}")
  elseif(LLVM_HAVE_LINK_VERSION_SCRIPT)
    # Gold and BFD ld require a version script rather than a plain list.
    set(native_export_file "${target_name}.exports")
    # FIXME: Don't write the "local:" line on OpenBSD.
    # in the export file, also add a linker script to version LLVM symbols (form: LLVM_N.M)
    add_custom_command(OUTPUT ${native_export_file}
      COMMAND echo "LLVM_${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR} {" > ${native_export_file}
      COMMAND grep -q "[[:alnum:]]" ${export_file} && echo "  global:" >> ${native_export_file} || :
      COMMAND sed -e "s/$/;/" -e "s/^/    /" < ${export_file} >> ${native_export_file}
      COMMAND echo "  local: *;" >> ${native_export_file}
      COMMAND echo "};" >> ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    if (${LLVM_LINKER_IS_SOLARISLD})
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS "  -Wl,-M,${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
    else()
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS "  -Wl,--version-script,${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
    endif()
  else()
    set(native_export_file "${target_name}.def")

    add_custom_command(OUTPUT ${native_export_file}
      COMMAND ${PYTHON_EXECUTABLE} -c "import sys;print(''.join(['EXPORTS\\n']+sys.stdin.readlines(),))"
        < ${export_file} > ${native_export_file}
      DEPENDS ${export_file}
      VERBATIM
      COMMENT "Creating export file for ${target_name}")
    set(export_file_linker_flag "${CMAKE_CURRENT_BINARY_DIR}/${native_export_file}")
    if(MSVC)
      set(export_file_linker_flag "/DEF:\"${export_file_linker_flag}\"")
    endif()
    set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                 LINK_FLAGS " ${export_file_linker_flag}")
  endif()

  add_custom_target(${target_name}_exports DEPENDS ${native_export_file})
  set_target_properties(${target_name}_exports PROPERTIES FOLDER "Misc")

  get_property(srcs TARGET ${target_name} PROPERTY SOURCES)
  foreach(src ${srcs})
    get_filename_component(extension ${src} EXT)
    if(extension STREQUAL ".cpp")
      set(first_source_file ${src})
      break()
    endif()
  endforeach()

  # Force re-linking when the exports file changes. Actually, it
  # forces recompilation of the source file. The LINK_DEPENDS target
  # property only works for makefile-based generators.
  # FIXME: This is not safe because this will create the same target
  # ${native_export_file} in several different file:
  # - One where we emitted ${target_name}_exports
  # - One where we emitted the build command for the following object.
  # set_property(SOURCE ${first_source_file} APPEND PROPERTY
  #   OBJECT_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${native_export_file})

  set_property(DIRECTORY APPEND
    PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${native_export_file})

  add_dependencies(${target_name} ${target_name}_exports)

  # Add dependency to *_exports later -- CMake issue 14747
  list(APPEND LLVM_COMMON_DEPENDS ${target_name}_exports)
  set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS} PARENT_SCOPE)
endfunction(add_llvm_symbol_exports)

if(NOT WIN32 AND NOT APPLE)
  # Detect what linker we have here
  execute_process(
    COMMAND ${CMAKE_C_COMPILER} -Wl,--version
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    )
  set(LLVM_LINKER_DETECTED ON)
  if("${stdout}" MATCHES "GNU gold")
    set(LLVM_LINKER_IS_GOLD ON)
    message(STATUS "Linker detection: GNU Gold")
  elseif("${stdout}" MATCHES "^LLD")
    set(LLVM_LINKER_IS_LLD ON)
    message(STATUS "Linker detection: LLD")
  elseif("${stdout}" MATCHES "GNU ld")
    set(LLVM_LINKER_IS_GNULD ON)
    message(STATUS "Linker detection: GNU ld")
  elseif("${stderr}" MATCHES "Solaris Link Editors")
    set(LLVM_LINKER_IS_SOLARISLD ON)
    message(STATUS "Linker detection: Solaris ld")
  else()
    set(LLVM_LINKER_DETECTED OFF)
    message(STATUS "Linker detection: unknown")
  endif()
endif()

function(add_link_opts target_name)
  # Don't use linker optimizations in debug builds since it slows down the
  # linker in a context where the optimizations are not important.
  if (NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")

    # Pass -O3 to the linker. This enabled different optimizations on different
    # linkers.
    if(NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin|SunOS|AIX" OR WIN32))
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS " -Wl,-O3")
    endif()

    if(LLVM_LINKER_IS_GOLD)
      # With gold gc-sections is always safe.
      set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                   LINK_FLAGS " -Wl,--gc-sections")
      # Note that there is a bug with -Wl,--icf=safe so it is not safe
      # to enable. See https://sourceware.org/bugzilla/show_bug.cgi?id=17704.
    endif()

    if(NOT LLVM_NO_DEAD_STRIP)
      if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        # ld64's implementation of -dead_strip breaks tools that use plugins.
        set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                     LINK_FLAGS " -Wl,-dead_strip")
      elseif(${CMAKE_SYSTEM_NAME} MATCHES "SunOS")
        set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                     LINK_FLAGS " -Wl,-z -Wl,discard-unused=sections")
      elseif(NOT WIN32 AND NOT LLVM_LINKER_IS_GOLD)
        # Object files are compiled with -ffunction-data-sections.
        # Versions of bfd ld < 2.23.1 have a bug in --gc-sections that breaks
        # tools that use plugins. Always pass --gc-sections once we require
        # a newer linker.
        set_property(TARGET ${target_name} APPEND_STRING PROPERTY
                     LINK_FLAGS " -Wl,--gc-sections")
      endif()
    endif()
  endif()
endfunction(add_link_opts)

# Set each output directory according to ${CMAKE_CONFIGURATION_TYPES}.
# Note: Don't set variables CMAKE_*_OUTPUT_DIRECTORY any more,
# or a certain builder, for eaxample, msbuild.exe, would be confused.
function(set_output_directory target)
  cmake_parse_arguments(ARG "" "BINARY_DIR;LIBRARY_DIR" "" ${ARGN})

  # module_dir -- corresponding to LIBRARY_OUTPUT_DIRECTORY.
  # It affects output of add_library(MODULE).
  if(WIN32 OR CYGWIN)
    # DLL platform
    set(module_dir ${ARG_BINARY_DIR})
  else()
    set(module_dir ${ARG_LIBRARY_DIR})
  endif()
  if(NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
    foreach(build_mode ${CMAKE_CONFIGURATION_TYPES})
      string(TOUPPER "${build_mode}" CONFIG_SUFFIX)
      if(ARG_BINARY_DIR)
        string(REPLACE ${CMAKE_CFG_INTDIR} ${build_mode} bi ${ARG_BINARY_DIR})
        set_target_properties(${target} PROPERTIES "RUNTIME_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${bi})
      endif()
      if(ARG_LIBRARY_DIR)
        string(REPLACE ${CMAKE_CFG_INTDIR} ${build_mode} li ${ARG_LIBRARY_DIR})
        set_target_properties(${target} PROPERTIES "ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${li})
      endif()
      if(module_dir)
        string(REPLACE ${CMAKE_CFG_INTDIR} ${build_mode} mi ${module_dir})
        set_target_properties(${target} PROPERTIES "LIBRARY_OUTPUT_DIRECTORY_${CONFIG_SUFFIX}" ${mi})
      endif()
    endforeach()
  else()
    if(ARG_BINARY_DIR)
      set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ARG_BINARY_DIR})
    endif()
    if(ARG_LIBRARY_DIR)
      set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ARG_LIBRARY_DIR})
    endif()
    if(module_dir)
      set_target_properties(${target} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${module_dir})
    endif()
  endif()
endfunction()

# If on Windows and building with MSVC, add the resource script containing the
# VERSIONINFO data to the project.  This embeds version resource information
# into the output .exe or .dll.
# TODO: Enable for MinGW Windows builds too.
#
function(add_windows_version_resource_file OUT_VAR)
  set(sources ${ARGN})
  if (MSVC)
    set(resource_file ${LLVM_SOURCE_DIR}/resources/windows_version_resource.rc)
    if(EXISTS ${resource_file})
      set(sources ${sources} ${resource_file})
      source_group("Resource Files" ${resource_file})
      set(windows_resource_file ${resource_file} PARENT_SCOPE)
    endif()
  endif(MSVC)

  set(${OUT_VAR} ${sources} PARENT_SCOPE)
endfunction(add_windows_version_resource_file)

# set_windows_version_resource_properties(name resource_file...
#   VERSION_MAJOR int
#     Optional major version number (defaults to LLVM_VERSION_MAJOR)
#   VERSION_MINOR int
#     Optional minor version number (defaults to LLVM_VERSION_MINOR)
#   VERSION_PATCHLEVEL int
#     Optional patchlevel version number (defaults to LLVM_VERSION_PATCH)
#   VERSION_STRING
#     Optional version string (defaults to PACKAGE_VERSION)
#   PRODUCT_NAME
#     Optional product name string (defaults to "LLVM")
#   )
function(set_windows_version_resource_properties name resource_file)
  cmake_parse_arguments(ARG
    ""
    "VERSION_MAJOR;VERSION_MINOR;VERSION_PATCHLEVEL;VERSION_STRING;PRODUCT_NAME"
    ""
    ${ARGN})

  if (NOT DEFINED ARG_VERSION_MAJOR)
    set(ARG_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
  endif()

  if (NOT DEFINED ARG_VERSION_MINOR)
    set(ARG_VERSION_MINOR ${LLVM_VERSION_MINOR})
  endif()

  if (NOT DEFINED ARG_VERSION_PATCHLEVEL)
    set(ARG_VERSION_PATCHLEVEL ${LLVM_VERSION_PATCH})
  endif()

  if (NOT DEFINED ARG_VERSION_STRING)
    set(ARG_VERSION_STRING ${PACKAGE_VERSION})
  endif()

  if (NOT DEFINED ARG_PRODUCT_NAME)
    set(ARG_PRODUCT_NAME "LLVM")
  endif()

  set_property(SOURCE ${resource_file}
               PROPERTY COMPILE_FLAGS /nologo)
  set_property(SOURCE ${resource_file}
               PROPERTY COMPILE_DEFINITIONS
               "RC_VERSION_FIELD_1=${ARG_VERSION_MAJOR}"
               "RC_VERSION_FIELD_2=${ARG_VERSION_MINOR}"
               "RC_VERSION_FIELD_3=${ARG_VERSION_PATCHLEVEL}"
               "RC_VERSION_FIELD_4=0"
               "RC_FILE_VERSION=\"${ARG_VERSION_STRING}\""
               "RC_INTERNAL_NAME=\"${name}\""
               "RC_PRODUCT_NAME=\"${ARG_PRODUCT_NAME}\""
               "RC_PRODUCT_VERSION=\"${ARG_VERSION_STRING}\"")
endfunction(set_windows_version_resource_properties)

# llvm_add_library(name sources...
#   SHARED;STATIC
#     STATIC by default w/o BUILD_SHARED_LIBS.
#     SHARED by default w/  BUILD_SHARED_LIBS.
#   OBJECT
#     Also create an OBJECT library target. Default if STATIC && SHARED.
#   MODULE
#     Target ${name} might not be created on unsupported platforms.
#     Check with "if(TARGET ${name})".
#   DISABLE_LLVM_LINK_LLVM_DYLIB
#     Do not link this library to libLLVM, even if
#     LLVM_LINK_LLVM_DYLIB is enabled.
#   OUTPUT_NAME name
#     Corresponds to OUTPUT_NAME in target properties.
#   DEPENDS targets...
#     Same semantics as add_dependencies().
#   LINK_COMPONENTS components...
#     Same as the variable LLVM_LINK_COMPONENTS.
#   LINK_LIBS lib_targets...
#     Same semantics as target_link_libraries().
#   ADDITIONAL_HEADERS
#     May specify header files for IDE generators.
#   SONAME
#     Should set SONAME link flags and create symlinks
#   PLUGIN_TOOL
#     The tool (i.e. cmake target) that this plugin will link against
#   )
function(llvm_add_library name)
  cmake_parse_arguments(ARG
    "MODULE;SHARED;STATIC;OBJECT;DISABLE_LLVM_LINK_LLVM_DYLIB;SONAME"
    "OUTPUT_NAME;PLUGIN_TOOL"
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS;OBJLIBS"
    ${ARGN})
  list(APPEND LLVM_COMMON_DEPENDS ${ARG_DEPENDS})
  if(ARG_ADDITIONAL_HEADERS)
    # Pass through ADDITIONAL_HEADERS.
    set(ARG_ADDITIONAL_HEADERS ADDITIONAL_HEADERS ${ARG_ADDITIONAL_HEADERS})
  endif()
  if(ARG_OBJLIBS)
    set(ALL_FILES ${ARG_OBJLIBS})
  else()
    llvm_process_sources(ALL_FILES ${ARG_UNPARSED_ARGUMENTS} ${ARG_ADDITIONAL_HEADERS})
  endif()

  if(ARG_MODULE)
    if(ARG_SHARED OR ARG_STATIC)
      message(WARNING "MODULE with SHARED|STATIC doesn't make sense.")
    endif()
    # Plugins that link against a tool are allowed even when plugins in general are not
    if(NOT LLVM_ENABLE_PLUGINS AND NOT (ARG_PLUGIN_TOOL AND LLVM_EXPORT_SYMBOLS_FOR_PLUGINS))
      message(STATUS "${name} ignored -- Loadable modules not supported on this platform.")
      return()
    endif()
  else()
    if(ARG_PLUGIN_TOOL)
      message(WARNING "PLUGIN_TOOL without MODULE doesn't make sense.")
    endif()
    if(BUILD_SHARED_LIBS AND NOT ARG_STATIC)
      set(ARG_SHARED TRUE)
    endif()
    if(NOT ARG_SHARED)
      set(ARG_STATIC TRUE)
    endif()
  endif()

  # Generate objlib
  if((ARG_SHARED AND ARG_STATIC) OR ARG_OBJECT)
    # Generate an obj library for both targets.
    set(obj_name "obj.${name}")
    add_library(${obj_name} OBJECT EXCLUDE_FROM_ALL
      ${ALL_FILES}
      )
    llvm_update_compile_flags(${obj_name})
    set(ALL_FILES "$<TARGET_OBJECTS:${obj_name}>")

    # Do add_dependencies(obj) later due to CMake issue 14747.
    list(APPEND objlibs ${obj_name})

    set_target_properties(${obj_name} PROPERTIES FOLDER "Object Libraries")
  endif()

  if(ARG_SHARED AND ARG_STATIC)
    # static
    set(name_static "${name}_static")
    if(ARG_OUTPUT_NAME)
      set(output_name OUTPUT_NAME "${ARG_OUTPUT_NAME}")
    endif()
    # DEPENDS has been appended to LLVM_COMMON_LIBS.
    llvm_add_library(${name_static} STATIC
      ${output_name}
      OBJLIBS ${ALL_FILES} # objlib
      LINK_LIBS ${ARG_LINK_LIBS}
      LINK_COMPONENTS ${ARG_LINK_COMPONENTS}
      )
    # FIXME: Add name_static to anywhere in TARGET ${name}'s PROPERTY.
    set(ARG_STATIC)
  endif()

  if(ARG_MODULE)
    add_library(${name} MODULE ${ALL_FILES})
    llvm_setup_rpath(${name})
  elseif(ARG_SHARED)
    add_windows_version_resource_file(ALL_FILES ${ALL_FILES})
    add_library(${name} SHARED ${ALL_FILES})

    llvm_setup_rpath(${name})

  else()
    add_library(${name} STATIC ${ALL_FILES})
  endif()

  setup_dependency_debugging(${name} ${LLVM_COMMON_DEPENDS})

  if(DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file})
    set(windows_resource_file ${windows_resource_file} PARENT_SCOPE)
  endif()

  set_output_directory(${name} BINARY_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR} LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  # $<TARGET_OBJECTS> doesn't require compile flags.
  if(NOT obj_name)
    llvm_update_compile_flags(${name})
  endif()
  add_link_opts( ${name} )
  if(ARG_OUTPUT_NAME)
    set_target_properties(${name}
      PROPERTIES
      OUTPUT_NAME ${ARG_OUTPUT_NAME}
      )
  endif()

  if(ARG_MODULE)
    set_target_properties(${name} PROPERTIES
      PREFIX ""
      SUFFIX ${LLVM_PLUGIN_EXT}
      )
  endif()

  if(ARG_SHARED)
    if(WIN32)
      set_target_properties(${name} PROPERTIES
        PREFIX ""
        )
    endif()

    # Set SOVERSION on shared libraries that lack explicit SONAME
    # specifier, on *nix systems that are not Darwin.
    if(UNIX AND NOT APPLE AND NOT ARG_SONAME)
      set_target_properties(${name}
        PROPERTIES
        # Since 4.0.0, the ABI version is indicated by the major version
        SOVERSION ${LLVM_VERSION_MAJOR}
        VERSION ${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}${LLVM_VERSION_SUFFIX})
    endif()
  endif()

  if(ARG_MODULE OR ARG_SHARED)
    # Do not add -Dname_EXPORTS to the command-line when building files in this
    # target. Doing so is actively harmful for the modules build because it
    # creates extra module variants, and not useful because we don't use these
    # macros.
    set_target_properties( ${name} PROPERTIES DEFINE_SYMBOL "" )

    if (LLVM_EXPORTED_SYMBOL_FILE)
      add_llvm_symbol_exports( ${name} ${LLVM_EXPORTED_SYMBOL_FILE} )
    endif()
  endif()

  if(ARG_SHARED AND UNIX)
    if(NOT APPLE AND ARG_SONAME)
      get_target_property(output_name ${name} OUTPUT_NAME)
      if(${output_name} STREQUAL "output_name-NOTFOUND")
        set(output_name ${name})
      endif()
      set(library_name ${output_name}-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX})
      set(api_name ${output_name}-${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}${LLVM_VERSION_SUFFIX})
      set_target_properties(${name} PROPERTIES OUTPUT_NAME ${library_name})
      llvm_install_library_symlink(${api_name} ${library_name} SHARED
        COMPONENT ${name}
        ALWAYS_GENERATE)
      llvm_install_library_symlink(${output_name} ${library_name} SHARED
        COMPONENT ${name}
        ALWAYS_GENERATE)
    endif()
  endif()

  if(ARG_MODULE AND LLVM_EXPORT_SYMBOLS_FOR_PLUGINS AND ARG_PLUGIN_TOOL AND (WIN32 OR CYGWIN))
    # On DLL platforms symbols are imported from the tool by linking against it.
    set(llvm_libs ${ARG_PLUGIN_TOOL})
  elseif (DEFINED LLVM_LINK_COMPONENTS OR DEFINED ARG_LINK_COMPONENTS)
    if (LLVM_LINK_LLVM_DYLIB AND NOT ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
      set(llvm_libs LLVM)
    else()
      llvm_map_components_to_libnames(llvm_libs
       ${ARG_LINK_COMPONENTS}
       ${LLVM_LINK_COMPONENTS}
       )
    endif()
  else()
    # Components have not been defined explicitly in CMake, so add the
    # dependency information for this library as defined by LLVMBuild.
    #
    # It would be nice to verify that we have the dependencies for this library
    # name, but using get_property(... SET) doesn't suffice to determine if a
    # property has been set to an empty value.
    get_property(lib_deps GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${name})
  endif()

  if(ARG_STATIC)
    set(libtype INTERFACE)
  else()
    # We can use PRIVATE since SO knows its dependent libs.
    set(libtype PRIVATE)
  endif()

  target_link_libraries(${name} ${libtype}
      ${ARG_LINK_LIBS}
      ${lib_deps}
      ${llvm_libs}
      )

  if(LLVM_COMMON_DEPENDS)
    add_dependencies(${name} ${LLVM_COMMON_DEPENDS})
    # Add dependencies also to objlibs.
    # CMake issue 14747 --  add_dependencies() might be ignored to objlib's user.
    foreach(objlib ${objlibs})
      add_dependencies(${objlib} ${LLVM_COMMON_DEPENDS})
    endforeach()
  endif()

  if(ARG_SHARED OR ARG_MODULE)
    llvm_externalize_debuginfo(${name})
  endif()
endfunction()

macro(add_llvm_library name)
  cmake_parse_arguments(ARG
    "SHARED;BUILDTREE_ONLY"
    ""
    ""
    ${ARGN})
  if( BUILD_SHARED_LIBS OR ARG_SHARED )
    llvm_add_library(${name} SHARED ${ARG_UNPARSED_ARGUMENTS})
  else()
    llvm_add_library(${name} ${ARG_UNPARSED_ARGUMENTS})
  endif()

  # Libraries that are meant to only be exposed via the build tree only are
  # never installed and are only exported as a target in the special build tree
  # config file.
  if (NOT ARG_BUILDTREE_ONLY)
    set_property( GLOBAL APPEND PROPERTY LLVM_LIBS ${name} )
  endif()

  if( EXCLUDE_FROM_ALL )
    set_target_properties( ${name} PROPERTIES EXCLUDE_FROM_ALL ON)
  elseif(ARG_BUILDTREE_ONLY)
    set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS_BUILDTREE_ONLY ${name})
  else()
    if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR ${name} STREQUAL "LTO" OR
        (LLVM_LINK_LLVM_DYLIB AND ${name} STREQUAL "LLVM"))
      set(install_dir lib${LLVM_LIBDIR_SUFFIX})
      if(ARG_SHARED OR BUILD_SHARED_LIBS)
        if(WIN32 OR CYGWIN OR MINGW)
          set(install_type RUNTIME)
          set(install_dir bin)
        else()
          set(install_type LIBRARY)
        endif()
      else()
        set(install_type ARCHIVE)
      endif()

      if(${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
          NOT LLVM_DISTRIBUTION_COMPONENTS)
        set(export_to_llvmexports EXPORT LLVMExports)
        set_property(GLOBAL PROPERTY LLVM_HAS_EXPORTS True)
      endif()

      install(TARGETS ${name}
              ${export_to_llvmexports}
              ${install_type} DESTINATION ${install_dir}
              COMPONENT ${name})

      if (NOT CMAKE_CONFIGURATION_TYPES)
        add_custom_target(install-${name}
                          DEPENDS ${name}
                          COMMAND "${CMAKE_COMMAND}"
                                  -DCMAKE_INSTALL_COMPONENT=${name}
                                  -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      endif()
    endif()
    set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Libraries")
endmacro(add_llvm_library name)

macro(add_llvm_loadable_module name)
  llvm_add_library(${name} MODULE ${ARGN})
  if(NOT TARGET ${name})
    # Add empty "phony" target
    add_custom_target(${name})
  else()
    if( EXCLUDE_FROM_ALL )
      set_target_properties( ${name} PROPERTIES EXCLUDE_FROM_ALL ON)
    else()
      if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
        if(WIN32 OR CYGWIN)
          # DLL platform
          set(dlldir "bin")
        else()
          set(dlldir "lib${LLVM_LIBDIR_SUFFIX}")
        endif()

        if(${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
            NOT LLVM_DISTRIBUTION_COMPONENTS)
          set(export_to_llvmexports EXPORT LLVMExports)
          set_property(GLOBAL PROPERTY LLVM_HAS_EXPORTS True)
        endif()

        install(TARGETS ${name}
                ${export_to_llvmexports}
                LIBRARY DESTINATION ${dlldir}
                ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX})
      endif()
      set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${name})
    endif()
  endif()

  set_target_properties(${name} PROPERTIES FOLDER "Loadable modules")
endmacro(add_llvm_loadable_module name)


macro(add_llvm_executable name)
  cmake_parse_arguments(ARG "DISABLE_LLVM_LINK_LLVM_DYLIB;IGNORE_EXTERNALIZE_DEBUGINFO;NO_INSTALL_RPATH" "" "DEPENDS" ${ARGN})
  llvm_process_sources( ALL_FILES ${ARG_UNPARSED_ARGUMENTS} )

  list(APPEND LLVM_COMMON_DEPENDS ${ARG_DEPENDS})

  # Generate objlib
  if(LLVM_ENABLE_OBJLIB)
    # Generate an obj library for both targets.
    set(obj_name "obj.${name}")
    add_library(${obj_name} OBJECT EXCLUDE_FROM_ALL
      ${ALL_FILES}
      )
    llvm_update_compile_flags(${obj_name})
    set(ALL_FILES "$<TARGET_OBJECTS:${obj_name}>")

    set_target_properties(${obj_name} PROPERTIES FOLDER "Object Libraries")
  endif()

  add_windows_version_resource_file(ALL_FILES ${ALL_FILES})

  if(XCODE)
    # Note: the dummy.cpp source file provides no definitions. However,
    # it forces Xcode to properly link the static library.
    list(APPEND ALL_FILES "${LLVM_MAIN_SRC_DIR}/cmake/dummy.cpp")
  endif()
  
  if( EXCLUDE_FROM_ALL )
    add_executable(${name} EXCLUDE_FROM_ALL ${ALL_FILES})
  else()
    add_executable(${name} ${ALL_FILES})
  endif()

  setup_dependency_debugging(${name} ${LLVM_COMMON_DEPENDS})

  if(NOT ARG_NO_INSTALL_RPATH)
    llvm_setup_rpath(${name})
  endif()

  if(DEFINED windows_resource_file)
    set_windows_version_resource_properties(${name} ${windows_resource_file})
  endif()

  # $<TARGET_OBJECTS> doesn't require compile flags.
  if(NOT LLVM_ENABLE_OBJLIB)
    llvm_update_compile_flags(${name})
  endif()
  add_link_opts( ${name} )

  # Do not add -Dname_EXPORTS to the command-line when building files in this
  # target. Doing so is actively harmful for the modules build because it
  # creates extra module variants, and not useful because we don't use these
  # macros.
  set_target_properties( ${name} PROPERTIES DEFINE_SYMBOL "" )

  if (LLVM_EXPORTED_SYMBOL_FILE)
    add_llvm_symbol_exports( ${name} ${LLVM_EXPORTED_SYMBOL_FILE} )
  endif(LLVM_EXPORTED_SYMBOL_FILE)

  if (LLVM_LINK_LLVM_DYLIB AND NOT ARG_DISABLE_LLVM_LINK_LLVM_DYLIB)
    set(USE_SHARED USE_SHARED)
  endif()

  set(EXCLUDE_FROM_ALL OFF)
  set_output_directory(${name} BINARY_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR} LIBRARY_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
  llvm_config( ${name} ${USE_SHARED} ${LLVM_LINK_COMPONENTS} )
  if( LLVM_COMMON_DEPENDS )
    add_dependencies( ${name} ${LLVM_COMMON_DEPENDS} )
  endif( LLVM_COMMON_DEPENDS )

  if(NOT ARG_IGNORE_EXTERNALIZE_DEBUGINFO)
    llvm_externalize_debuginfo(${name})
  endif()
  if (LLVM_PTHREAD_LIB)
    # libpthreads overrides some standard library symbols, so main
    # executable must be linked with it in order to provide consistent
    # API for all shared libaries loaded by this executable.
    target_link_libraries(${name} ${LLVM_PTHREAD_LIB})
  endif()
endmacro(add_llvm_executable name)

function(export_executable_symbols target)
  if (LLVM_EXPORTED_SYMBOL_FILE)
    # The symbol file should contain the symbols we want the executable to
    # export
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS 1)
  elseif (LLVM_EXPORT_SYMBOLS_FOR_PLUGINS)
    # Extract the symbols to export from the static libraries that the
    # executable links against.
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS 1)
    set(exported_symbol_file ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${target}.symbols)
    # We need to consider not just the direct link dependencies, but also the
    # transitive link dependencies. Do this by starting with the set of direct
    # dependencies, then the dependencies of those dependencies, and so on.
    get_target_property(new_libs ${target} LINK_LIBRARIES)
    set(link_libs ${new_libs})
    while(NOT "${new_libs}" STREQUAL "")
      foreach(lib ${new_libs})
        if(TARGET ${lib})
          get_target_property(lib_type ${lib} TYPE)
          if("${lib_type}" STREQUAL "STATIC_LIBRARY")
            list(APPEND static_libs ${lib})
          else()
            list(APPEND other_libs ${lib})
          endif()
          get_target_property(transitive_libs ${lib} INTERFACE_LINK_LIBRARIES)
          foreach(transitive_lib ${transitive_libs})
            list(FIND link_libs ${transitive_lib} idx)
            if(TARGET ${transitive_lib} AND idx EQUAL -1)
              list(APPEND newer_libs ${transitive_lib})
              list(APPEND link_libs ${transitive_lib})
            endif()
          endforeach(transitive_lib)
        endif()
      endforeach(lib)
      set(new_libs ${newer_libs})
      set(newer_libs "")
    endwhile()
    if (MSVC)
      set(mangling microsoft)
    else()
      set(mangling itanium)
    endif()
    add_custom_command(OUTPUT ${exported_symbol_file}
                       COMMAND ${PYTHON_EXECUTABLE} ${LLVM_MAIN_SRC_DIR}/utils/extract_symbols.py --mangling=${mangling} ${static_libs} -o ${exported_symbol_file}
                       WORKING_DIRECTORY ${LLVM_LIBRARY_OUTPUT_INTDIR}
                       DEPENDS ${LLVM_MAIN_SRC_DIR}/utils/extract_symbols.py ${static_libs}
                       VERBATIM
                       COMMENT "Generating export list for ${target}")
    add_llvm_symbol_exports( ${target} ${exported_symbol_file} )
    # If something links against this executable then we want a
    # transitive link against only the libraries whose symbols
    # we aren't exporting.
    set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${other_libs}")
    # The default import library suffix that cmake uses for cygwin/mingw is
    # ".dll.a", but for clang.exe that causes a collision with libclang.dll,
    # where the import libraries of both get named libclang.dll.a. Use a suffix
    # of ".exe.a" to avoid this.
    if(CYGWIN OR MINGW)
      set_target_properties(${target} PROPERTIES IMPORT_SUFFIX ".exe.a")
    endif()
  elseif(NOT (WIN32 OR CYGWIN))
    # On Windows auto-exporting everything doesn't work because of the limit on
    # the size of the exported symbol table, but on other platforms we can do
    # it without any trouble.
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS 1)
    if (APPLE)
      set_property(TARGET ${target} APPEND_STRING PROPERTY
        LINK_FLAGS " -rdynamic")
    endif()
  endif()
endfunction()

if(NOT LLVM_TOOLCHAIN_TOOLS)
  set (LLVM_TOOLCHAIN_TOOLS
    llvm-ar
    llvm-ranlib
    llvm-lib
    llvm-objdump
    )
endif()

macro(add_llvm_tool name)
  if( NOT LLVM_BUILD_TOOLS )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${name} ${ARGN})

  if ( ${name} IN_LIST LLVM_TOOLCHAIN_TOOLS OR NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    if( LLVM_BUILD_TOOLS )
      if(${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
          NOT LLVM_DISTRIBUTION_COMPONENTS)
        set(export_to_llvmexports EXPORT LLVMExports)
        set_property(GLOBAL PROPERTY LLVM_HAS_EXPORTS True)
      endif()

      install(TARGETS ${name}
              ${export_to_llvmexports}
              RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
              COMPONENT ${name})

      if (NOT CMAKE_CONFIGURATION_TYPES)
        add_custom_target(install-${name}
                          DEPENDS ${name}
                          COMMAND "${CMAKE_COMMAND}"
                                  -DCMAKE_INSTALL_COMPONENT=${name}
                                  -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
      endif()
    endif()
  endif()
  if( LLVM_BUILD_TOOLS )
    set_property(GLOBAL APPEND PROPERTY LLVM_EXPORTS ${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Tools")
endmacro(add_llvm_tool name)


macro(add_llvm_example name)
  if( NOT LLVM_BUILD_EXAMPLES )
    set(EXCLUDE_FROM_ALL ON)
  endif()
  add_llvm_executable(${name} ${ARGN})
  if( LLVM_BUILD_EXAMPLES )
    install(TARGETS ${name} RUNTIME DESTINATION examples)
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "Examples")
endmacro(add_llvm_example name)

# This is a macro that is used to create targets for executables that are needed
# for development, but that are not intended to be installed by default.
macro(add_llvm_utility name)
  if ( NOT LLVM_BUILD_UTILS )
    set(EXCLUDE_FROM_ALL ON)
  endif()

  add_llvm_executable(${name} DISABLE_LLVM_LINK_LLVM_DYLIB ${ARGN})
  set_target_properties(${name} PROPERTIES FOLDER "Utils")
  if( LLVM_INSTALL_UTILS AND LLVM_BUILD_UTILS )
    install (TARGETS ${name}
      RUNTIME DESTINATION ${LLVM_UTILS_INSTALL_DIR}
      COMPONENT ${name})
    if (NOT CMAKE_CONFIGURATION_TYPES)
      add_custom_target(install-${name}
                        DEPENDS ${name}
                        COMMAND "${CMAKE_COMMAND}"
                                -DCMAKE_INSTALL_COMPONENT=${name}
                                -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
    endif()
  endif()
endmacro(add_llvm_utility name)

macro(add_llvm_fuzzer name)
  if( LLVM_USE_SANITIZE_COVERAGE )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=fuzzer")
    add_llvm_executable(${name} ${ARGN})
    set_target_properties(${name} PROPERTIES FOLDER "Fuzzers")
  endif()
endmacro()

macro(add_llvm_target target_name)
  include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
  add_llvm_library(LLVM${target_name} ${ARGN})
  set( CURRENT_LLVM_TARGET LLVM${target_name} )
endmacro(add_llvm_target)

function(canonicalize_tool_name name output)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}/" "" nameStrip ${name})
  string(REPLACE "-" "_" nameUNDERSCORE ${nameStrip})
  string(TOUPPER ${nameUNDERSCORE} nameUPPER)
  set(${output} "${nameUPPER}" PARENT_SCOPE)
endfunction(canonicalize_tool_name)

# Custom add_subdirectory wrapper
# Takes in a project name (i.e. LLVM), the subdirectory name, and an optional
# path if it differs from the name.
macro(add_llvm_subdirectory project type name)
  set(add_llvm_external_dir "${ARGN}")
  if("${add_llvm_external_dir}" STREQUAL "")
    set(add_llvm_external_dir ${name})
  endif()
  canonicalize_tool_name(${name} nameUPPER)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${add_llvm_external_dir}/CMakeLists.txt)
    # Treat it as in-tree subproject.
    option(${project}_${type}_${nameUPPER}_BUILD
           "Whether to build ${name} as part of ${project}" On)
    mark_as_advanced(${project}_${type}_${name}_BUILD)
    if(${project}_${type}_${nameUPPER}_BUILD)
      add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/${add_llvm_external_dir} ${add_llvm_external_dir})
      # Don't process it in add_llvm_implicit_projects().
      set(${project}_${type}_${nameUPPER}_BUILD OFF)
    endif()
  else()
    set(LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR
      "${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR}"
      CACHE PATH "Path to ${name} source directory")
    set(${project}_${type}_${nameUPPER}_BUILD_DEFAULT ON)
    if(NOT LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR OR NOT EXISTS ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR})
      set(${project}_${type}_${nameUPPER}_BUILD_DEFAULT OFF)
    endif()
    if("${LLVM_EXTERNAL_${nameUPPER}_BUILD}" STREQUAL "OFF")
      set(${project}_${type}_${nameUPPER}_BUILD_DEFAULT OFF)
    endif()
    option(${project}_${type}_${nameUPPER}_BUILD
      "Whether to build ${name} as part of LLVM"
      ${${project}_${type}_${nameUPPER}_BUILD_DEFAULT})
    if (${project}_${type}_${nameUPPER}_BUILD)
      if(EXISTS ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR})
        add_subdirectory(${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR} ${add_llvm_external_dir})
      elseif(NOT "${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR}" STREQUAL "")
        message(WARNING "Nonexistent directory for ${name}: ${LLVM_EXTERNAL_${nameUPPER}_SOURCE_DIR}")
      endif()
      # FIXME: It'd be redundant.
      set(${project}_${type}_${nameUPPER}_BUILD Off)
    endif()
  endif()
endmacro()

# Add external project that may want to be built as part of llvm such as Clang,
# lld, and Polly. This adds two options. One for the source directory of the
# project, which defaults to ${CMAKE_CURRENT_SOURCE_DIR}/${name}. Another to
# enable or disable building it with everything else.
# Additional parameter can be specified as the name of directory.
macro(add_llvm_external_project name)
  add_llvm_subdirectory(LLVM TOOL ${name} ${ARGN})
endmacro()

macro(add_llvm_tool_subdirectory name)
  add_llvm_external_project(${name})
endmacro(add_llvm_tool_subdirectory)

function(get_project_name_from_src_var var output)
  string(REGEX MATCH "LLVM_EXTERNAL_(.*)_SOURCE_DIR"
         MACHED_TOOL "${var}")
  if(MACHED_TOOL)
    set(${output} ${CMAKE_MATCH_1} PARENT_SCOPE)
  else()
    set(${output} PARENT_SCOPE)
  endif()
endfunction()

function(create_subdirectory_options project type)
  file(GLOB sub-dirs "${CMAKE_CURRENT_SOURCE_DIR}/*")
  foreach(dir ${sub-dirs})
    if(IS_DIRECTORY "${dir}" AND EXISTS "${dir}/CMakeLists.txt")
      canonicalize_tool_name(${dir} name)
      option(${project}_${type}_${name}_BUILD
           "Whether to build ${name} as part of ${project}" On)
      mark_as_advanced(${project}_${type}_${name}_BUILD)
    endif()
  endforeach()
endfunction(create_subdirectory_options)

function(create_llvm_tool_options)
  create_subdirectory_options(LLVM TOOL)
endfunction(create_llvm_tool_options)

function(llvm_add_implicit_projects project)
  set(list_of_implicit_subdirs "")
  file(GLOB sub-dirs "${CMAKE_CURRENT_SOURCE_DIR}/*")
  foreach(dir ${sub-dirs})
    if(IS_DIRECTORY "${dir}" AND EXISTS "${dir}/CMakeLists.txt")
      canonicalize_tool_name(${dir} name)
      if (${project}_TOOL_${name}_BUILD)
        get_filename_component(fn "${dir}" NAME)
        list(APPEND list_of_implicit_subdirs "${fn}")
      endif()
    endif()
  endforeach()

  foreach(external_proj ${list_of_implicit_subdirs})
    add_llvm_subdirectory(${project} TOOL "${external_proj}" ${ARGN})
  endforeach()
endfunction(llvm_add_implicit_projects)

function(add_llvm_implicit_projects)
  llvm_add_implicit_projects(LLVM)
endfunction(add_llvm_implicit_projects)

# Generic support for adding a unittest.
function(add_unittest test_suite test_name)
  if( NOT LLVM_BUILD_TESTS )
    set(EXCLUDE_FROM_ALL ON)
  endif()

  include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googletest/include)
  include_directories(${LLVM_MAIN_SRC_DIR}/utils/unittest/googlemock/include)
  if (NOT LLVM_ENABLE_THREADS)
    list(APPEND LLVM_COMPILE_DEFINITIONS GTEST_HAS_PTHREAD=0)
  endif ()

  if (SUPPORTS_VARIADIC_MACROS_FLAG)
    list(APPEND LLVM_COMPILE_FLAGS "-Wno-variadic-macros")
  endif ()
  # Some parts of gtest rely on this GNU extension, don't warn on it.
  if(SUPPORTS_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS_FLAG)
    list(APPEND LLVM_COMPILE_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
  endif()

  set(LLVM_REQUIRES_RTTI OFF)

  list(APPEND LLVM_LINK_COMPONENTS Support) # gtest needs it for raw_ostream
  add_llvm_executable(${test_name} IGNORE_EXTERNALIZE_DEBUGINFO NO_INSTALL_RPATH ${ARGN})
  set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
  set_output_directory(${test_name} BINARY_DIR ${outdir} LIBRARY_DIR ${outdir})
  # libpthreads overrides some standard library symbols, so main
  # executable must be linked with it in order to provide consistent
  # API for all shared libaries loaded by this executable.
  target_link_libraries(${test_name} gtest_main gtest ${LLVM_PTHREAD_LIB})

  add_dependencies(${test_suite} ${test_name})
  get_target_property(test_suite_folder ${test_suite} FOLDER)
  if (NOT ${test_suite_folder} STREQUAL "NOTFOUND")
    set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
  endif ()
endfunction()

function(llvm_add_go_executable binary pkgpath)
  cmake_parse_arguments(ARG "ALL" "" "DEPENDS;GOFLAGS" ${ARGN})

  if(LLVM_BINDINGS MATCHES "go")
    # FIXME: This should depend only on the libraries Go needs.
    get_property(llvmlibs GLOBAL PROPERTY LLVM_LIBS)
    set(binpath ${CMAKE_BINARY_DIR}/bin/${binary}${CMAKE_EXECUTABLE_SUFFIX})
    set(cc "${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}")
    set(cxx "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
    set(cppflags "")
    get_property(include_dirs DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
    foreach(d ${include_dirs})
      set(cppflags "${cppflags} -I${d}")
    endforeach(d)
    set(ldflags "${CMAKE_EXE_LINKER_FLAGS}")
    add_custom_command(OUTPUT ${binpath}
      COMMAND ${CMAKE_BINARY_DIR}/bin/llvm-go "go=${GO_EXECUTABLE}" "cc=${cc}" "cxx=${cxx}" "cppflags=${cppflags}" "ldflags=${ldflags}" "packages=${LLVM_GO_PACKAGES}"
              ${ARG_GOFLAGS} build -o ${binpath} ${pkgpath}
      DEPENDS llvm-config ${CMAKE_BINARY_DIR}/bin/llvm-go${CMAKE_EXECUTABLE_SUFFIX}
              ${llvmlibs} ${ARG_DEPENDS}
      COMMENT "Building Go executable ${binary}"
      VERBATIM)
    if (ARG_ALL)
      add_custom_target(${binary} ALL DEPENDS ${binpath})
    else()
      add_custom_target(${binary} DEPENDS ${binpath})
    endif()
  endif()
endfunction()

# This function canonicalize the CMake variables passed by names
# from CMake boolean to 0/1 suitable for passing into Python or C++,
# in place.
function(llvm_canonicalize_cmake_booleans)
  foreach(var ${ARGN})
    if(${var})
      set(${var} 1 PARENT_SCOPE)
    else()
      set(${var} 0 PARENT_SCOPE)
    endif()
  endforeach()
endfunction(llvm_canonicalize_cmake_booleans)

# This function provides an automatic way to 'configure'-like generate a file
# based on a set of common and custom variables, specifically targeting the
# variables needed for the 'lit.site.cfg' files. This function bundles the
# common variables that any Lit instance is likely to need, and custom
# variables can be passed in.
function(configure_lit_site_cfg input output)
  foreach(c ${LLVM_TARGETS_TO_BUILD})
    set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
  endforeach(c)
  set(TARGETS_TO_BUILD ${TARGETS_BUILT})

  set(SHLIBEXT "${LTDL_SHLIB_EXT}")

  # Configuration-time: See Unit/lit.site.cfg.in
  if (CMAKE_CFG_INTDIR STREQUAL ".")
    set(LLVM_BUILD_MODE ".")
  else ()
    set(LLVM_BUILD_MODE "%(build_mode)s")
  endif ()

  # They below might not be the build tree but provided binary tree.
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLVM_TOOLS_DIR ${LLVM_TOOLS_BINARY_DIR})
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLVM_LIBS_DIR  ${LLVM_LIBRARY_DIR})

  # SHLIBDIR points the build tree.
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} SHLIBDIR "${LLVM_SHLIB_OUTPUT_INTDIR}")

  set(PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})
  # FIXME: "ENABLE_SHARED" doesn't make sense, since it is used just for
  # plugins. We may rename it.
  if(LLVM_ENABLE_PLUGINS)
    set(ENABLE_SHARED "1")
  else()
    set(ENABLE_SHARED "0")
  endif()

  if(LLVM_ENABLE_ASSERTIONS AND NOT MSVC_IDE)
    set(ENABLE_ASSERTIONS "1")
  else()
    set(ENABLE_ASSERTIONS "0")
  endif()

  set(HOST_OS ${CMAKE_SYSTEM_NAME})
  set(HOST_ARCH ${CMAKE_SYSTEM_PROCESSOR})

  set(HOST_CC "${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1}")
  set(HOST_CXX "${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_ARG1}")
  set(HOST_LDFLAGS "${CMAKE_EXE_LINKER_FLAGS}")

  set(LIT_SITE_CFG_IN_HEADER  "## Autogenerated from ${input}\n## Do not edit!")

  # Override config_target_triple (and the env)
  if(LLVM_TARGET_TRIPLE_ENV)
    # This is expanded into the heading.
    string(CONCAT LIT_SITE_CFG_IN_HEADER "${LIT_SITE_CFG_IN_HEADER}\n\n"
      "import os\n"
      "target_env = \"${LLVM_TARGET_TRIPLE_ENV}\"\n"
      "config.target_triple = config.environment[target_env] = os.environ.get(target_env, \"${TARGET_TRIPLE}\")\n"
      )

    # This is expanded to; config.target_triple = ""+config.target_triple+""
    set(TARGET_TRIPLE "\"+config.target_triple+\"")
  endif()

  configure_file(${input} ${output} @ONLY)
endfunction()

# A raw function to create a lit target. This is used to implement the testuite
# management functions.
function(add_lit_target target comment)
  cmake_parse_arguments(ARG "" "" "PARAMS;DEPENDS;ARGS" ${ARGN})
  set(LIT_ARGS "${ARG_ARGS} ${LLVM_LIT_ARGS}")
  separate_arguments(LIT_ARGS)
  if (NOT CMAKE_CFG_INTDIR STREQUAL ".")
    list(APPEND LIT_ARGS --param build_mode=${CMAKE_CFG_INTDIR})
  endif ()
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
    set (LIT_COMMAND "${PYTHON_EXECUTABLE};${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py"
         CACHE STRING "Command used to spawn llvm-lit")
  else()
    find_program(LIT_COMMAND NAMES llvm-lit lit.py lit)
  endif ()
  list(APPEND LIT_COMMAND ${LIT_ARGS})
  foreach(param ${ARG_PARAMS})
    list(APPEND LIT_COMMAND --param ${param})
  endforeach()
  if (ARG_UNPARSED_ARGUMENTS)
    add_custom_target(${target}
      COMMAND ${LIT_COMMAND} ${ARG_UNPARSED_ARGUMENTS}
      COMMENT "${comment}"
      USES_TERMINAL
      )
  else()
    add_custom_target(${target}
      COMMAND ${CMAKE_COMMAND} -E echo "${target} does nothing, no tools built.")
    message(STATUS "${target} does nothing.")
  endif()
  if (ARG_DEPENDS)
    add_dependencies(${target} ${ARG_DEPENDS})
  endif()

  # Tests should be excluded from "Build Solution".
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
endfunction()

# A function to add a set of lit test suites to be driven through 'check-*' targets.
function(add_lit_testsuite target comment)
  cmake_parse_arguments(ARG "" "" "PARAMS;DEPENDS;ARGS" ${ARGN})

  # EXCLUDE_FROM_ALL excludes the test ${target} out of check-all.
  if(NOT EXCLUDE_FROM_ALL)
    # Register the testsuites, params and depends for the global check rule.
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_TESTSUITES ${ARG_UNPARSED_ARGUMENTS})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_PARAMS ${ARG_PARAMS})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_DEPENDS ${ARG_DEPENDS})
    set_property(GLOBAL APPEND PROPERTY LLVM_LIT_EXTRA_ARGS ${ARG_ARGS})
  endif()

  # Produce a specific suffixed check rule.
  add_lit_target(${target} ${comment}
    ${ARG_UNPARSED_ARGUMENTS}
    PARAMS ${ARG_PARAMS}
    DEPENDS ${ARG_DEPENDS}
    ARGS ${ARG_ARGS}
    )
endfunction()

function(add_lit_testsuites project directory)
  if (NOT CMAKE_CONFIGURATION_TYPES)
    cmake_parse_arguments(ARG "" "" "PARAMS;DEPENDS;ARGS" ${ARGN})

    # Search recursively for test directories by assuming anything not
    # in a directory called Inputs contains tests.
    file(GLOB_RECURSE to_process LIST_DIRECTORIES true ${directory}/*)
    foreach(lit_suite ${to_process})
      if(NOT IS_DIRECTORY ${lit_suite})
        continue()
      endif()
      string(FIND ${lit_suite} Inputs is_inputs)
      string(FIND ${lit_suite} Output is_output)
      if (NOT (is_inputs EQUAL -1 AND is_output EQUAL -1))
        continue()
      endif()

      # Create a check- target for the directory.
      string(REPLACE ${directory} "" name_slash ${lit_suite})
      if (name_slash)
        string(REPLACE "/" "-" name_slash ${name_slash})
        string(REPLACE "\\" "-" name_dashes ${name_slash})
        string(TOLOWER "${project}${name_dashes}" name_var)
        add_lit_target("check-${name_var}" "Running lit suite ${lit_suite}"
          ${lit_suite}
          PARAMS ${ARG_PARAMS}
          DEPENDS ${ARG_DEPENDS}
          ARGS ${ARG_ARGS}
        )
      endif()
    endforeach()
  endif()
endfunction()

function(llvm_install_library_symlink name dest type)
  cmake_parse_arguments(ARG "ALWAYS_GENERATE" "COMPONENT" "" ${ARGN})
  foreach(path ${CMAKE_MODULE_PATH})
    if(EXISTS ${path}/LLVMInstallSymlink.cmake)
      set(INSTALL_SYMLINK ${path}/LLVMInstallSymlink.cmake)
      break()
    endif()
  endforeach()

  set(component ${ARG_COMPONENT})
  if(NOT component)
    set(component ${name})
  endif()

  set(full_name ${CMAKE_${type}_LIBRARY_PREFIX}${name}${CMAKE_${type}_LIBRARY_SUFFIX})
  set(full_dest ${CMAKE_${type}_LIBRARY_PREFIX}${dest}${CMAKE_${type}_LIBRARY_SUFFIX})

  set(output_dir lib${LLVM_LIBDIR_SUFFIX})
  if(WIN32 AND "${type}" STREQUAL "SHARED")
    set(output_dir bin)
  endif()

  install(SCRIPT ${INSTALL_SYMLINK}
          CODE "install_symlink(${full_name} ${full_dest} ${output_dir})"
          COMPONENT ${component})

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT ARG_ALWAYS_GENERATE)
    add_custom_target(install-${name}
                      DEPENDS ${name} ${dest} install-${dest}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${name}
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
  endif()
endfunction()

function(llvm_install_symlink name dest)
  cmake_parse_arguments(ARG "ALWAYS_GENERATE" "COMPONENT" "" ${ARGN})
  foreach(path ${CMAKE_MODULE_PATH})
    if(EXISTS ${path}/LLVMInstallSymlink.cmake)
      set(INSTALL_SYMLINK ${path}/LLVMInstallSymlink.cmake)
      break()
    endif()
  endforeach()

  if(ARG_COMPONENT)
    set(component ${ARG_COMPONENT})
  else()
    if(ARG_ALWAYS_GENERATE)
      set(component ${dest})
    else()
      set(component ${name})
    endif()
  endif()

  set(full_name ${name}${CMAKE_EXECUTABLE_SUFFIX})
  set(full_dest ${dest}${CMAKE_EXECUTABLE_SUFFIX})

  install(SCRIPT ${INSTALL_SYMLINK}
          CODE "install_symlink(${full_name} ${full_dest} ${LLVM_TOOLS_INSTALL_DIR})"
          COMPONENT ${component})

  if (NOT CMAKE_CONFIGURATION_TYPES AND NOT ARG_ALWAYS_GENERATE)
    add_custom_target(install-${name}
                      DEPENDS ${name} ${dest} install-${dest}
                      COMMAND "${CMAKE_COMMAND}"
                              -DCMAKE_INSTALL_COMPONENT=${name}
                              -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
  endif()
endfunction()

function(add_llvm_tool_symlink link_name target)
  cmake_parse_arguments(ARG "ALWAYS_GENERATE" "OUTPUT_DIR" "" ${ARGN})
  set(dest_binary "$<TARGET_FILE:${target}>")

  # This got a bit gross... For multi-configuration generators the target
  # properties return the resolved value of the string, not the build system
  # expression. To reconstruct the platform-agnostic path we have to do some
  # magic. First we grab one of the types, and a type-specific path. Then from
  # the type-specific path we find the last occurrence of the type in the path,
  # and replace it with CMAKE_CFG_INTDIR. This allows the build step to be type
  # agnostic again. 
  if(NOT ARG_OUTPUT_DIR)
    # If you're not overriding the OUTPUT_DIR, we can make the link relative in
    # the same directory.
    if(UNIX)
      set(dest_binary "$<TARGET_FILE_NAME:${target}>")
    endif()
    if(CMAKE_CONFIGURATION_TYPES)
      list(GET CMAKE_CONFIGURATION_TYPES 0 first_type)
      string(TOUPPER ${first_type} first_type_upper)
      set(first_type_suffix _${first_type_upper})
    endif()
    get_target_property(target_type ${target} TYPE)
    if(${target_type} STREQUAL "STATIC_LIBRARY")
      get_target_property(ARG_OUTPUT_DIR ${target} ARCHIVE_OUTPUT_DIRECTORY${first_type_suffix})
    elseif(UNIX AND ${target_type} STREQUAL "SHARED_LIBRARY")
      get_target_property(ARG_OUTPUT_DIR ${target} LIBRARY_OUTPUT_DIRECTORY${first_type_suffix})
    else()
      get_target_property(ARG_OUTPUT_DIR ${target} RUNTIME_OUTPUT_DIRECTORY${first_type_suffix})
    endif()
    if(CMAKE_CONFIGURATION_TYPES)
      string(FIND "${ARG_OUTPUT_DIR}" "/${first_type}/" type_start REVERSE)
      string(SUBSTRING "${ARG_OUTPUT_DIR}" 0 ${type_start} path_prefix)
      string(SUBSTRING "${ARG_OUTPUT_DIR}" ${type_start} -1 path_suffix)
      string(REPLACE "/${first_type}/" "/${CMAKE_CFG_INTDIR}/"
             path_suffix ${path_suffix})
      set(ARG_OUTPUT_DIR ${path_prefix}${path_suffix})
    endif()
  endif()

  if(UNIX)
    set(LLVM_LINK_OR_COPY create_symlink)
  else()
    set(LLVM_LINK_OR_COPY copy)
  endif()

  set(output_path "${ARG_OUTPUT_DIR}/${link_name}${CMAKE_EXECUTABLE_SUFFIX}")

  set(target_name ${link_name})
  if(TARGET ${link_name})
    set(target_name ${link_name}-link)
  endif()


  if(ARG_ALWAYS_GENERATE)
    set_property(DIRECTORY APPEND PROPERTY
      ADDITIONAL_MAKE_CLEAN_FILES ${dest_binary})
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${dest_binary}" "${output_path}")
  else()
    add_custom_command(OUTPUT ${output_path}
                     COMMAND ${CMAKE_COMMAND} -E ${LLVM_LINK_OR_COPY} "${dest_binary}" "${output_path}"
                     DEPENDS ${target})
    add_custom_target(${target_name} ALL DEPENDS ${target} ${output_path})
    set_target_properties(${target_name} PROPERTIES FOLDER Tools)

    # Make sure both the link and target are toolchain tools
    if (${link_name} IN_LIST LLVM_TOOLCHAIN_TOOLS AND ${target} IN_LIST LLVM_TOOLCHAIN_TOOLS)
      set(TOOL_IS_TOOLCHAIN ON)
    endif()

    if ((TOOL_IS_TOOLCHAIN OR NOT LLVM_INSTALL_TOOLCHAIN_ONLY) AND LLVM_BUILD_TOOLS)
      llvm_install_symlink(${link_name} ${target})
    endif()
  endif()
endfunction()

function(llvm_externalize_debuginfo name)
  if(NOT LLVM_EXTERNALIZE_DEBUGINFO)
    return()
  endif()

  if(NOT LLVM_EXTERNALIZE_DEBUGINFO_SKIP_STRIP)
    if(APPLE)
      set(strip_command COMMAND xcrun strip -Sxl $<TARGET_FILE:${name}>)
    else()
      set(strip_command COMMAND strip -gx $<TARGET_FILE:${name}>)
    endif()
  endif()

  if(APPLE)
    if(CMAKE_CXX_FLAGS MATCHES "-flto"
      OR CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE} MATCHES "-flto")

      set(lto_object ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${name}-lto.o)
      set_property(TARGET ${name} APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,-object_path_lto,${lto_object}")
    endif()
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND xcrun dsymutil $<TARGET_FILE:${name}>
      ${strip_command}
      )
  else()
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND objcopy --only-keep-debug $<TARGET_FILE:${name}> $<TARGET_FILE:${name}>.debug
      ${strip_command} -R .gnu_debuglink
      COMMAND objcopy --add-gnu-debuglink=$<TARGET_FILE:${name}>.debug $<TARGET_FILE:${name}>
      )
  endif()
endfunction()

function(llvm_setup_rpath name)
  if(CMAKE_INSTALL_RPATH)
    return()
  endif()

  if(LLVM_INSTALL_PREFIX AND NOT (LLVM_INSTALL_PREFIX STREQUAL CMAKE_INSTALL_PREFIX))
    set(extra_libdir ${LLVM_LIBRARY_DIR})
  elseif(LLVM_BUILD_LIBRARY_DIR)
    set(extra_libdir ${LLVM_LIBRARY_DIR})
  endif()

  if (APPLE)
    set(_install_name_dir INSTALL_NAME_DIR "@rpath")
    set(_install_rpath "@loader_path/../lib" ${extra_libdir})
  elseif(UNIX)
    set(_install_rpath "\$ORIGIN/../lib${LLVM_LIBDIR_SUFFIX}" ${extra_libdir})
    if(${CMAKE_SYSTEM_NAME} MATCHES "(FreeBSD|DragonFly)")
      set_property(TARGET ${name} APPEND_STRING PROPERTY
                   LINK_FLAGS " -Wl,-z,origin ")
    elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux" AND NOT LLVM_LINKER_IS_GOLD)
      # $ORIGIN is not interpreted at link time by ld.bfd
      set_property(TARGET ${name} APPEND_STRING PROPERTY
                   LINK_FLAGS " -Wl,-rpath-link,${LLVM_LIBRARY_OUTPUT_INTDIR} ")
    endif()
  else()
    return()
  endif()

  set_target_properties(${name} PROPERTIES
                        BUILD_WITH_INSTALL_RPATH On
                        INSTALL_RPATH "${_install_rpath}"
                        ${_install_name_dir})
endfunction()

function(setup_dependency_debugging name)
  if(NOT LLVM_DEPENDENCY_DEBUGGING)
    return()
  endif()

  if("intrinsics_gen" IN_LIST ARGN)
    return()
  endif()

  set(deny_attributes_gen "(deny file* (literal \"${LLVM_BINARY_DIR}/include/llvm/IR/Attributes.gen\"))")
  set(deny_intrinsics_gen "(deny file* (literal \"${LLVM_BINARY_DIR}/include/llvm/IR/Intrinsics.gen\"))")

  set(sandbox_command "sandbox-exec -p '(version 1) (allow default) ${deny_attributes_gen} ${deny_intrinsics_gen}'")
  set_target_properties(${name} PROPERTIES RULE_LAUNCH_COMPILE ${sandbox_command})
endfunction()
