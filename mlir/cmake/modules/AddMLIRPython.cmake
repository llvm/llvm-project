################################################################################
# Python modules
# MLIR's Python modules are both directly used by the core project and are
# available for use and embedding into external projects (in their own
# namespace and with their own deps). In order to facilitate this, python
# artifacts are split between declarations, which make a subset of
# things available to be built and "add", which in line with the normal LLVM
# nomenclature, adds libraries.
################################################################################

# Function: declare_mlir_python_sources
# Declares pure python sources as part of a named grouping that can be built
# later.
# Arguments:
#   ROOT_DIR: Directory where the python namespace begins (defaults to
#     CMAKE_CURRENT_SOURCE_DIR). For non-relocatable sources, this will
#     typically just be the root of the python source tree (current directory).
#     For relocatable sources, this will point deeper into the directory that
#     can be relocated. For generated sources, can be relative to
#     CMAKE_CURRENT_BINARY_DIR. Generated and non generated sources cannot be
#     mixed.
#   ADD_TO_PARENT: Adds this source grouping to a previously declared source
#     grouping. Source groupings form a DAG.
#   SOURCES: List of specific source files relative to ROOT_DIR to include.
#   SOURCES_GLOB: List of glob patterns relative to ROOT_DIR to include.
#   DEST_PREFIX: Destination prefix to prepend to files in the python
#     package directory namespace.
function(declare_mlir_python_sources name)
  cmake_parse_arguments(ARG
    ""
    "ROOT_DIR;ADD_TO_PARENT;DEST_PREFIX"
    "SOURCES;SOURCES_GLOB"
    ${ARGN})

  if(NOT ARG_ROOT_DIR)
    set(ARG_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  set(_install_destination "src/python/${name}")

  # Process the glob.
  set(_glob_sources)
  if(ARG_SOURCES_GLOB)
    set(_glob_spec ${ARG_SOURCES_GLOB})
    list(TRANSFORM _glob_spec PREPEND "${ARG_ROOT_DIR}/")
    file(GLOB_RECURSE _glob_sources
      RELATIVE "${ARG_ROOT_DIR}"
      ${_glob_spec}
    )
    list(APPEND ARG_SOURCES ${_glob_sources})
  endif()

  # We create a custom target to carry properties and dependencies for
  # generated sources.
  add_library(${name} INTERFACE)
  set(_file_depends "${ARG_SOURCES}")
  list(TRANSFORM _file_depends PREPEND "${ARG_ROOT_DIR}/")
  set_target_properties(${name} PROPERTIES
    # Yes: Leading-lowercase property names are load bearing and the recommended
    # way to do this: https://gitlab.kitware.com/cmake/cmake/-/issues/19261
    # Note that ROOT_DIR and FILE_DEPENDS are not exported because they are
    # only relevant to in-tree uses.
    EXPORT_PROPERTIES "mlir_python_SOURCES_TYPE;mlir_python_DEST_PREFIX;mlir_python_ROOT_DIR;mlir_python_SOURCES;mlir_python_DEPENDS"
    mlir_python_SOURCES_TYPE pure
    mlir_python_ROOT_DIR "${ARG_ROOT_DIR}"
    mlir_python_DEST_PREFIX "${ARG_DEST_PREFIX}"
    mlir_python_SOURCES "${ARG_SOURCES}"
    mlir_python_FILE_DEPENDS "${_file_depends}"
    mlir_python_DEPENDS ""
  )
  # Note that an "include" directory has no meaning to such faux targets,
  # but it is a CMake supported way to specify a directory search list in a
  # way that works both in-tree and out. It has some super powers which are
  # not possible to emulate with custom properties (because of the prohibition
  # on using generator expressions in exported custom properties and the
  # special dispensation for $<INSTALL_PREFIX>).
  target_include_directories(${name} INTERFACE
    "$<BUILD_INTERFACE:${ARG_ROOT_DIR}>"
    "$<INSTALL_INTERFACE:${_install_destination}>"
  )

  # Add to parent.
  if(ARG_ADD_TO_PARENT)
    set_property(TARGET ${ARG_ADD_TO_PARENT} APPEND PROPERTY mlir_python_DEPENDS ${name})
  endif()

  # Install.
  set_property(GLOBAL APPEND PROPERTY MLIR_EXPORTS ${name})
  if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    _mlir_python_install_sources(
      ${name} "${ARG_ROOT_DIR}" "${_install_destination}"
      ${ARG_SOURCES}
    )
  endif()
endfunction()

# Function: declare_mlir_python_extension
# Declares a buildable python extension from C++ source files. The built
# module is considered a python source file and included as everything else.
# Arguments:
#   ROOT_DIR: Root directory where sources are interpreted relative to.
#     Defaults to CMAKE_CURRENT_SOURCE_DIR.
#   MODULE_NAME: Local import name of the module (i.e. "_mlir").
#   ADD_TO_PARENT: Same as for declare_mlir_python_sources.
#   SOURCES: C++ sources making up the module.
#   PRIVATE_LINK_LIBS: List of libraries to link in privately to the module
#     regardless of how it is included in the project (generally should be
#     static libraries that can be included with hidden visibility).
#   EMBED_CAPI_LINK_LIBS: Dependent CAPI libraries that this extension depends
#     on. These will be collected for all extensions and put into an
#     aggregate dylib that is linked against.
function(declare_mlir_python_extension name)
  cmake_parse_arguments(ARG
    ""
    "ROOT_DIR;MODULE_NAME;ADD_TO_PARENT"
    "SOURCES;PRIVATE_LINK_LIBS;EMBED_CAPI_LINK_LIBS"
    ${ARGN})

  if(NOT ARG_ROOT_DIR)
    set(ARG_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  set(_install_destination "src/python/${name}")

  add_library(${name} INTERFACE)
  set_target_properties(${name} PROPERTIES
    # Yes: Leading-lowercase property names are load bearing and the recommended
    # way to do this: https://gitlab.kitware.com/cmake/cmake/-/issues/19261
    # Note that ROOT_DIR and FILE_DEPENDS are not exported because they are
    # only relevant to in-tree uses.
    EXPORT_PROPERTIES "mlir_python_SOURCES_TYPE;mlir_python_ROOT_DIR;mlir_python_EXTENSION_MODULE_NAME;mlir_python_CPP_SOURCES;mlir_python_PRIVATE_LINK_LIBS;mlir_python_EMBED_CAPI_LINK_LIBS;mlir_python_DEPENDS"
    mlir_python_SOURCES_TYPE extension
    mlir_python_ROOT_DIR "${ARG_ROOT_DIR}"
    mlir_python_EXTENSION_MODULE_NAME "${ARG_MODULE_NAME}"
    mlir_python_CPP_SOURCES "${ARG_SOURCES}"
    mlir_python_PRIVATE_LINK_LIBS "${ARG_PRIVATE_LINK_LIBS}"
    mlir_python_EMBED_CAPI_LINK_LIBS "${ARG_EMBED_CAPI_LINK_LIBS}"
    mlir_python_FILE_DEPENDS ""
    mlir_python_DEPENDS ""
  )
  # Note that an "include" directory has no meaning to such faux targets,
  # but it is a CMake supported way to specify an install-prefix relative
  # directory. It has some super powers which are not possible to emulate
  # with custom properties (because of the prohibition on using generator
  # expressions in exported custom properties and the special dispensation
  # for $<INSTALL_PREFIX> and $<INSTALL_INTERFACE>). On imported targets,
  # this is used as a single value, not as a list, so it must only have one
  # item in it.
  target_include_directories(${name} INTERFACE
    "$<INSTALL_INTERFACE:${_install_destination}>"
  )

  # Add to parent.
  if(ARG_ADD_TO_PARENT)
    set_property(TARGET ${ARG_ADD_TO_PARENT} APPEND PROPERTY mlir_python_DEPENDS ${name})
  endif()

  # Install.
  set_property(GLOBAL APPEND PROPERTY MLIR_EXPORTS ${name})
  if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
    _mlir_python_install_sources(
      ${name} "${ARG_ROOT_DIR}" "src/python/${name}"
      ${ARG_SOURCES}
    )
  endif()
endfunction()

function(_mlir_python_install_sources name source_root_dir destination)
  foreach(source_relative_path ${ARGN})
    # Transform "a/b/c.py" -> "${install_prefix}/a/b" for installation.
    get_filename_component(
      dest_relative_path "${source_relative_path}" DIRECTORY
      BASE_DIR "${source_root_dir}"
    )
    install(
      FILES "${source_root_dir}/${source_relative_path}"
      DESTINATION "${destination}/${dest_relative_path}"
      COMPONENT "${name}"
    )
  endforeach()
  get_target_export_arg(${name} MLIR export_to_mlirtargets UMBRELLA mlir-libraries)
  install(TARGETS ${name}
    COMPONENT ${name}
    ${export_to_mlirtargets}
  )
endfunction()

# Function: add_mlir_python_modules
# Adds python modules to a project, building them from a list of declared
# source groupings (see declare_mlir_python_sources and
# declare_mlir_python_extension). One of these must be called for each
# packaging root in use.
# Arguments:
#   ROOT_PREFIX: The directory in the build tree to emit sources. This will
#     typically be something like ${MY_BINARY_DIR}/python_packages/foobar
#     for non-relocatable modules or a deeper directory tree for relocatable.
#   INSTALL_PREFIX: Prefix into the install tree for installing the package.
#     Typically mirrors the path above but without an absolute path.
#   DECLARED_SOURCES: List of declared source groups to include. The entire
#     DAG of source modules is included.
#   COMMON_CAPI_LINK_LIBS: List of dylibs (typically one) to make every
#     extension depend on (see mlir_python_add_common_capi_library).
function(add_mlir_python_modules name)
  cmake_parse_arguments(ARG
    ""
    "ROOT_PREFIX;INSTALL_PREFIX;COMMON_CAPI_LINK_LIBS"
    "DECLARED_SOURCES"
    ${ARGN})
  # Helper to process an individual target.
  function(_process_target modules_target sources_target)
    get_target_property(_source_type ${sources_target} mlir_python_SOURCES_TYPE)

    get_target_property(_python_root_dir ${sources_target} mlir_python_ROOT_DIR)
    if(NOT _python_root_dir)
      message(FATAL_ERROR "Target ${sources_target} lacks mlir_python_ROOT_DIR property")
    endif()

    if(_source_type STREQUAL "pure")
      # Pure python sources to link into the tree.
      get_target_property(_python_sources ${sources_target} mlir_python_SOURCES)
      get_target_property(_specified_dest_prefix ${sources_target} mlir_python_DEST_PREFIX)
      foreach(_source_relative_path ${_python_sources})
        set(_dest_relative_path "${_source_relative_path}")
        if(_specified_dest_prefix)
          set(_dest_relative_path "${_specified_dest_prefix}/${_dest_relative_path}")
        endif()
        set(_src_path "${_python_root_dir}/${_source_relative_path}")
        set(_dest_path "${ARG_ROOT_PREFIX}/${_dest_relative_path}")

        get_filename_component(_dest_dir "${_dest_path}" DIRECTORY)
        get_filename_component(_install_path "${ARG_INSTALL_PREFIX}/${_dest_relative_path}" DIRECTORY)

        file(MAKE_DIRECTORY "${_dest_dir}")

        # On Windows create_symlink requires special permissions. Use copy_if_different instead.
        if(CMAKE_HOST_WIN32)
          set(_link_or_copy copy_if_different)
        else()
          set(_link_or_copy create_symlink)
        endif()

        add_custom_command(
          TARGET ${modules_target} PRE_BUILD
          COMMENT "Copying python source ${_src_path} -> ${_dest_path}"
          DEPENDS "${_src_path}"
          BYPRODUCTS "${_dest_path}"
          COMMAND "${CMAKE_COMMAND}" -E ${_link_or_copy}
              "${_src_path}" "${_dest_path}"
        )
        install(
          FILES "${_src_path}"
          DESTINATION "${_install_path}"
          COMPONENT ${modules_target}
        )
      endforeach()
    elseif(_source_type STREQUAL "extension")
      # Native CPP extension.
      get_target_property(_module_name ${sources_target} mlir_python_EXTENSION_MODULE_NAME)
      get_target_property(_cpp_sources ${sources_target} mlir_python_CPP_SOURCES)
      get_target_property(_private_link_libs ${sources_target} mlir_python_PRIVATE_LINK_LIBS)
      # Transform relative source to based on root dir.
      list(TRANSFORM _cpp_sources PREPEND "${_python_root_dir}/")
      set(_extension_target "${name}.extension.${_module_name}.dso")
      add_mlir_python_extension(${_extension_target} "${_module_name}"
        INSTALL_COMPONENT ${modules_target}
        INSTALL_DIR "${ARG_INSTALL_PREFIX}/_mlir_libs"
        OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/_mlir_libs"
        SOURCES ${_cpp_sources}
        LINK_LIBS PRIVATE
          ${_private_link_libs}
          ${ARG_COMMON_CAPI_LINK_LIBS}
      )
      add_dependencies(${name} ${_extension_target})
      mlir_python_setup_extension_rpath(${_extension_target})
    else()
      message(SEND_ERROR "Unrecognized source type '${_source_type}' for python source target ${sources_target}")
      return()
    endif()
  endfunction()

  _flatten_mlir_python_targets(_flat_targets ${ARG_DECLARED_SOURCES})
  # Collect dependencies.
  set(_depends)
  foreach(sources_target ${_flat_targets})
    get_target_property(_local_depends ${sources_target} mlir_python_FILE_DEPENDS)
    if(_local_depends)
      list(APPEND _depends ${_local_depends})
    endif()
  endforeach()

  # Build the modules target.
  add_custom_target(${name} ALL DEPENDS ${_depends})
  foreach(sources_target ${_flat_targets})
    _process_target(${name} ${sources_target})
  endforeach()

  # Create an install target.
  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(
      install-${name}
      DEPENDS ${name}
      COMPONENT ${name})
  endif()
endfunction()

# Function: declare_mlir_dialect_python_bindings
# Helper to generate source groups for dialects, including both static source
# files and a TD_FILE to generate wrappers.
#
# This will generate a source group named ${ADD_TO_PARENT}.${DIALECT_NAME}.
#
# Arguments:
#   ROOT_DIR: Same as for declare_mlir_python_sources().
#   ADD_TO_PARENT: Same as for declare_mlir_python_sources(). Unique names
#     for the subordinate source groups are derived from this.
#   TD_FILE: Tablegen file to generate source for (relative to ROOT_DIR).
#   DIALECT_NAME: Python name of the dialect.
#   SOURCES: Same as declare_mlir_python_sources().
#   SOURCES_GLOB: Same as declare_mlir_python_sources().
#   DEPENDS: Additional dependency targets.
function(declare_mlir_dialect_python_bindings)
  cmake_parse_arguments(ARG
    ""
    "ROOT_DIR;ADD_TO_PARENT;TD_FILE;DIALECT_NAME"
    "SOURCES;SOURCES_GLOB;DEPENDS"
    ${ARGN})
  # Sources.
  set(_dialect_target "${ARG_ADD_TO_PARENT}.${ARG_DIALECT_NAME}")
  declare_mlir_python_sources(${_dialect_target}
    ROOT_DIR "${ARG_ROOT_DIR}"
    ADD_TO_PARENT "${ARG_ADD_TO_PARENT}"
    SOURCES "${ARG_SOURCES}"
    SOURCES_GLOB "${ARG_SOURCES_GLOB}"
  )

  # Tablegen
  if(ARG_TD_FILE)
    set(tblgen_target "${ARG_ADD_TO}.${ARG_DIALECT_NAME}.tablegen")
    set(td_file "${ARG_ROOT_DIR}/${ARG_TD_FILE}")
    get_filename_component(relative_td_directory "${ARG_TD_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${relative_td_directory}")
    set(dialect_filename "${relative_td_directory}/_${ARG_DIALECT_NAME}_ops_gen.py")
    set(LLVM_TARGET_DEFINITIONS ${td_file})
    mlir_tablegen("${dialect_filename}" -gen-python-op-bindings
                  -bind-dialect=${ARG_DIALECT_NAME})
    add_public_tablegen_target(${tblgen_target})
    if(ARG_DEPENDS)
      add_dependencies(${tblgen_target} ${ARG_DEPENDS})
    endif()

    # Generated.
    declare_mlir_python_sources("${ARG_ADD_TO_PARENT}.${ARG_DIALECT_NAME}.ops_gen"
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
      ADD_TO_PARENT "${_dialect_target}"
      SOURCES "${dialect_filename}"
    )
  endif()
endfunction()

# Function: mlir_python_setup_extension_rpath
# Sets RPATH properties on a target, assuming that it is being output to
# an _mlir_libs directory with all other libraries. For static linkage,
# the RPATH will just be the origin. If linking dynamically, then the LLVM
# library directory will be added.
# Arguments:
#   RELATIVE_INSTALL_ROOT: If building dynamically, an RPATH entry will be
#     added to the install tree lib/ directory by first traversing this
#     path relative to the installation location. Typically a number of ".."
#     entries, one for each level of the install path.
function(mlir_python_setup_extension_rpath target)
  cmake_parse_arguments(ARG
    ""
    "RELATIVE_INSTALL_ROOT"
    ""
    ${ARGN})

  # RPATH handling.
  # For the build tree, include the LLVM lib directory and the current
  # directory for RPATH searching. For install, just the current directory
  # (assumes that needed dependencies have been installed).
  if(NOT APPLE AND NOT UNIX)
    return()
  endif()

  set(_origin_prefix "\$ORIGIN")
  if(APPLE)
    set(_origin_prefix "@loader_path")
  endif()
  set_target_properties(${target} PROPERTIES
    BUILD_WITH_INSTALL_RPATH OFF
    BUILD_RPATH "${_origin_prefix}"
    INSTALL_RPATH "${_origin_prefix}"
  )

  # For static builds, that is all that is needed: all dependencies will be in
  # the one directory. For shared builds, then we also need to add the global
  # lib directory. This will be absolute for the build tree and relative for
  # install.
  # When we have access to CMake >= 3.20, there is a helper to calculate this.
  if(BUILD_SHARED_LIBS AND ARG_RELATIVE_INSTALL_ROOT)
    get_filename_component(_real_lib_dir "${LLVM_LIBRARY_OUTPUT_INTDIR}" REALPATH)
    set_property(TARGET ${target} APPEND PROPERTY
      BUILD_RPATH "${_real_lib_dir}")
    set_property(TARGET ${target} APPEND PROPERTY
      INSTALL_RPATH "${_origin_prefix}/${ARG_RELATIVE_INSTALL_ROOT}/lib${LLVM_LIBDIR_SUFFIX}")
  endif()
endfunction()

# Function: add_mlir_python_common_capi_library
# Adds a shared library which embeds dependent CAPI libraries needed to link
# all extensions.
# Arguments:
#   INSTALL_COMPONENT: Name of the install component. Typically same as the
#     target name passed to add_mlir_python_modules().
#   INSTALL_DESTINATION: Prefix into the install tree in which to install the
#     library.
#   OUTPUT_DIRECTORY: Full path in the build tree in which to create the
#     library. Typically, this will be the common _mlir_libs directory where
#     all extensions are emitted.
#   RELATIVE_INSTALL_ROOT: See mlir_python_setup_extension_rpath().
#   DECLARED_SOURCES: Source groups from which to discover dependent
#     EMBED_CAPI_LINK_LIBS.
#   EMBED_LIBS: Additional libraries to embed (must be built with OBJECTS and
#     have an "obj.${name}" object library associated).
function(add_mlir_python_common_capi_library name)
  cmake_parse_arguments(ARG
    ""
    "INSTALL_COMPONENT;INSTALL_DESTINATION;OUTPUT_DIRECTORY;RELATIVE_INSTALL_ROOT"
    "DECLARED_SOURCES;EMBED_LIBS"
    ${ARGN})
  # Collect all explicit and transitive embed libs.
  set(_embed_libs ${ARG_EMBED_LIBS})
  _flatten_mlir_python_targets(_all_source_targets ${ARG_DECLARED_SOURCES})
  foreach(t ${_all_source_targets})
    get_target_property(_local_embed_libs ${t} mlir_python_EMBED_CAPI_LINK_LIBS)
    if(_local_embed_libs)
      list(APPEND _embed_libs ${_local_embed_libs})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _embed_libs)

  # Generate the aggregate .so that everything depends on.
  add_mlir_aggregate(${name}
    SHARED
    DISABLE_INSTALL
    EMBED_LIBS ${_embed_libs}
  )

  if(MSVC)
    set_property(TARGET ${name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
    BINARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
    # Needed for windows (and don't hurt others).
    RUNTIME_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
  )
  mlir_python_setup_extension_rpath(${name}
    RELATIVE_INSTALL_ROOT "${ARG_RELATIVE_INSTALL_ROOT}"
  )
  install(TARGETS ${name}
    COMPONENT ${ARG_INSTALL_COMPONENT}
    LIBRARY DESTINATION "${ARG_INSTALL_DESTINATION}"
    RUNTIME DESTINATION "${ARG_INSTALL_DESTINATION}"
  )

endfunction()

function(_flatten_mlir_python_targets output_var)
  set(_flattened)
  foreach(t ${ARGN})
    get_target_property(_source_type ${t} mlir_python_SOURCES_TYPE)
    get_target_property(_depends ${t} mlir_python_DEPENDS)
    if(_source_type)
      list(APPEND _flattened "${t}")
      if(_depends)
        _flatten_mlir_python_targets(_local_flattened ${_depends})
        list(APPEND _flattened ${_local_flattened})
      endif()
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _flattened)
  set(${output_var} "${_flattened}" PARENT_SCOPE)
endfunction()

################################################################################
# Build python extension
################################################################################
function(add_mlir_python_extension libname extname)
  cmake_parse_arguments(ARG
  ""
  "INSTALL_COMPONENT;INSTALL_DIR;OUTPUT_DIRECTORY"
  "SOURCES;LINK_LIBS"
  ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR " Unhandled arguments to add_mlir_python_extension(${libname}, ... : ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if("${ARG_SOURCES}" STREQUAL "")
    message(FATAL_ERROR " Missing SOURCES argument to add_mlir_python_extension(${libname}, ...")
  endif()

  # The actual extension library produces a shared-object or DLL and has
  # sources that must be compiled in accordance with pybind11 needs (RTTI and
  # exceptions).
  pybind11_add_module(${libname}
    ${ARG_SOURCES}
  )

  # The extension itself must be compiled with RTTI and exceptions enabled.
  # Also, some warning classes triggered by pybind11 are disabled.
  target_compile_options(${libname} PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      # Enable RTTI and exceptions.
      -frtti -fexceptions
    >
    $<$<CXX_COMPILER_ID:MSVC>:
      # Enable RTTI and exceptions.
      /EHsc /GR>
  )

  # Configure the output to match python expectations.
  set_target_properties(
    ${libname} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
    OUTPUT_NAME "${extname}"
    NO_SONAME ON
  )

  if(WIN32)
    # Need to also set the RUNTIME_OUTPUT_DIRECTORY on Windows in order to
    # control where the .dll gets written.
    set_target_properties(
      ${libname} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
      ARCHIVE_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
    )
  endif()

  target_link_libraries(${libname}
    PRIVATE
    ${ARG_LINK_LIBS}
  )

  target_link_options(${libname}
    PRIVATE
      # On Linux, disable re-export of any static linked libraries that
      # came through.
      $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
  )

  ################################################################################
  # Install
  ################################################################################
  if(ARG_INSTALL_DIR)
    install(TARGETS ${libname}
      COMPONENT ${ARG_INSTALL_COMPONENT}
      LIBRARY DESTINATION ${ARG_INSTALL_DIR}
      ARCHIVE DESTINATION ${ARG_INSTALL_DIR}
      # NOTE: Even on DLL-platforms, extensions go in the lib directory tree.
      RUNTIME DESTINATION ${ARG_INSTALL_DIR}
    )
  endif()
endfunction()
