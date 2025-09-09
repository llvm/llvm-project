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
function(declare_mlir_python_sources name)
  cmake_parse_arguments(ARG
    ""
    "ROOT_DIR;ADD_TO_PARENT"
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
  set_target_properties(${name} PROPERTIES
    # Yes: Leading-lowercase property names are load bearing and the recommended
    # way to do this: https://gitlab.kitware.com/cmake/cmake/-/issues/19261
    EXPORT_PROPERTIES "mlir_python_SOURCES_TYPE;mlir_python_DEPENDS"
    mlir_python_SOURCES_TYPE pure
    mlir_python_DEPENDS ""
  )

  # Use the interface include directories and sources on the target to carry the
  # properties we would like to export. These support generator expressions and
  # allow us to properly specify paths in both the local build and install scenarios.
  # The one caveat here is that because we don't directly build against the interface
  # library, we need to specify the INCLUDE_DIRECTORIES and SOURCES properties as well
  # via private properties because the evaluation would happen at configuration time
  # instead of build time.
  # Eventually this could be done using a FILE_SET simplifying the logic below.
  # FILE_SET is available in cmake 3.23+, so it is not an option at the moment.
  target_include_directories(${name} INTERFACE
    "$<BUILD_INTERFACE:${ARG_ROOT_DIR}>"
    "$<INSTALL_INTERFACE:${_install_destination}>"
  )
  set_property(TARGET ${name} PROPERTY INCLUDE_DIRECTORIES ${ARG_ROOT_DIR})

  if(ARG_SOURCES)
    list(TRANSFORM ARG_SOURCES PREPEND "${ARG_ROOT_DIR}/" OUTPUT_VARIABLE _build_sources)
    list(TRANSFORM ARG_SOURCES PREPEND "${_install_destination}/" OUTPUT_VARIABLE _install_sources)
    target_sources(${name}
      INTERFACE
        "$<INSTALL_INTERFACE:${_install_sources}>"
        "$<BUILD_INTERFACE:${_build_sources}>"
      PRIVATE ${_build_sources}
    )
  endif()

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

# Function: generate_type_stubs
# Turns on automatic type stub generation (via nanobind's stubgen) for extension modules.
# Arguments:
#   MODULE_NAME: The name of the extension module as specified in declare_mlir_python_extension.
#   DEPENDS_TARGET: The dso target corresponding to the extension module
#     (e.g., something like StandalonePythonModules.extension._standaloneDialectsNanobind.dso)
#   MLIR_DEPENDS_TARGET: The dso target corresponding to the main/core extension module
#     (e.g., something like StandalonePythonModules.extension._mlir.dso)
#   OUTPUT_DIR: The root output directory to emit the type stubs into.
# Outputs:
#   NB_STUBGEN_CUSTOM_TARGET: The target corresponding to generation which other targets can depend on.
function(generate_type_stubs MODULE_NAME DEPENDS_TARGET MLIR_DEPENDS_TARGET OUTPUT_DIR)
  cmake_parse_arguments(ARG
    ""
    ""
    "OUTPUTS"
    ${ARGN})
  if(EXISTS ${nanobind_DIR}/../src/stubgen.py)
    set(NB_STUBGEN "${nanobind_DIR}/../src/stubgen.py")
  elseif(EXISTS ${nanobind_DIR}/../stubgen.py)
    set(NB_STUBGEN "${nanobind_DIR}/../stubgen.py")
  else()
    message(FATAL_ERROR "generate_type_stubs(): could not locate 'stubgen.py'!")
  endif()
  file(REAL_PATH "${NB_STUBGEN}" NB_STUBGEN)

  set(_module "${MLIR_PYTHON_PACKAGE_PREFIX}._mlir_libs.${MODULE_NAME}")
  file(REAL_PATH "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/.." _import_path)

  set(NB_STUBGEN_CMD
      "${Python_EXECUTABLE}"
      "${NB_STUBGEN}"
      --module
      "${_module}"
      -i
      "${_import_path}"
      --recursive
      --include-private
      --output-dir
      "${OUTPUT_DIR}")

  list(TRANSFORM ARG_OUTPUTS PREPEND "${OUTPUT_DIR}/" OUTPUT_VARIABLE _generated_type_stubs)
  add_custom_command(
    OUTPUT ${_generated_type_stubs}
    COMMAND ${NB_STUBGEN_CMD}
    WORKING_DIRECTORY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}"
    DEPENDS
      "${MLIR_DEPENDS_TARGET}.extension._mlir.dso"
      "${MLIR_DEPENDS_TARGET}.sources.MLIRPythonSources.Core.Python"
      "${DEPENDS_TARGET}"
  )
  set(_name "MLIRPythonModuleStubs_${_module}")
  add_custom_target("${_name}" ALL DEPENDS ${_generated_type_stubs})
  set(NB_STUBGEN_CUSTOM_TARGET "${_name}" PARENT_SCOPE)
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
#   PYTHON_BINDINGS_LIBRARY: Either pybind11 or nanobind.
#   GENERATE_TYPE_STUBS: List of generated type stubs expected from stubgen relative to _mlir_libs.
function(declare_mlir_python_extension name)
  cmake_parse_arguments(ARG
    ""
    "ROOT_DIR;MODULE_NAME;ADD_TO_PARENT;PYTHON_BINDINGS_LIBRARY"
    "SOURCES;PRIVATE_LINK_LIBS;EMBED_CAPI_LINK_LIBS;GENERATE_TYPE_STUBS"
    ${ARGN})

  if(NOT ARG_ROOT_DIR)
    set(ARG_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  set(_install_destination "src/python/${name}")

  if(NOT ARG_PYTHON_BINDINGS_LIBRARY)
    set(ARG_PYTHON_BINDINGS_LIBRARY "pybind11")
  endif()

  add_library(${name} INTERFACE)
  set_target_properties(${name} PROPERTIES
    # Yes: Leading-lowercase property names are load bearing and the recommended
    # way to do this: https://gitlab.kitware.com/cmake/cmake/-/issues/19261
    EXPORT_PROPERTIES "mlir_python_SOURCES_TYPE;mlir_python_EXTENSION_MODULE_NAME;mlir_python_EMBED_CAPI_LINK_LIBS;mlir_python_DEPENDS;mlir_python_BINDINGS_LIBRARY;mlir_python_GENERATE_TYPE_STUBS"
    mlir_python_SOURCES_TYPE extension
    mlir_python_EXTENSION_MODULE_NAME "${ARG_MODULE_NAME}"
    mlir_python_EMBED_CAPI_LINK_LIBS "${ARG_EMBED_CAPI_LINK_LIBS}"
    mlir_python_DEPENDS ""
    mlir_python_BINDINGS_LIBRARY "${ARG_PYTHON_BINDINGS_LIBRARY}"
    mlir_python_GENERATE_TYPE_STUBS "${ARG_GENERATE_TYPE_STUBS}"
  )

  # Set the interface source and link_libs properties of the target
  # These properties support generator expressions and are automatically exported
  list(TRANSFORM ARG_SOURCES PREPEND "${ARG_ROOT_DIR}/" OUTPUT_VARIABLE _build_sources)
  list(TRANSFORM ARG_SOURCES PREPEND "${_install_destination}/" OUTPUT_VARIABLE _install_sources)
  target_sources(${name} INTERFACE
    "$<BUILD_INTERFACE:${_build_sources}>"
    "$<INSTALL_INTERFACE:${_install_sources}>"
  )
  target_link_libraries(${name} INTERFACE
    ${ARG_PRIVATE_LINK_LIBS}
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

function(_mlir_python_install_sources name source_root_dir destination)
  foreach(source_relative_path ${ARGN})
    # Transform "a/b/c.py" -> "${install_prefix}/a/b" for installation.
    get_filename_component(
      dest_relative_dir "${source_relative_path}" DIRECTORY
      BASE_DIR "${source_root_dir}"
    )
    install(
      FILES "${source_root_dir}/${source_relative_path}"
      DESTINATION "${destination}/${dest_relative_dir}"
      COMPONENT mlir-python-sources
    )
  endforeach()
  get_target_export_arg(${name} MLIR export_to_mlirtargets
    UMBRELLA mlir-python-sources)
  install(TARGETS ${name}
    COMPONENT mlir-python-sources
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
    "ROOT_PREFIX;INSTALL_PREFIX"
    "COMMON_CAPI_LINK_LIBS;DECLARED_SOURCES"
    ${ARGN})
  # Helper to process an individual target.
  function(_process_target modules_target sources_target)
    get_target_property(_source_type ${sources_target} mlir_python_SOURCES_TYPE)

    if(_source_type STREQUAL "pure")
      # Pure python sources to link into the tree.
      set(_pure_sources_target "${modules_target}.sources.${sources_target}")
      add_mlir_python_sources_target(${_pure_sources_target}
        INSTALL_COMPONENT ${modules_target}
        INSTALL_DIR ${ARG_INSTALL_PREFIX}
        OUTPUT_DIRECTORY ${ARG_ROOT_PREFIX}
        SOURCES_TARGETS ${sources_target}
      )
      add_dependencies(${modules_target} ${_pure_sources_target})
    elseif(_source_type STREQUAL "extension")
      # Native CPP extension.
      get_target_property(_module_name ${sources_target} mlir_python_EXTENSION_MODULE_NAME)
      get_target_property(_bindings_library ${sources_target} mlir_python_BINDINGS_LIBRARY)
      # Transform relative source to based on root dir.
      set(_extension_target "${modules_target}.extension.${_module_name}.dso")
      add_mlir_python_extension(${_extension_target} "${_module_name}"
        INSTALL_COMPONENT ${modules_target}
        INSTALL_DIR "${ARG_INSTALL_PREFIX}/_mlir_libs"
        OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/_mlir_libs"
        PYTHON_BINDINGS_LIBRARY ${_bindings_library}
        LINK_LIBS PRIVATE
          ${sources_target}
          ${ARG_COMMON_CAPI_LINK_LIBS}
      )
      add_dependencies(${modules_target} ${_extension_target})
      mlir_python_setup_extension_rpath(${_extension_target})
      get_target_property(_generate_type_stubs ${sources_target} mlir_python_GENERATE_TYPE_STUBS)
      if(_generate_type_stubs)
        generate_type_stubs(
          ${_module_name}
          ${_extension_target}
          ${name}
          "${CMAKE_CURRENT_SOURCE_DIR}/mlir/_mlir_libs"
          OUTPUTS "${_generate_type_stubs}"
        )
        add_dependencies("${modules_target}" "${NB_STUBGEN_CUSTOM_TARGET}")
        set(_stubgen_target "${MLIR_PYTHON_PACKAGE_PREFIX}.${_module_name}_type_stub_gen")
        declare_mlir_python_sources(
          ${_stubgen_target}
          ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir/_mlir_libs"
          ADD_TO_PARENT "${sources_target}"
          SOURCES "${_generate_type_stubs}"
        )
        set(_pure_sources_target "${modules_target}.sources.${sources_target}_type_stub_gen")
        add_mlir_python_sources_target(${_pure_sources_target}
          INSTALL_COMPONENT ${modules_target}
          INSTALL_DIR "${ARG_INSTALL_PREFIX}/_mlir_libs"
          OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/_mlir_libs"
          SOURCES_TARGETS ${_stubgen_target}
        )
        add_dependencies(${modules_target} ${_pure_sources_target})
      endif()
    else()
      message(SEND_ERROR "Unrecognized source type '${_source_type}' for python source target ${sources_target}")
      return()
    endif()
  endfunction()

  # Build the modules target.
  add_custom_target(${name} ALL)
  _flatten_mlir_python_targets(_flat_targets ${ARG_DECLARED_SOURCES})
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
#   GEN_ENUM_BINDINGS: Generate enum bindings.
#   GEN_ENUM_BINDINGS_TD_FILE: Optional Tablegen file to generate enums for (relative to ROOT_DIR).
#     This file is where the *EnumAttrs are defined, not where the *Enums are defined.
#     **WARNING**: This arg will shortly be removed when the just-below TODO is satisfied. Use at your
#     risk.
#
# TODO: Right now `TD_FILE` can't be the actual dialect tablegen file, since we
#       use its path to determine where to place the generated python file. If
#       we made the output path an additional argument here we could remove the
#       need for the separate "wrapper" .td files
function(declare_mlir_dialect_python_bindings)
  cmake_parse_arguments(ARG
    "GEN_ENUM_BINDINGS"
    "ROOT_DIR;ADD_TO_PARENT;TD_FILE;DIALECT_NAME"
    "SOURCES;SOURCES_GLOB;DEPENDS;GEN_ENUM_BINDINGS_TD_FILE"
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
    set(tblgen_target "${_dialect_target}.tablegen")
    set(td_file "${ARG_ROOT_DIR}/${ARG_TD_FILE}")
    get_filename_component(relative_td_directory "${ARG_TD_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${relative_td_directory}")
    set(dialect_filename "${relative_td_directory}/_${ARG_DIALECT_NAME}_ops_gen.py")
    set(LLVM_TARGET_DEFINITIONS ${td_file})
    mlir_tablegen("${dialect_filename}"
      -gen-python-op-bindings -bind-dialect=${ARG_DIALECT_NAME}
      DEPENDS ${ARG_DEPENDS}
    )
    add_public_tablegen_target(${tblgen_target})

    set(_sources ${dialect_filename})
    if(ARG_GEN_ENUM_BINDINGS OR ARG_GEN_ENUM_BINDINGS_TD_FILE)
      if(ARG_GEN_ENUM_BINDINGS_TD_FILE)
        set(td_file "${ARG_ROOT_DIR}/${ARG_GEN_ENUM_BINDINGS_TD_FILE}")
        set(LLVM_TARGET_DEFINITIONS ${td_file})
      endif()
      set(enum_filename "${relative_td_directory}/_${ARG_DIALECT_NAME}_enum_gen.py")
      mlir_tablegen(${enum_filename} -gen-python-enum-bindings)
      list(APPEND _sources ${enum_filename})
    endif()

    # Generated.
    declare_mlir_python_sources("${_dialect_target}.ops_gen"
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
      ADD_TO_PARENT "${_dialect_target}"
      SOURCES ${_sources}
    )
  endif()
endfunction()

# Function: declare_mlir_dialect_extension_python_bindings
# Helper to generate source groups for dialect extensions, including both
# static source files and a TD_FILE to generate wrappers.
#
# This will generate a source group named ${ADD_TO_PARENT}.${EXTENSION_NAME}.
#
# Arguments:
#   ROOT_DIR: Same as for declare_mlir_python_sources().
#   ADD_TO_PARENT: Same as for declare_mlir_python_sources(). Unique names
#     for the subordinate source groups are derived from this.
#   TD_FILE: Tablegen file to generate source for (relative to ROOT_DIR).
#   DIALECT_NAME: Python name of the dialect.
#   EXTENSION_NAME: Python name of the dialect extension.
#   SOURCES: Same as declare_mlir_python_sources().
#   SOURCES_GLOB: Same as declare_mlir_python_sources().
#   DEPENDS: Additional dependency targets.
#   GEN_ENUM_BINDINGS: Generate enum bindings.
#   GEN_ENUM_BINDINGS_TD_FILE: Optional Tablegen file to generate enums for (relative to ROOT_DIR).
#     This file is where the *Attrs are defined, not where the *Enums are defined.
#     **WARNING**: This arg will shortly be removed when the TODO for
#     declare_mlir_dialect_python_bindings is satisfied. Use at your risk.
function(declare_mlir_dialect_extension_python_bindings)
  cmake_parse_arguments(ARG
    "GEN_ENUM_BINDINGS"
    "ROOT_DIR;ADD_TO_PARENT;TD_FILE;DIALECT_NAME;EXTENSION_NAME"
    "SOURCES;SOURCES_GLOB;DEPENDS;GEN_ENUM_BINDINGS_TD_FILE"
    ${ARGN})
  # Source files.
  set(_extension_target "${ARG_ADD_TO_PARENT}.${ARG_EXTENSION_NAME}")
  declare_mlir_python_sources(${_extension_target}
    ROOT_DIR "${ARG_ROOT_DIR}"
    ADD_TO_PARENT "${ARG_ADD_TO_PARENT}"
    SOURCES "${ARG_SOURCES}"
    SOURCES_GLOB "${ARG_SOURCES_GLOB}"
  )

  # Tablegen
  if(ARG_TD_FILE)
    set(tblgen_target "${ARG_ADD_TO_PARENT}.${ARG_EXTENSION_NAME}.tablegen")
    set(td_file "${ARG_ROOT_DIR}/${ARG_TD_FILE}")
    get_filename_component(relative_td_directory "${ARG_TD_FILE}" DIRECTORY)
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${relative_td_directory}")
    set(output_filename "${relative_td_directory}/_${ARG_EXTENSION_NAME}_ops_gen.py")
    set(LLVM_TARGET_DEFINITIONS ${td_file})
    mlir_tablegen("${output_filename}" -gen-python-op-bindings
                  -bind-dialect=${ARG_DIALECT_NAME}
                  -dialect-extension=${ARG_EXTENSION_NAME})
    add_public_tablegen_target(${tblgen_target})
    if(ARG_DEPENDS)
      add_dependencies(${tblgen_target} ${ARG_DEPENDS})
    endif()

    set(_sources ${output_filename})
    if(ARG_GEN_ENUM_BINDINGS OR ARG_GEN_ENUM_BINDINGS_TD_FILE)
      if(ARG_GEN_ENUM_BINDINGS_TD_FILE)
        set(td_file "${ARG_ROOT_DIR}/${ARG_GEN_ENUM_BINDINGS_TD_FILE}")
        set(LLVM_TARGET_DEFINITIONS ${td_file})
      endif()
      set(enum_filename "${relative_td_directory}/_${ARG_EXTENSION_NAME}_enum_gen.py")
      mlir_tablegen(${enum_filename} -gen-python-enum-bindings)
      list(APPEND _sources ${enum_filename})
    endif()

    declare_mlir_python_sources("${_extension_target}.ops_gen"
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}"
      ADD_TO_PARENT "${_extension_target}"
      SOURCES ${_sources}
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
#   DECLARED_HEADERS: Source groups from which to discover headers that belong
#     to the library and should be installed with it.
#   DECLARED_SOURCES: Source groups from which to discover dependent
#     EMBED_CAPI_LINK_LIBS.
#   EMBED_LIBS: Additional libraries to embed (must be built with OBJECTS and
#     have an "obj.${name}" object library associated).
function(add_mlir_python_common_capi_library name)
  cmake_parse_arguments(ARG
    ""
    "INSTALL_COMPONENT;INSTALL_DESTINATION;OUTPUT_DIRECTORY;RELATIVE_INSTALL_ROOT"
    "DECLARED_HEADERS;DECLARED_SOURCES;EMBED_LIBS"
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

  # Process any headers associated with the library
  _flatten_mlir_python_targets(_flat_header_targets ${ARG_DECLARED_HEADERS})
  set(_header_sources_target "${name}.sources")
  add_mlir_python_sources_target(${_header_sources_target}
    INSTALL_COMPONENT ${ARG_INSTALL_COMPONENT}
    INSTALL_DIR "${ARG_INSTALL_DESTINATION}/include"
    OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}/include"
    SOURCES_TARGETS ${_flat_header_targets}
  )
  add_dependencies(${name} ${_header_sources_target})

  if(WIN32)
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

# Function: add_mlir_python_sources_target
# Adds a target corresponding to an interface target that carries source
# information. This target is responsible for "building" the sources by
# placing them in the correct locations in the build and install trees.
# Arguments:
#   INSTALL_COMPONENT: Name of the install component. Typically same as the
#     target name passed to add_mlir_python_modules().
#   INSTALL_DESTINATION: Prefix into the install tree in which to install the
#     library.
#   OUTPUT_DIRECTORY: Full path in the build tree in which to create the
#     library. Typically, this will be the common _mlir_libs directory where
#     all extensions are emitted.
#   SOURCES_TARGETS: List of interface libraries that carry source information.
function(add_mlir_python_sources_target name)
  cmake_parse_arguments(ARG
  ""
  "INSTALL_COMPONENT;INSTALL_DIR;OUTPUT_DIRECTORY"
  "SOURCES_TARGETS"
  ${ARGN})

  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unhandled arguments to add_mlir_python_sources_target(${name}, ... : ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # On Windows create_symlink requires special permissions. Use copy_if_different instead.
  if(CMAKE_HOST_WIN32)
    set(_link_or_copy copy_if_different)
  else()
    set(_link_or_copy create_symlink)
  endif()

  foreach(_sources_target ${ARG_SOURCES_TARGETS})
    get_target_property(_src_paths ${_sources_target} SOURCES)
    if(NOT _src_paths)
      get_target_property(_src_paths ${_sources_target} INTERFACE_SOURCES)
      if(NOT _src_paths)
        break()
      endif()
    endif()

    get_target_property(_root_dir ${_sources_target} INCLUDE_DIRECTORIES)
    if(NOT _root_dir)
      get_target_property(_root_dir ${_sources_target} INTERFACE_INCLUDE_DIRECTORIES)
    endif()

    # Initialize an empty list of all Python source destination paths.
    set(all_dest_paths "")
    foreach(_src_path ${_src_paths})
      file(RELATIVE_PATH _source_relative_path "${_root_dir}" "${_src_path}")
      set(_dest_path "${ARG_OUTPUT_DIRECTORY}/${_source_relative_path}")

      get_filename_component(_dest_dir "${_dest_path}" DIRECTORY)
      file(MAKE_DIRECTORY "${_dest_dir}")

      add_custom_command(
        OUTPUT "${_dest_path}"
        COMMENT "Copying python source ${_src_path} -> ${_dest_path}"
        DEPENDS "${_src_path}"
        COMMAND "${CMAKE_COMMAND}" -E ${_link_or_copy}
            "${_src_path}" "${_dest_path}"
      )

      # Track the symlink or copy command output.
      list(APPEND all_dest_paths "${_dest_path}")

      if(ARG_INSTALL_DIR)
        # We have to install each file individually because we need to preserve
        # the relative directory structure in the install destination.
        # As an example, ${_source_relative_path} may be dialects/math.py
        # which would be transformed to ${ARG_INSTALL_DIR}/dialects
        # here. This could be moved outside of the loop and cleaned up
        # if we used FILE_SETS (introduced in CMake 3.23).
        get_filename_component(_install_destination "${ARG_INSTALL_DIR}/${_source_relative_path}" DIRECTORY)
        install(
          FILES ${_src_path}
          DESTINATION "${_install_destination}"
          COMPONENT ${ARG_INSTALL_COMPONENT}
        )
      endif()
    endforeach()
  endforeach()

  # Create a new custom target that depends on all symlinked or copied sources.
  add_custom_target("${name}" DEPENDS ${all_dest_paths})
endfunction()

################################################################################
# Build python extension
################################################################################
function(add_mlir_python_extension libname extname)
  cmake_parse_arguments(ARG
  ""
  "INSTALL_COMPONENT;INSTALL_DIR;OUTPUT_DIRECTORY;PYTHON_BINDINGS_LIBRARY"
  "SOURCES;LINK_LIBS"
  ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unhandled arguments to add_mlir_python_extension(${libname}, ... : ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  # The extension itself must be compiled with RTTI and exceptions enabled.
  # Also, some warning classes triggered by pybind11 are disabled.
  set(eh_rtti_enable)
  if (MSVC)
    set(eh_rtti_enable /EHsc /GR)
  elseif(LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL)
    set(eh_rtti_enable -frtti -fexceptions)
  endif ()

  # The actual extension library produces a shared-object or DLL and has
  # sources that must be compiled in accordance with pybind11 needs (RTTI and
  # exceptions).
  if(NOT DEFINED ARG_PYTHON_BINDINGS_LIBRARY OR ARG_PYTHON_BINDINGS_LIBRARY STREQUAL "pybind11")
    pybind11_add_module(${libname}
      ${ARG_SOURCES}
    )
  elseif(ARG_PYTHON_BINDINGS_LIBRARY STREQUAL "nanobind")
    nanobind_add_module(${libname}
      NB_DOMAIN ${MLIR_BINDINGS_PYTHON_NB_DOMAIN}
      FREE_THREADED
      ${ARG_SOURCES}
    )

    if (NOT MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES
        AND (LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL))
      # Avoid some warnings from upstream nanobind.
      # If a superproject set MLIR_DISABLE_CONFIGURE_PYTHON_DEV_PACKAGES, let
      # the super project handle compile options as it wishes.
      get_property(NB_LIBRARY_TARGET_NAME TARGET ${libname} PROPERTY LINK_LIBRARIES)
      target_compile_options(${NB_LIBRARY_TARGET_NAME}
        PRIVATE
          -Wall -Wextra -Wpedantic
          -Wno-c++98-compat-extra-semi
          -Wno-cast-qual
          -Wno-covered-switch-default
          -Wno-deprecated-literal-operator
          -Wno-nested-anon-types
          -Wno-unused-parameter
          -Wno-zero-length-array
          ${eh_rtti_enable})

      target_compile_options(${libname}
        PRIVATE
          -Wall -Wextra -Wpedantic
          -Wno-c++98-compat-extra-semi
          -Wno-cast-qual
          -Wno-covered-switch-default
          -Wno-deprecated-literal-operator
          -Wno-nested-anon-types
          -Wno-unused-parameter
          -Wno-zero-length-array
          ${eh_rtti_enable})
    endif()

    if(APPLE)
      # NanobindAdaptors.h uses PyClassMethod_New to build `pure_subclass`es but nanobind
      # doesn't declare this API as undefined in its linker flags. So we need to declare it as such
      # for downstream users that do not do something like `-undefined dynamic_lookup`.
      # Same for the rest.
      target_link_options(${libname} PUBLIC
        "LINKER:-U,_PyClassMethod_New"
        "LINKER:-U,_PyCode_Addr2Location"
        "LINKER:-U,_PyFrame_GetLasti"
      )
    endif()
  endif()

  target_compile_options(${libname} PRIVATE ${eh_rtti_enable})

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

  if(WIN32)
    # On Windows, pyconfig.h (and by extension python.h) hardcode the version of the
    # python library which will be used for linkage depending on the flavor of the build.
    # pybind11 has a workaround which depends on the definition of Py_DEBUG (if Py_DEBUG
    # is not passed in as a compile definition, pybind11 undefs _DEBUG when including
    # python.h, so that the release python library would be used).
    # Since mlir uses pybind11, we can leverage their workaround by never directly
    # pyconfig.h or python.h and instead relying on the pybind11 headers to include the
    # necessary python headers. This results in mlir always linking against the
    # release python library via the (undocumented) cmake property Python3_LIBRARY_RELEASE.
    target_link_libraries(${libname} PRIVATE ${Python3_LIBRARY_RELEASE})
  endif()

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
