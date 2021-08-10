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
  add_custom_target(${name})
  set(_file_depends "${ARG_SOURCES}")
  list(TRANSFORM _file_depends PREPEND "${ARG_ROOT_DIR}/")
  set_target_properties(${name} PROPERTIES
    PYTHON_SOURCES_TYPE pure
    PYTHON_ROOT_DIR "${ARG_ROOT_DIR}"
    PYTHON_DEST_PREFIX "${ARG_DEST_PREFIX}"
    PYTHON_SOURCES "${ARG_SOURCES}"
    PYTHON_FILE_DEPENDS "${_file_depends}"
    PYTHON_DEPENDS ""
  )

  # Add to parent.
  if(ARG_ADD_TO_PARENT)
    set_property(TARGET ${ARG_ADD_TO_PARENT} APPEND PROPERTY PYTHON_DEPENDS ${name})
  endif()
endfunction()

# Function: declare_mlir_python_extension
# Declares a buildable python extension from C++ source files. The built
# module is considered a python source file and included as everything else.
# Arguments:
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
    "MODULE_NAME;ADD_TO_PARENT"
    "SOURCES;PRIVATE_LINK_LIBS;EMBED_CAPI_LINK_LIBS"
    ${ARGN})

  add_custom_target(${name})
  set_target_properties(${name} PROPERTIES
    PYTHON_SOURCES_TYPE extension
    PYTHON_EXTENSION_MODULE_NAME "${ARG_MODULE_NAME}"
    PYTHON_CPP_SOURCES "${ARG_SOURCES}"
    PYTHON_PRIVATE_LINK_LIBS "${ARG_PRIVATE_LINK_LIBS}"
    PYTHON_EMBED_CAPI_LINK_LIBS "${ARG_EMBED_CAPI_LINK_LIBS}"
    PYTHON_FILE_DEPENDS ""
    PYTHON_DEPENDS ""
  )

  # Add to parent.
  if(ARG_ADD_TO_PARENT)
    set_property(TARGET ${ARG_ADD_TO_PARENT} APPEND PROPERTY PYTHON_DEPENDS ${name})
  endif()
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
    get_target_property(_source_type ${sources_target} PYTHON_SOURCES_TYPE)
    if(_source_type STREQUAL "pure")
      # Pure python sources to link into the tree.
      get_target_property(_python_root_dir ${sources_target} PYTHON_ROOT_DIR)
      get_target_property(_python_sources ${sources_target} PYTHON_SOURCES)
      get_target_property(_specified_dest_prefix ${sources_target} PYTHON_DEST_PREFIX)
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
        add_custom_command(
          TARGET ${modules_target} PRE_BUILD
          COMMENT "Copying python source ${_src_path} -> ${_dest_path}"
          DEPENDS "${_src_path}"
          BYPRODUCTS "${_dest_path}"
          COMMAND "${CMAKE_COMMAND}" -E create_symlink
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
      get_target_property(_module_name ${sources_target} PYTHON_EXTENSION_MODULE_NAME)
      get_target_property(_cpp_sources ${sources_target} PYTHON_CPP_SOURCES)
      get_target_property(_private_link_libs ${sources_target} PYTHON_PRIVATE_LINK_LIBS)
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
    get_target_property(_local_depends ${sources_target} PYTHON_FILE_DEPENDS)
    list(APPEND _depends ${_local_depends})
  endforeach()

  # Build the modules target.
  add_custom_target(${name} ALL DEPENDS ${_depends})
  foreach(sources_target ${_flat_targets})
    _process_target(${name} ${sources_target})
  endforeach()

  # Create an install target.
  if (NOT LLVM_ENABLE_IDE)
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
  # TODO: Upgrade to the aggregate utility in https://reviews.llvm.org/D106419
  # once ready.

  # Collect all explicit and transitive embed libs.
  set(_embed_libs ${ARG_EMBED_LIBS})
  _flatten_mlir_python_targets(_all_source_targets ${ARG_DECLARED_SOURCES})
  foreach(t ${_all_source_targets})
    get_target_property(_local_embed_libs ${t} PYTHON_EMBED_CAPI_LINK_LIBS)
    if(_local_embed_libs)
      list(APPEND _embed_libs ${_local_embed_libs})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _embed_libs)

  foreach(lib ${_embed_libs})
    if(XCODE)
      # Xcode doesn't support object libraries, so we have to trick it into
      # linking the static libraries instead.
      list(APPEND _deps "-force_load" ${lib})
    else()
      list(APPEND _objects $<TARGET_OBJECTS:obj.${lib}>)
    endif()
    # Accumulate transitive deps of each exported lib into _DEPS.
    list(APPEND _deps $<TARGET_PROPERTY:${lib},LINK_LIBRARIES>)
  endforeach()

  add_mlir_library(${name}
    PARTIAL_SOURCES_INTENDED
    SHARED
    DISABLE_INSTALL
    ${_objects}
    EXCLUDE_FROM_LIBMLIR
    LINK_LIBS
    ${_deps}
  )
  if(MSVC)
    set_property(TARGET ${name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
    BINARY_OUTPUT_DIRECTORY "${ARG_OUTPUT_DIRECTORY}"
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
    get_target_property(_source_type ${t} PYTHON_SOURCES_TYPE)
    get_target_property(_depends ${t} PYTHON_DEPENDS)
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
  if (ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR " Unhandled arguments to add_mlir_python_extension(${libname}, ... : ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if ("${ARG_SOURCES}" STREQUAL "")
    message(FATAL_ERROR " Missing SOURCES argument to add_mlir_python_extension(${libname}, ...")
  endif()

  # Build-time RPath layouts require to be a directory one up from the
  # binary root.
  # TODO: Don't reference the LLVM_BINARY_DIR here: the invariant is that
  # the output directory must be at the same level of the lib directory
  # where libMLIR.so is installed. This is presently not optimal from a
  # project separation perspective and a discussion on how to better
  # segment MLIR libraries needs to happen.
  # TODO: Remove this when downstreams are moved off of it.
  if(NOT ARG_OUTPUT_DIRECTORY)
    set(ARG_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/python)
  endif()

  # Normally on unix-like platforms, extensions are built as "MODULE" libraries
  # and do not explicitly link to the python shared object. This allows for
  # some greater deployment flexibility since the extension will bind to
  # symbols in the python interpreter on load. However, it also keeps the
  # linker from erroring on undefined symbols, leaving this to (usually obtuse)
  # runtime errors. Building in "SHARED" mode with an explicit link to the
  # python libraries allows us to build with the expectation of no undefined
  # symbols, which is better for development. Note that not all python
  # configurations provide build-time libraries to link against, in which
  # case, we fall back to MODULE linking.
  if(Python3_LIBRARIES STREQUAL "" OR NOT MLIR_BINDINGS_PYTHON_LOCK_VERSION)
    set(PYEXT_LINK_MODE MODULE)
    set(PYEXT_LIBADD)
  else()
    set(PYEXT_LINK_MODE SHARED)
    set(PYEXT_LIBADD ${Python3_LIBRARIES})
  endif()

  # The actual extension library produces a shared-object or DLL and has
  # sources that must be compiled in accordance with pybind11 needs (RTTI and
  # exceptions).
  add_library(${libname}
    ${PYEXT_LINK_MODE}
    ${ARG_SOURCES}
  )

  target_include_directories(${libname} PRIVATE
    "${Python3_INCLUDE_DIRS}"
    "${pybind11_INCLUDE_DIR}"
  )

  target_link_directories(${libname} PRIVATE
    "${Python3_LIBRARY_DIRS}"
  )

  # The extension itself must be compiled with RTTI and exceptions enabled.
  # Also, some warning classes triggered by pybind11 are disabled.
  target_compile_options(${libname} PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
      # Enable RTTI and exceptions.
      -frtti -fexceptions
      # Noisy pybind warnings
      -Wno-unused-value
      -Wno-covered-switch-default
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
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}${PYTHON_MODULE_EXTENSION}"
    NO_SONAME ON
  )

  if(WIN32)
    # Need to also set the RUNTIME_OUTPUT_DIRECTORY on Windows in order to
    # control where the .dll gets written.
    set_target_properties(
      ${libname} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
    )
  endif()

  # pybind11 requires binding code to be compiled with -fvisibility=hidden
  # For static linkage, better code can be generated if the entire project
  # compiles that way, but that is not enforced here. Instead, include a linker
  # script that explicitly hides anything but the PyInit_* symbols, allowing gc
  # to take place.
  set_target_properties(${libname} PROPERTIES CXX_VISIBILITY_PRESET "hidden")

  # Python extensions depends *only* on the public API and LLVMSupport unless
  # if further dependencies are added explicitly.
  target_link_libraries(${libname}
    PRIVATE
    ${ARG_LINK_LIBS}
    ${PYEXT_LIBADD}
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
  if (ARG_INSTALL_DIR)
    install(TARGETS ${libname}
      COMPONENT ${ARG_INSTALL_COMPONENT}
      LIBRARY DESTINATION ${ARG_INSTALL_DIR}
      ARCHIVE DESTINATION ${ARG_INSTALL_DIR}
      # NOTE: Even on DLL-platforms, extensions go in the lib directory tree.
      RUNTIME DESTINATION ${ARG_INSTALL_DIR}
    )
  endif()
endfunction()
