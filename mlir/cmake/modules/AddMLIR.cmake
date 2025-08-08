include(TableGen)
include(GNUInstallDirs)
include(LLVMDistributionSupport)

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this td file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  # Append the includes used for this file to the tablegen_compile_commands
  # file.
  file(APPEND ${CMAKE_BINARY_DIR}/tablegen_compile_commands.yml
      "--- !FileInfo:\n"
      "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
      "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
  )
endfunction()

# Clear out any pre-existing compile_commands file before processing. This
# allows for generating a clean compile_commands on each configure.
file(REMOVE ${CMAKE_BINARY_DIR}/pdll_compile_commands.yml)

# Declare a helper function/copy of tablegen rule for using tablegen without
# additional tblgen specific flags when invoking PDLL generator.
function(_pdll_tablegen project ofn)
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  # Validate calling context.
  if(NOT ${project}_TABLEGEN_EXE)
    message(FATAL_ERROR "${project}_TABLEGEN_EXE not set")
  endif()

  # Use depfile instead of globbing arbitrary *.td(s) for Ninja. We force
  # CMake versions older than v3.30 on Windows to use the fallback behavior
  # due to a depfile parsing bug on Windows paths in versions prior to 3.30.
  # https://gitlab.kitware.com/cmake/cmake/-/issues/25943
  # CMake versions older than v3.23 on other platforms use the fallback
  # behavior as v3.22 and earlier fail to parse some depfiles that get
  # generated, and this behavior was fixed in CMake commit
  # e04a352cca523eba2ac0d60063a3799f5bb1c69e.
  cmake_policy(GET CMP0116 cmp0116_state)
  if(CMAKE_GENERATOR MATCHES "Ninja" AND cmp0116_state STREQUAL NEW
     AND NOT (CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_LESS 3.30)
     AND NOT (CMAKE_VERSION VERSION_LESS 3.23))
    # CMake emits build targets as relative paths but Ninja doesn't identify
    # absolute path (in *.d) as relative path (in build.ninja). Post CMP0116,
    # CMake handles this discrepancy for us. Otherwise, we use the fallback
    # logic.
    set(additional_cmdline
      -o ${ofn}
      -d ${ofn}.d
      DEPFILE ${ofn}.d
      )
    set(local_tds)
    set(global_tds)
  else()
    file(GLOB local_tds "*.td")
    file(GLOB_RECURSE global_tds "${LLVM_MAIN_INCLUDE_DIR}/llvm/*.td")
    set(additional_cmdline
      -o ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      )
  endif()

  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE
      ${CMAKE_CURRENT_SOURCE_DIR}/${LLVM_TARGET_DEFINITIONS})
  endif()

  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    # Visual Studio has problems with llvm-tblgen's native --write-if-changed
    # behavior. Since it doesn't do restat optimizations anyway, just don't
    # pass --write-if-changed there.
    set(tblgen_change_flag)
  else()
    set(tblgen_change_flag "--write-if-changed")
  endif()

  # We need both _TABLEGEN_TARGET and _TABLEGEN_EXE in the  DEPENDS list
  # (both the target and the file) to have .inc files rebuilt on
  # a tablegen change, as cmake does not propagate file-level dependencies
  # of custom targets. See the following ticket for more information:
  # https://cmake.org/Bug/view.php?id=15858
  # The dependency on both, the target and the file, produces the same
  # dependency twice in the result file when
  # ("${${project}_TABLEGEN_TARGET}" STREQUAL "${${project}_TABLEGEN_EXE}")
  # but lets us having smaller and cleaner code here.
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out empty items before prepending each entry with -I
  list(REMOVE_ITEM tblgen_includes "")
  list(TRANSFORM tblgen_includes PREPEND -I)

  set(tablegen_exe ${${project}_TABLEGEN_EXE})
  set(tablegen_depends ${${project}_TABLEGEN_TARGET} ${tablegen_exe})

  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND ${tablegen_exe} ${ARG_UNPARSED_ARGUMENTS} -I ${CMAKE_CURRENT_SOURCE_DIR}
    ${tblgen_includes}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${tblgen_change_flag}
    ${additional_cmdline}
    # The file in LLVM_TARGET_DEFINITIONS may be not in the current
    # directory and local_tds may not contain it, so we must
    # explicitly list it here:
    DEPENDS ${ARG_DEPENDS} ${tablegen_depends}
      ${local_tds} ${global_tds}
    ${LLVM_TARGET_DEFINITIONS_ABSOLUTE}
    ${LLVM_TARGET_DEPENDS}
    COMMENT "Building ${ofn}..."
    )

  # `make clean' must remove all those generated files:
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES ${ofn})

  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn} PARENT_SCOPE)
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/${ofn} PROPERTIES
    GENERATED 1)
endfunction()

# Declare a PDLL library in the current directory.
function(add_mlir_pdll_library target inputFile ofn)
  set(LLVM_TARGET_DEFINITIONS ${inputFile})

  _pdll_tablegen(MLIR_PDLL ${ofn} -x=cpp ${ARGN})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)

  # Get the current set of include paths for this pdll file.
  cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
  get_directory_property(tblgen_includes INCLUDE_DIRECTORIES)
  list(APPEND tblgen_includes ${ARG_EXTRA_INCLUDES})
  # Filter out any empty include items.
  list(REMOVE_ITEM tblgen_includes "")

  # Build the absolute path for the current input file.
  if (IS_ABSOLUTE ${LLVM_TARGET_DEFINITIONS})
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${inputFile})
  else()
    set(LLVM_TARGET_DEFINITIONS_ABSOLUTE ${CMAKE_CURRENT_SOURCE_DIR}/${inputFile})
  endif()

  # Append the includes used for this file to the pdll_compilation_commands
  # file.
  file(APPEND ${CMAKE_BINARY_DIR}/pdll_compile_commands.yml
      "--- !FileInfo:\n"
      "  filepath: \"${LLVM_TARGET_DEFINITIONS_ABSOLUTE}\"\n"
      "  includes: \"${CMAKE_CURRENT_SOURCE_DIR};${tblgen_includes}\"\n"
  )

  add_public_tablegen_target(${target})
endfunction()

# Declare a dialect in the include directory
function(add_mlir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs -typedefs-dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
  mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
  add_public_tablegen_target(MLIR${dialect}IncGen)
  add_dependencies(mlir-headers MLIR${dialect}IncGen)
endfunction()

# Declare sharded dialect operation declarations and definitions
function(add_sharded_ops ops_target shard_count)
  set(LLVM_TARGET_DEFINITIONS ${ops_target}.td)
  mlir_tablegen(${ops_target}.h.inc -gen-op-decls -op-shard-count=${shard_count})
  mlir_tablegen(${ops_target}.cpp.inc -gen-op-defs -op-shard-count=${shard_count})
  set(LLVM_TARGET_DEFINITIONS ${ops_target}.cpp)
  foreach(index RANGE ${shard_count})
    set(SHARDED_SRC ${ops_target}.${index}.cpp)
    list(APPEND SHARDED_SRCS ${SHARDED_SRC})
    tablegen(MLIR_SRC_SHARDER ${SHARDED_SRC} -op-shard-index=${index})
    set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${SHARDED_SRC})
  endforeach()
  add_public_tablegen_target(MLIR${ops_target}ShardGen)
  set(SHARDED_SRCS ${SHARDED_SRCS} PARENT_SCOPE)
endfunction()

# Declare a dialect in the include directory
function(add_mlir_interface interface)
  set(LLVM_TARGET_DEFINITIONS ${interface}.td)
  mlir_tablegen(${interface}.h.inc -gen-op-interface-decls)
  mlir_tablegen(${interface}.cpp.inc -gen-op-interface-defs)
  add_public_tablegen_target(MLIR${interface}IncGen)
  add_dependencies(mlir-generic-headers MLIR${interface}IncGen)
endfunction()


# Generate Documentation
function(add_mlir_doc doc_filename output_file output_directory command)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  # The MLIR docs use Hugo, so we allow Hugo specific features here.
  tablegen(MLIR ${output_file}.md ${command} -allow-hugo-specific-features ${ARGN})
  set(GEN_DOC_FILE ${MLIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  set_target_properties(${output_file}DocGen PROPERTIES FOLDER "MLIR/Tablegenning/Docs")
  add_dependencies(mlir-doc ${output_file}DocGen)
endfunction()

# Sets ${srcs} to contain the list of additional headers for the target. Extra
# arguments are included into the list of additional headers.
function(_set_mlir_additional_headers_as_srcs)
  set(srcs)
  if(MSVC_IDE OR XCODE)
    # Add public headers
    file(RELATIVE_PATH lib_path
      ${MLIR_SOURCE_DIR}/lib/
      ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT lib_path MATCHES "^[.][.]")
      file( GLOB_RECURSE headers
        ${MLIR_SOURCE_DIR}/include/mlir/${lib_path}/*.h
        ${MLIR_SOURCE_DIR}/include/mlir/${lib_path}/*.def
      )
      set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY ON)

      file( GLOB_RECURSE tds
        ${MLIR_SOURCE_DIR}/include/mlir/${lib_path}/*.td
      )
      source_group("TableGen descriptions" FILES ${tds})
      set_source_files_properties(${tds}} PROPERTIES HEADER_FILE_ONLY ON)

      if(headers OR tds)
        set(srcs ${headers} ${tds})
      endif()
    endif()
  endif(MSVC_IDE OR XCODE)
  if(srcs OR ARGN)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARGN} # It may contain unparsed unknown args.
      PARENT_SCOPE
      )
  endif()
endfunction()

# Checks that the LLVM components are not listed in the extra arguments,
# assumed to be coming from the LINK_LIBS variable.
function(_check_llvm_components_usage name)
  # LINK_COMPONENTS is necessary to allow libLLVM.so to be properly
  # substituted for individual library dependencies if LLVM_LINK_LLVM_DYLIB
  # Perhaps this should be in llvm_add_library instead?  However, it fails
  # on libclang-cpp.so
  get_property(llvm_component_libs GLOBAL PROPERTY LLVM_COMPONENT_LIBS)
  foreach(lib ${ARGN})
    if(${lib} IN_LIST llvm_component_libs)
      message(SEND_ERROR "${name} specifies LINK_LIBS ${lib}, but LINK_LIBS cannot be used for LLVM libraries.  Please use LINK_COMPONENTS instead.")
    endif()
  endforeach()
endfunction()

function(add_mlir_example_library name)
  cmake_parse_arguments(ARG
    "SHARED;DISABLE_INSTALL"
    ""
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
    ${ARGN})
  _set_mlir_additional_headers_as_srcs(${ARG_ADDITIONAL_HEADERS})
  if (ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    if(BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED)
    else()
      set(LIBTYPE STATIC)
    endif()
  endif()

  # MLIR libraries uniformly depend on LLVMSupport.  Just specify it once here.
  list(APPEND ARG_LINK_COMPONENTS Support)
  _check_llvm_components_usage(${name} ${ARG_LINK_LIBS})

  list(APPEND ARG_DEPENDS mlir-generic-headers)

  llvm_add_library(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs} DEPENDS ${ARG_DEPENDS} LINK_COMPONENTS ${ARG_LINK_COMPONENTS} LINK_LIBS ${ARG_LINK_LIBS})
  set_target_properties(${name} PROPERTIES FOLDER "MLIR/Examples")
  if (LLVM_BUILD_EXAMPLES AND NOT ${ARG_DISABLE_INSTALL})
    add_mlir_library_install(${name})
  else()
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL ON)
  endif()
endfunction()

# Declare an mlir library which can be compiled in libMLIR.so
# In addition to everything that llvm_add_library accepts, this
# also has the following option:
# EXCLUDE_FROM_LIBMLIR
#   Don't include this library in libMLIR.so.  This option should be used
#   for test libraries, executable-specific libraries, or rarely used libraries
#   with large dependencies.  When using it, please link libraries included
#   in libMLIR via mlir_target_link_libraries(), to ensure that the library
#   does not pull in static dependencies when MLIR_LINK_MLIR_DYLIB=ON is used.
# OBJECT
#   The library's object library is referenced using "obj.${name}". For this to
#   work reliably, this flag ensures that the OBJECT library exists.
# ENABLE_AGGREGATION
#   Exports additional metadata,
#   and installs additional object files needed to include this as part of an
#   aggregate shared library.
#   TODO: Make this the default for all MLIR libraries once all libraries
#   are compatible with building an object library.
function(add_mlir_library name)
  cmake_parse_arguments(ARG
    "SHARED;INSTALL_WITH_TOOLCHAIN;EXCLUDE_FROM_LIBMLIR;DISABLE_INSTALL;ENABLE_AGGREGATION;OBJECT"
    ""
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
    ${ARGN})
  _set_mlir_additional_headers_as_srcs(${ARG_ADDITIONAL_HEADERS})

  # Determine type of library.
  if(ARG_SHARED)
    set(LIBTYPE SHARED)
  else()
    # llvm_add_library ignores BUILD_SHARED_LIBS if STATIC is explicitly set,
    # so we need to handle it here.
    if(BUILD_SHARED_LIBS)
      set(LIBTYPE SHARED)
    else()
      set(LIBTYPE STATIC)
    endif()
  endif()

  # Is an object library needed...?
  # Note that the XCode generator doesn't handle object libraries correctly and
  # usability is degraded in the Visual Studio solution generators.
  # llvm_add_library may also itself decide to create an object library.
  set(NEEDS_OBJECT_LIB OFF)
  if(ARG_OBJECT)
    # Yes, because the target "obj.${name}" is referenced.
    set(NEEDS_OBJECT_LIB ON)
  endif ()
  if(LLVM_BUILD_LLVM_DYLIB AND NOT ARG_EXCLUDE_FROM_LIBMLIR AND NOT XCODE)
    # Yes, because in addition to the shared library, the object files are
    # needed for linking into libMLIR.so (see mlir/tools/mlir-shlib/CMakeLists.txt).
    # For XCode, -force_load is used instead.
    # Windows is not supported (LLVM_BUILD_LLVM_DYLIB=ON will cause an error).
    set(NEEDS_OBJECT_LIB ON)
    set_property(GLOBAL APPEND PROPERTY MLIR_STATIC_LIBS ${name})
    set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS ${ARG_LINK_COMPONENTS})
    set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
  endif ()
  if(ARG_ENABLE_AGGREGATION AND NOT XCODE)
    # Yes, because this library is added to an aggergate library such as
    # libMLIR-C.so which is links together all the object files.
    # For XCode, -force_load is used instead.
    set(NEEDS_OBJECT_LIB ON)
  endif()
  if (NOT ARG_SHARED AND NOT ARG_EXCLUDE_FROM_LIBMLIR AND NOT XCODE AND NOT MSVC_IDE)
    # Yes, but only for legacy reasons. Also avoid object libraries for
    # Visual Studio solutions.
    set(NEEDS_OBJECT_LIB ON)
  endif()
  if(NEEDS_OBJECT_LIB)
    list(APPEND LIBTYPE OBJECT)
  endif()

  # MLIR libraries uniformly depend on LLVMSupport.  Just specify it once here.
  list(APPEND ARG_LINK_COMPONENTS Support)
  _check_llvm_components_usage(${name} ${ARG_LINK_LIBS})

  list(APPEND ARG_DEPENDS mlir-generic-headers)
  llvm_add_library(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs} DEPENDS ${ARG_DEPENDS} LINK_COMPONENTS ${ARG_LINK_COMPONENTS} LINK_LIBS ${ARG_LINK_LIBS})

  if(TARGET ${name})
    target_link_libraries(${name} INTERFACE ${LLVM_COMMON_LIBS})
    if(ARG_INSTALL_WITH_TOOLCHAIN)
      set_target_properties(${name} PROPERTIES MLIR_INSTALL_WITH_TOOLCHAIN TRUE)
    endif()
    if(NOT ARG_DISABLE_INSTALL)
      add_mlir_library_install(${name})
    endif()
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "MLIR/Libraries")

  # Setup aggregate.
  if(ARG_ENABLE_AGGREGATION)
    # Compute and store the properties needed to build aggregates.
    set(AGGREGATE_OBJECTS)
    set(AGGREGATE_OBJECT_LIB)
    set(AGGREGATE_DEPS)
    if(XCODE)
      # XCode has limited support for object libraries. Instead, add dep flags
      # that force the entire library to be embedded.
      list(APPEND AGGREGATE_DEPS "-force_load" "${name}")
    elseif(TARGET obj.${name})
      # FIXME: *.obj can also be added via target_link_libraries since CMake 3.12.
      list(APPEND AGGREGATE_OBJECTS "$<TARGET_OBJECTS:obj.${name}>")
      list(APPEND AGGREGATE_OBJECT_LIB "obj.${name}")
    else()
      message(SEND_ERROR "Aggregate library not supported on this platform")
    endif()

    # For each declared dependency, transform it into a generator expression
    # which excludes it if the ultimate link target is excluding the library.
    set(NEW_LINK_LIBRARIES)
    get_target_property(CURRENT_LINK_LIBRARIES  ${name} LINK_LIBRARIES)
    get_mlir_filtered_link_libraries(NEW_LINK_LIBRARIES ${CURRENT_LINK_LIBRARIES})
    set_target_properties(${name} PROPERTIES LINK_LIBRARIES "${NEW_LINK_LIBRARIES}")
    list(APPEND AGGREGATE_DEPS ${NEW_LINK_LIBRARIES})
    set_target_properties(${name} PROPERTIES
      EXPORT_PROPERTIES "MLIR_AGGREGATE_OBJECT_LIB_IMPORTED;MLIR_AGGREGATE_DEP_LIBS_IMPORTED"
      MLIR_AGGREGATE_OBJECTS "${AGGREGATE_OBJECTS}"
      MLIR_AGGREGATE_DEPS "${AGGREGATE_DEPS}"
      MLIR_AGGREGATE_OBJECT_LIB_IMPORTED "${AGGREGATE_OBJECT_LIB}"
      MLIR_AGGREGATE_DEP_LIBS_IMPORTED "${CURRENT_LINK_LIBRARIES}"
    )

    # In order for out-of-tree projects to build aggregates of this library,
    # we need to install the OBJECT library.
    if(TARGET "obj.${name}" AND MLIR_INSTALL_AGGREGATE_OBJECTS AND NOT ARG_DISABLE_INSTALL)
      add_mlir_library_install(obj.${name})
    endif()
  endif()
endfunction(add_mlir_library)

macro(add_mlir_tool name)
  llvm_add_tool(MLIR ${ARGV})
endmacro()

# Sets a variable with a transformed list of link libraries such individual
# libraries will be dynamically excluded when evaluated on a final library
# which defines an MLIR_AGGREGATE_EXCLUDE_LIBS which contains any of the
# libraries. Each link library can be a generator expression but must not
# resolve to an arity > 1 (i.e. it can be optional).
function(get_mlir_filtered_link_libraries output)
  set(_results)
  foreach(linklib ${ARGN})
    # In English, what this expression does:
    # For each link library, resolve the property MLIR_AGGREGATE_EXCLUDE_LIBS
    # on the context target (i.e. the executable or shared library being linked)
    # and, if it is not in that list, emit the library name. Otherwise, empty.
    list(APPEND _results
      "$<$<NOT:$<IN_LIST:${linklib},$<GENEX_EVAL:$<TARGET_PROPERTY:MLIR_AGGREGATE_EXCLUDE_LIBS>>>>:${linklib}>"
    )
  endforeach()
  set(${output} "${_results}" PARENT_SCOPE)
endfunction(get_mlir_filtered_link_libraries)

# Declares an aggregate library. Such a library is a combination of arbitrary
# regular add_mlir_library() libraries with the special feature that they can
# be configured to statically embed some subset of their dependencies, as is
# typical when creating a .so/.dylib/.dll or a mondo static library.
#
# It is always safe to depend on the aggregate directly in order to compile/link
# against the superset of embedded entities and transitive deps.
#
# Arguments:
#   PUBLIC_LIBS: list of dependent libraries to add to the
#     INTERFACE_LINK_LIBRARIES property, exporting them to users. This list
#     will be transitively filtered to exclude any EMBED_LIBS.
#   EMBED_LIBS: list of dependent libraries that should be embedded directly
#     into this library. Each of these must be an add_mlir_library() library
#     without DISABLE_AGGREGATE.
#
# Note: This is a work in progress and is presently only sufficient for certain
# non nested cases involving the C-API.
function(add_mlir_aggregate name)
  cmake_parse_arguments(ARG
    "SHARED;STATIC"
    ""
    "PUBLIC_LIBS;EMBED_LIBS"
    ${ARGN})
  set(_libtype)
  if(ARG_STATIC)
    list(APPEND _libtype STATIC)
  endif()
  if(ARG_SHARED)
    list(APPEND _libtype SHARED)
  endif()
  set(_debugmsg)

  set(_embed_libs)
  set(_objects)
  set(_deps)
  foreach(lib ${ARG_EMBED_LIBS})
    # We have to handle imported vs in-tree differently:
    #   in-tree: To support arbitrary ordering, the generator expressions get
    #     set on the dependent target when it is constructed and then just
    #     eval'd here. This means we can build an aggregate from targets that
    #     may not yet be defined, which is typical for in-tree.
    #   imported: Exported properties do not support generator expressions, so
    #     we imperatively query and manage the expansion here. This is fine
    #     because imported targets will always be found/configured first and
    #     do not need to support arbitrary ordering. If CMake every supports
    #     exporting generator expressions, then this can be simplified.
    set(_is_imported OFF)
    if(TARGET ${lib})
      get_target_property(_is_imported ${lib} IMPORTED)
    endif()

    if(NOT _is_imported)
      # Evaluate the in-tree generator expressions directly (this allows target
      # order independence, since these aren't evaluated until the generate
      # phase).
      # What these expressions do:
      # In the context of this aggregate, resolve the list of OBJECTS and DEPS
      # that each library advertises and patch it into the whole.
      set(_local_objects $<TARGET_GENEX_EVAL:${name},$<TARGET_PROPERTY:${lib},MLIR_AGGREGATE_OBJECTS>>)
      set(_local_deps $<TARGET_GENEX_EVAL:${name},$<TARGET_PROPERTY:${lib},MLIR_AGGREGATE_DEPS>>)
    else()
      # It is an imported target, which can only have flat strings populated
      # (no generator expressions).
      # Rebuild the generator expressions from the imported flat string lists.
      if(NOT MLIR_INSTALL_AGGREGATE_OBJECTS)
        message(SEND_ERROR "Cannot build aggregate from imported targets which were not installed via MLIR_INSTALL_AGGREGATE_OBJECTS (for ${lib}).")
      endif()

      get_property(_has_object_lib_prop TARGET ${lib} PROPERTY MLIR_AGGREGATE_OBJECT_LIB_IMPORTED SET)
      get_property(_has_dep_libs_prop TARGET ${lib} PROPERTY MLIR_AGGREGATE_DEP_LIBS_IMPORTED SET)
      if(NOT _has_object_lib_prop OR NOT _has_dep_libs_prop)
        message(SEND_ERROR "Cannot create an aggregate out of imported ${lib}: It is missing properties indicating that it was built for aggregation")
      endif()
      get_target_property(_imp_local_object_lib ${lib} MLIR_AGGREGATE_OBJECT_LIB_IMPORTED)
      get_target_property(_imp_dep_libs ${lib} MLIR_AGGREGATE_DEP_LIBS_IMPORTED)
      set(_local_objects)
      if(_imp_local_object_lib)
        set(_local_objects "$<TARGET_OBJECTS:${_imp_local_object_lib}>")
      endif()
      # We should just be able to do this:
      #   get_mlir_filtered_link_libraries(_local_deps ${_imp_dep_libs})
      # However, CMake complains about the unqualified use of the one-arg
      # $<TARGET_PROPERTY> expression. So we do the same thing but use the
      # two-arg form which takes an explicit target.
      foreach(_imp_dep_lib ${_imp_dep_libs})
        # In English, what this expression does:
        # For each link library, resolve the property MLIR_AGGREGATE_EXCLUDE_LIBS
        # on the context target (i.e. the executable or shared library being linked)
        # and, if it is not in that list, emit the library name. Otherwise, empty.
        list(APPEND _local_deps
          "$<$<NOT:$<IN_LIST:${_imp_dep_lib},$<GENEX_EVAL:$<TARGET_PROPERTY:${name},MLIR_AGGREGATE_EXCLUDE_LIBS>>>>:${_imp_dep_lib}>"
        )
      endforeach()
    endif()

    list(APPEND _embed_libs ${lib})
    list(APPEND _objects ${_local_objects})
    list(APPEND _deps ${_local_deps})

    string(APPEND _debugmsg
      ": EMBED_LIB ${lib}:\n"
      "    OBJECTS = ${_local_objects}\n"
      "    DEPS = ${_local_deps}\n\n")
  endforeach()

  add_mlir_library(${name}
    ${_libtype}
    ${ARG_UNPARSED_ARGUMENTS}
    PARTIAL_SOURCES_INTENDED
    EXCLUDE_FROM_LIBMLIR
    LINK_LIBS PRIVATE
    ${_deps}
    ${ARG_PUBLIC_LIBS}
  )
  target_sources(${name} PRIVATE ${_objects})

  # Linux defaults to allowing undefined symbols in shared libraries whereas
  # many other platforms are more strict. We want these libraries to be
  # self contained, and we want any undefined symbols to be reported at
  # library construction time, not at library use, so make Linux strict too.
  # We make an exception for sanitizer builds, since the AddressSanitizer
  # run-time doesn't get linked into shared libraries.
  if((CMAKE_SYSTEM_NAME STREQUAL "Linux") AND (NOT LLVM_USE_SANITIZER))
    target_link_options(${name} PRIVATE
      "LINKER:-z,defs"
    )
  endif()

  # TODO: Should be transitive.
  set_target_properties(${name} PROPERTIES
    MLIR_AGGREGATE_EXCLUDE_LIBS "${_embed_libs}")
  if(WIN32)
    set_property(TARGET ${name} PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
  endif()

  # Debugging generator expressions can be hard. Uncomment the below to emit
  # files next to the library with a lot of debug information:
  # string(APPEND _debugmsg
  #   ": MAIN LIBRARY:\n"
  #   "    OBJECTS = ${_objects}\n"
  #   "    SOURCES = $<TARGET_GENEX_EVAL:${name},$<TARGET_PROPERTY:${name},SOURCES>>\n"
  #   "    DEPS = ${_deps}\n"
  #   "    LINK_LIBRARIES = $<TARGET_GENEX_EVAL:${name},$<TARGET_PROPERTY:${name},LINK_LIBRARIES>>\n"
  #   "    MLIR_AGGREGATE_EXCLUDE_LIBS = $<TARGET_GENEX_EVAL:${name},$<TARGET_PROPERTY:${name},MLIR_AGGREGATE_EXCLUDE_LIBS>>\n"
  # )
  # file(GENERATE OUTPUT
  #   "${CMAKE_CURRENT_BINARY_DIR}/${name}.aggregate_debug.txt"
  #   CONTENT "${_debugmsg}"
  # )
endfunction(add_mlir_aggregate)

# Adds an MLIR library target for installation.
# This is usually done as part of add_mlir_library but is broken out for cases
# where non-standard library builds can be installed.
function(add_mlir_library_install name)
  get_target_property(_install_with_toolchain ${name} MLIR_INSTALL_WITH_TOOLCHAIN)
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY OR _install_with_toolchain)
    get_target_export_arg(${name} MLIR export_to_mlirtargets UMBRELLA mlir-libraries)
    install(TARGETS ${name}
      COMPONENT ${name}
      ${export_to_mlirtargets}
      LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
      # Note that CMake will create a directory like:
      #   objects-${CMAKE_BUILD_TYPE}/obj.LibName
      # and put object files there.
      OBJECTS DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    )

    if (NOT LLVM_ENABLE_IDE)
      add_llvm_install_targets(install-${name}
                              DEPENDS ${name}
                              COMPONENT ${name})
    endif()
    set_property(GLOBAL APPEND PROPERTY MLIR_ALL_LIBS ${name})
    set_property(GLOBAL APPEND PROPERTY MLIR_EXPORTS ${name})
  endif()
endfunction()

# Declare an mlir library which is part of the public C-API.
function(add_mlir_public_c_api_library name)
  add_mlir_library(${name}
    ${ARGN}
    OBJECT
    EXCLUDE_FROM_LIBMLIR
    ENABLE_AGGREGATION
    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir-c
  )
  # API libraries compile with hidden visibility and macros that enable
  # exporting from the DLL. Only apply to the obj lib, which only affects
  # the exports via a shared library.
  set_target_properties(obj.${name}
    PROPERTIES
    CXX_VISIBILITY_PRESET hidden
  )
  target_compile_definitions(obj.${name}
    PRIVATE
    -DMLIR_CAPI_BUILDING_LIBRARY=1
  )
endfunction()

# Declare the library associated with a dialect.
function(add_mlir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_DIALECT_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_mlir_dialect_library)

# Declare the library associated with a conversion.
function(add_mlir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_CONVERSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_mlir_conversion_library)

# Declare the library associated with an extension.
function(add_mlir_extension_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_EXTENSION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_mlir_extension_library)

# Declare the library associated with a translation.
function(add_mlir_translation_library name)
  set_property(GLOBAL APPEND PROPERTY MLIR_TRANSLATION_LIBS ${name})
  add_mlir_library(${ARGV} DEPENDS mlir-headers)
endfunction(add_mlir_translation_library)

# Verification tools to aid debugging.
function(mlir_check_link_libraries name)
  if(TARGET ${name})
    get_target_property(type ${name} TYPE)
    if (${type} STREQUAL "INTERFACE_LIBRARY")
      get_target_property(libs ${name} INTERFACE_LINK_LIBRARIES)
    else()
      get_target_property(libs ${name} LINK_LIBRARIES)
    endif()
    # message("${name} libs are: ${libs}")
    set(linking_llvm 0)
    foreach(lib ${libs})
      if(lib)
        if(${lib} MATCHES "^LLVM$")
          set(linking_llvm 1)
        endif()
        if((${lib} MATCHES "^LLVM.+") AND ${linking_llvm})
          # This will almost always cause execution problems, since the
          # same symbol might be loaded from 2 separate libraries.  This
          # often comes from referring to an LLVM library target
          # explicitly in target_link_libraries()
          message("WARNING: ${name} links LLVM and ${lib}!")
        endif()
      endif()
    endforeach()
  endif()
endfunction(mlir_check_link_libraries)

function(mlir_check_all_link_libraries name)
  mlir_check_link_libraries(${name})
  if(TARGET ${name})
    get_target_property(libs ${name} LINK_LIBRARIES)
    # message("${name} libs are: ${libs}")
    foreach(lib ${libs})
      mlir_check_link_libraries(${lib})
    endforeach()
  endif()
endfunction(mlir_check_all_link_libraries)

# Link target against a list of MLIR libraries. If MLIR_LINK_MLIR_DYLIB is
# enabled, this will link against the MLIR dylib instead of the static
# libraries.
#
# This function should be used instead of target_link_libraries() when linking
# MLIR libraries that are part of the MLIR dylib. For libraries that are not
# part of the dylib (like test libraries), target_link_libraries() should be
# used.
function(mlir_target_link_libraries target type)
  if (TARGET obj.${target})
    target_link_libraries(obj.${target} ${ARGN})
  endif()

  if (MLIR_LINK_MLIR_DYLIB)
    target_link_libraries(${target} ${type} MLIR)
  else()
    target_link_libraries(${target} ${type} ${ARGN})
  endif()
endfunction()
