include(GNUInstallDirs)
include(LLVMDistributionSupport)

function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
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
  tablegen(MLIR ${output_file}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${MLIR_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(mlir-doc ${output_file}DocGen)
endfunction()

# Declare an mlir library which can be compiled in libMLIR.so
# In addition to everything that llvm_add_librar accepts, this
# also has the following option:
# EXCLUDE_FROM_LIBMLIR
#   Don't include this library in libMLIR.so.  This option should be used
#   for test libraries, executable-specific libraries, or rarely used libraries
#   with large dependencies.
# ENABLE_AGGREGATION
#   Forces generation of an OBJECT library, exports additional metadata,
#   and installs additional object files needed to include this as part of an
#   aggregate shared library.
#   TODO: Make this the default for all MLIR libraries once all libraries
#   are compatible with building an object library.
function(add_mlir_library name)
  cmake_parse_arguments(ARG
    "SHARED;INSTALL_WITH_TOOLCHAIN;EXCLUDE_FROM_LIBMLIR;DISABLE_INSTALL;ENABLE_AGGREGATION"
    ""
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
    ${ARGN})
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
  if(srcs OR ARG_ADDITIONAL_HEADERS)
    set(srcs
      ADDITIONAL_HEADERS
      ${srcs}
      ${ARG_ADDITIONAL_HEADERS} # It may contain unparsed unknown args.
      )
  endif()

  # Is an object library needed.
  set(NEEDS_OBJECT_LIB OFF)
  if(ARG_ENABLE_AGGREGATION)
    set(NEEDS_OBJECT_LIB ON)
  endif()

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
    # Test libraries and such shouldn't be include in libMLIR.so
    if(NOT ARG_EXCLUDE_FROM_LIBMLIR)
      set(NEEDS_OBJECT_LIB ON)
      set_property(GLOBAL APPEND PROPERTY MLIR_STATIC_LIBS ${name})
      set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS ${ARG_LINK_COMPONENTS})
      set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
    endif()
  endif()

  if(NEEDS_OBJECT_LIB AND NOT XCODE)
    # The Xcode generator doesn't handle object libraries correctly.
    # We special case xcode when building aggregates.
    list(APPEND LIBTYPE OBJECT)
  endif()

  # MLIR libraries uniformly depend on LLVMSupport.  Just specify it once here.
  list(APPEND ARG_LINK_COMPONENTS Support)

  # LINK_COMPONENTS is necessary to allow libLLVM.so to be properly
  # substituted for individual library dependencies if LLVM_LINK_LLVM_DYLIB
  # Perhaps this should be in llvm_add_library instead?  However, it fails
  # on libclang-cpp.so
  get_property(llvm_component_libs GLOBAL PROPERTY LLVM_COMPONENT_LIBS)
  foreach(lib ${ARG_LINK_LIBS})
    if(${lib} IN_LIST llvm_component_libs)
      message(SEND_ERROR "${name} specifies LINK_LIBS ${lib}, but LINK_LIBS cannot be used for LLVM libraries.  Please use LINK_COMPONENTS instead.")
    endif()
  endforeach()

  list(APPEND ARG_DEPENDS mlir-generic-headers)
  llvm_add_library(${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs} DEPENDS ${ARG_DEPENDS} LINK_COMPONENTS ${ARG_LINK_COMPONENTS} LINK_LIBS ${ARG_LINK_LIBS})

  if(TARGET ${name})
    target_link_libraries(${name} INTERFACE ${LLVM_COMMON_LIBS})
    if(NOT ARG_DISABLE_INSTALL)
      add_mlir_library_install(${name})
    endif()
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "MLIR libraries")

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
    else()
      list(APPEND AGGREGATE_OBJECTS "$<TARGET_OBJECTS:obj.${name}>")
      list(APPEND AGGREGATE_OBJECT_LIB "obj.${name}")
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
    if(MLIR_INSTALL_AGGREGATE_OBJECTS AND NOT ARG_DISABLE_INSTALL)
      add_mlir_library_install(obj.${name})
    endif()
  endif()
endfunction(add_mlir_library)

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
  # TODO: Should be transitive.
  set_target_properties(${name} PROPERTIES
    MLIR_AGGREGATE_EXCLUDE_LIBS "${_embed_libs}")
  if(MSVC)
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
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
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
  endif()
  set_property(GLOBAL APPEND PROPERTY MLIR_EXPORTS ${name})
endfunction()

# Declare an mlir library which is part of the public C-API.
function(add_mlir_public_c_api_library name)
  add_mlir_library(${name}
    ${ARGN}
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
