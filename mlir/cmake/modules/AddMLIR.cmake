function(mlir_tablegen ofn)
  tablegen(MLIR ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# Declare a dialect in the include directory
function(add_mlir_dialect dialect dialect_namespace)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
  mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls)
  mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs)
  mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
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
function(add_mlir_doc doc_filename command output_file output_directory)
  set(LLVM_TARGET_DEFINITIONS ${doc_filename}.td)
  tablegen(MLIR ${output_file}.md ${command})
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
function(add_mlir_library name)
  cmake_parse_arguments(ARG
    "SHARED;INSTALL_WITH_TOOLCHAIN;EXCLUDE_FROM_LIBMLIR"
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
    if(NOT XCODE)
      # The Xcode generator doesn't handle object libraries correctly.
      list(APPEND LIBTYPE OBJECT)
    endif()
    # Test libraries and such shouldn't be include in libMLIR.so
    if(NOT ARG_EXCLUDE_FROM_LIBMLIR)
      set_property(GLOBAL APPEND PROPERTY MLIR_STATIC_LIBS ${name})
      set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS ${ARG_LINK_COMPONENTS})
      set_property(GLOBAL APPEND PROPERTY MLIR_LLVM_LINK_COMPONENTS ${LLVM_LINK_COMPONENTS})
    endif()
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
    add_mlir_library_install(${name})
  else()
    # Add empty "phony" target
    add_custom_target(${name})
  endif()
  set_target_properties(${name} PROPERTIES FOLDER "MLIR libraries")
endfunction(add_mlir_library)

# Adds an MLIR library target for installation.
# This is usually done as part of add_mlir_library but is broken out for cases
# where non-standard library builds can be installed.
function(add_mlir_library_install name)
  if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  set(export_to_mlirtargets)
  if (${name} IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
      "mlir-libraries" IN_LIST LLVM_DISTRIBUTION_COMPONENTS OR
      NOT LLVM_DISTRIBUTION_COMPONENTS)
      set(export_to_mlirtargets EXPORT MLIRTargets)
    set_property(GLOBAL PROPERTY MLIR_HAS_EXPORTS True)
  endif()

  install(TARGETS ${name}
    COMPONENT ${name}
    ${export_to_mlirtargets}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX}
    RUNTIME DESTINATION bin)

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-${name}
                            DEPENDS ${name}
                            COMPONENT ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY MLIR_ALL_LIBS ${name})
  endif()
  set_property(GLOBAL APPEND PROPERTY MLIR_EXPORTS ${name})
endfunction()

# Declare an mlir library which is part of the public C-API and will be
# compiled and exported into libMLIRPublicAPI.so/MLIRPublicAPI.dll.
# This shared library is built regardless of the overall setting of building
# libMLIR.so (which exports the C++ implementation).
function(add_mlir_public_c_api_library name)
  add_mlir_library(${name}
    ${ARGN}
    # NOTE: Generates obj.${name} which is used for shared library building.
    OBJECT
    EXCLUDE_FROM_LIBMLIR
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
  set_property(GLOBAL APPEND PROPERTY MLIR_PUBLIC_C_API_LIBS ${name})
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
