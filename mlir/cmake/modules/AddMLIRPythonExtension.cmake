################################################################################
# Build python extension
################################################################################
function(add_mlir_python_extension libname extname)
  cmake_parse_arguments(ARG
  ""
  "INSTALL_DIR"
  "SOURCES;LINK_LIBS"
  ${ARGN})
  if (ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR " Unhandled arguments to add_mlir_python_extension(${libname}, ... : ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if ("${ARG_SOURCES}" STREQUAL "")
    message(FATAL_ERROR " Missing SOURCES argument to add_mlir_python_extension(${libname}, ...")
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
  if(Python3_LIBRARIES STREQUAL "" OR NOT MLIR_PYTHON_BINDINGS_VERSION_LOCKED)
    set(PYEXT_LINK_MODE MODULE)
    set(PYEXT_LIBADD)
  else()
    set(PYEXT_LINK_MODE SHARED)
    set(PYEXT_LIBADD ${Python3_LIBRARIES})
  endif()

  # The actual extension library produces a shared-object or DLL and has
  # sources that must be compiled in accordance with pybind11 needs (RTTI and
  # exceptions).
  add_library(${libname} ${PYEXT_LINK_MODE}
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
    # Build-time RPath layouts require to be a directory one up from the
    # binary root.
    # TODO: Don't reference the LLVM_BINARY_DIR here: the invariant is that
    # the output directory must be at the same level of the lib directory
    # where libMLIR.so is installed. This is presently not optimal from a
    # project separation perspective and a discussion on how to better
    # segment MLIR libraries needs to happen.
    LIBRARY_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/python
    OUTPUT_NAME "${extname}"
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_SUFFIX}${PYTHON_MODULE_EXTENSION}"
  )

  if(WIN32)
    # Need to also set the RUNTIME_OUTPUT_DIRECTORY on Windows in order to
    # control where the .dll gets written.
    set_target_properties(
      ${libname} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${LLVM_BINARY_DIR}/python
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
    MLIRPublicAPI
    LLVMSupport
    ${ARG_LINK_LIBS}
    ${PYEXT_LIBADD}
  )

  target_link_options(${libname}
    PRIVATE
      # On Linux, disable re-export of any static linked libraries that
      # came through.
      $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
  )

  llvm_setup_rpath(${libname})

  ################################################################################
  # Install
  ################################################################################
  if (ARG_INSTALL_DIR)
    install(TARGETS ${libname}
      COMPONENT ${libname}
      LIBRARY DESTINATION ${ARG_INSTALL_DIR}
      ARCHIVE DESTINATION ${ARG_INSTALL_DIR}
      # NOTE: Even on DLL-platforms, extensions go in the lib directory tree.
      RUNTIME DESTINATION ${ARG_INSTALL_DIR}
    )
  endif()

  if (NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(
      install-${libname}
      DEPENDS ${libname}
      COMPONENT ${libname})
  endif()

endfunction()

function(add_mlir_dialect_python_bindings tblgen_target)
  cmake_parse_arguments(ARG
    ""
    "TD_FILE;DIALECT_NAME"
    "DEPENDS"
    ${ARGN})

  set(dialect_filename "_${ARG_DIALECT_NAME}_ops_gen.py")
  set(LLVM_TARGET_DEFINITIONS ${ARG_TD_FILE})
  mlir_tablegen("${dialect_filename}" -gen-python-op-bindings
                -bind-dialect=${ARG_DIALECT_NAME})
  add_public_tablegen_target(
    ${tblgen_target})
  if(ARG_DEPENDS)
    add_dependencies(${tblgen_target} ${ARG_DEPENDS})
  endif()

  add_custom_command(
    TARGET ${tblgen_target} POST_BUILD
    COMMENT "Copying generated python source \"dialects/${dialect_filename}\""
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${CMAKE_CURRENT_BINARY_DIR}/${dialect_filename}"
      "${PROJECT_BINARY_DIR}/python/mlir/dialects/${dialect_filename}")
endfunction()

