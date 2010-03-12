foreach(c ${LLVM_TARGETS_TO_BUILD})
  set(TARGETS_BUILT "${TARGETS_BUILT} ${c}")
endforeach(c)
set(TARGETS_TO_BUILD ${TARGETS_BUILT})

# FIXME: This won't work for project files, we need to use a --param.
set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib/${CMAKE_CFG_INTDIR}")
set(SHLIBEXT "${LTDL_SHLIB_EXT}")

if(BUILD_SHARED_LIBS)
  set(LLVM_SHARED_LIBS_ENABLED "1")
else()
  set(LLVM_SHARED_LIBS_ENABLED "0")
endif(BUILD_SHARED_LIBS)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(SHLIBPATH_VAR "DYLD_LIBRARY_PATH")
else() # Default for all other unix like systems.
  # CMake hardcodes the library locaction using rpath.
  # Therefore LD_LIBRARY_PATH is not required to run binaries in the
  # build dir. We pass it anyways. 
  set(SHLIBPATH_VAR "LD_LIBRARY_PATH")
endif()

include(FindPythonInterp)
if(PYTHONINTERP_FOUND)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/site.exp.in
    ${CMAKE_CURRENT_BINARY_DIR}/site.exp)

  MAKE_DIRECTORY(${CMAKE_CURRENT_BINARY_DIR}/Unit)

  add_custom_target(check
    COMMAND sed -e "s#\@LLVM_SOURCE_DIR\@#${LLVM_MAIN_SRC_DIR}#"
                -e "s#\@LLVM_BINARY_DIR\@#${LLVM_BINARY_DIR}#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_BINARY_DIR}/${CMAKE_CFG_INTDIR}#"
                -e "s#\@LLVMGCCDIR\@##"
                ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in >
                ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    COMMAND sed -e "s#\@LLVM_SOURCE_DIR\@#${LLVM_MAIN_SRC_DIR}#"
                -e "s#\@LLVM_BINARY_DIR\@#${LLVM_BINARY_DIR}#"
                -e "s#\@LLVM_TOOLS_DIR\@#${LLVM_TOOLS_BINARY_DIR}/${CMAKE_CFG_INTDIR}#"
                -e "s#\@LLVMGCCDIR\@##"
                -e "s#\@LLVM_BUILD_MODE\@#${CMAKE_CFG_INTDIR}#"
                -e "s#\@ENABLE_SHARED\@#${LLVM_SHARED_LIBS_ENABLED}#"
                -e "s#\@SHLIBPATH_VAR\@#${SHLIBPATH_VAR}#"
                ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in >
                ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
    COMMAND ${PYTHON_EXECUTABLE}
                ${LLVM_SOURCE_DIR}/utils/lit/lit.py
                --param llvm_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                --param llvm_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
                -sv
                ${CMAKE_CURRENT_BINARY_DIR}
                DEPENDS
                COMMENT "Running LLVM regression tests")

endif()
