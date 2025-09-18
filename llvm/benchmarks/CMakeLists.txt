set(LLVM_LINK_COMPONENTS
  AsmParser
  Core
  SandboxIR
  Support)

add_benchmark(DummyYAML DummyYAML.cpp PARTIAL_SOURCES_INTENDED)
add_benchmark(xxhash xxhash.cpp PARTIAL_SOURCES_INTENDED)
add_benchmark(GetIntrinsicForClangBuiltin GetIntrinsicForClangBuiltin.cpp PARTIAL_SOURCES_INTENDED)
add_benchmark(FormatVariadicBM FormatVariadicBM.cpp PARTIAL_SOURCES_INTENDED)
add_benchmark(GetIntrinsicInfoTableEntriesBM GetIntrinsicInfoTableEntriesBM.cpp PARTIAL_SOURCES_INTENDED)
add_benchmark(SandboxIRBench SandboxIRBench.cpp PARTIAL_SOURCES_INTENDED)

add_benchmark(RuntimeLibcallsBench RuntimeLibcalls.cpp PARTIAL_SOURCES_INTENDED)


# Extract the list of symbols in a random utility as sample data.
set(SYMBOL_TEST_DATA_FILE "sample_symbol_list.txt")

get_host_tool_path(llvm-nm LLVM_NM llvm_nm_exe llvm_nm_target)
get_host_tool_path(llc LLC llc_exe llc_target)

if(TARGET ${llc_target} AND TARGET ${llvm_nm_target})
  add_custom_command(OUTPUT ${SYMBOL_TEST_DATA_FILE}
    COMMAND ${llvm_nm_exe} --no-demangle --no-sort
    --format=just-symbols
    ${SYMBOL_TEST_DATA_SOURCE_BINARY} > ${SYMBOL_TEST_DATA_FILE}
   DEPENDS ${llvm_nm_target} ${llc_target})

  add_custom_target(generate-runtime-libcalls-sample-symbol-list
                    DEPENDS ${SYMBOL_TEST_DATA_FILE})

  add_dependencies(RuntimeLibcallsBench generate-runtime-libcalls-sample-symbol-list)
  target_compile_definitions(RuntimeLibcallsBench PRIVATE
    -DSYMBOL_TEST_DATA_FILE="${CMAKE_CURRENT_BINARY_DIR}/${SYMBOL_TEST_DATA_FILE}")
endif()
