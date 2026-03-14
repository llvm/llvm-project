# Enable the tests

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
include(CheckCXXCompilerFlag)

add_cxx_compiler_flag(-Wno-unused-variable)

# NOTE: Some tests use `<cassert>` to perform the test. Therefore we must
# strip -DNDEBUG from the default CMake flags in DEBUG mode.
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if( NOT uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" )
  add_definitions( -UNDEBUG )
  add_definitions(-DTEST_BENCHMARK_LIBRARY_HAS_NO_ASSERTIONS)
  # Also remove /D NDEBUG to avoid MSVC warnings about conflicting defines.
  foreach (flags_var_to_scrub
      CMAKE_CXX_FLAGS_RELEASE
      CMAKE_CXX_FLAGS_RELWITHDEBINFO
      CMAKE_CXX_FLAGS_MINSIZEREL
      CMAKE_C_FLAGS_RELEASE
      CMAKE_C_FLAGS_RELWITHDEBINFO
      CMAKE_C_FLAGS_MINSIZEREL)
    string (REGEX REPLACE "(^| )[/-]D *NDEBUG($| )" " "
      "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
  endforeach()
endif()

if (NOT BUILD_SHARED_LIBS)
  add_definitions(-DBENCHMARK_STATIC_DEFINE)
endif()

check_cxx_compiler_flag(-O3 BENCHMARK_HAS_O3_FLAG)
set(BENCHMARK_O3_FLAG "")
if (BENCHMARK_HAS_O3_FLAG)
  set(BENCHMARK_O3_FLAG "-O3")
endif()

# NOTE: These flags must be added after find_package(Threads REQUIRED) otherwise
# they will break the configuration check.
if (DEFINED BENCHMARK_CXX_LINKER_FLAGS)
  list(APPEND CMAKE_EXE_LINKER_FLAGS ${BENCHMARK_CXX_LINKER_FLAGS})
endif()

add_library(output_test_helper STATIC output_test_helper.cc output_test.h)
target_link_libraries(output_test_helper PRIVATE benchmark::benchmark)

macro(compile_benchmark_test name)
  add_executable(${name} "${name}.cc")
  target_link_libraries(${name} benchmark::benchmark ${CMAKE_THREAD_LIBS_INIT})
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC")
  target_compile_options( ${name} PRIVATE --diag_suppress partial_override )
  endif()
endmacro(compile_benchmark_test)

macro(compile_benchmark_test_with_main name)
  add_executable(${name} "${name}.cc")
  target_link_libraries(${name} benchmark::benchmark_main)
endmacro(compile_benchmark_test_with_main)

macro(compile_output_test name)
  add_executable(${name} "${name}.cc" output_test.h)
  target_link_libraries(${name} output_test_helper benchmark::benchmark_main
          ${BENCHMARK_CXX_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endmacro(compile_output_test)

macro(benchmark_add_test)
  add_test(${ARGV})
  if(WIN32 AND BUILD_SHARED_LIBS)
    cmake_parse_arguments(TEST "" "NAME" "" ${ARGN})
    set_tests_properties(${TEST_NAME} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:benchmark::benchmark>")
  endif()
endmacro(benchmark_add_test)

# Demonstration executable
compile_benchmark_test(benchmark_test)
benchmark_add_test(NAME benchmark COMMAND benchmark_test --benchmark_min_time=0.01s)

compile_benchmark_test(spec_arg_test)
benchmark_add_test(NAME spec_arg COMMAND spec_arg_test --benchmark_filter=BM_NotChosen)

compile_benchmark_test(spec_arg_verbosity_test)
benchmark_add_test(NAME spec_arg_verbosity COMMAND spec_arg_verbosity_test --v=42)

compile_benchmark_test(benchmark_setup_teardown_test)
benchmark_add_test(NAME benchmark_setup_teardown COMMAND benchmark_setup_teardown_test)

compile_benchmark_test(filter_test)
macro(add_filter_test name filter expect)
  benchmark_add_test(NAME ${name} COMMAND filter_test --benchmark_min_time=0.01s --benchmark_filter=${filter} ${expect})
  benchmark_add_test(NAME ${name}_list_only COMMAND filter_test --benchmark_list_tests --benchmark_filter=${filter} ${expect})
endmacro(add_filter_test)

compile_benchmark_test(benchmark_min_time_flag_time_test)
benchmark_add_test(NAME min_time_flag_time COMMAND benchmark_min_time_flag_time_test)

compile_benchmark_test(benchmark_min_time_flag_iters_test)
benchmark_add_test(NAME min_time_flag_iters COMMAND benchmark_min_time_flag_iters_test)

add_filter_test(filter_simple "Foo" 3)
add_filter_test(filter_simple_negative "-Foo" 2)
add_filter_test(filter_suffix "BM_.*" 4)
add_filter_test(filter_suffix_negative "-BM_.*" 1)
add_filter_test(filter_regex_all ".*" 5)
add_filter_test(filter_regex_all_negative "-.*" 0)
add_filter_test(filter_regex_blank "" 5)
add_filter_test(filter_regex_blank_negative "-" 0)
add_filter_test(filter_regex_none "monkey" 0)
add_filter_test(filter_regex_none_negative "-monkey" 5)
add_filter_test(filter_regex_wildcard ".*Foo.*" 3)
add_filter_test(filter_regex_wildcard_negative "-.*Foo.*" 2)
add_filter_test(filter_regex_begin "^BM_.*" 4)
add_filter_test(filter_regex_begin_negative "-^BM_.*" 1)
add_filter_test(filter_regex_begin2 "^N" 1)
add_filter_test(filter_regex_begin2_negative "-^N" 4)
add_filter_test(filter_regex_end ".*Ba$" 1)
add_filter_test(filter_regex_end_negative "-.*Ba$" 4)

compile_benchmark_test(options_test)
benchmark_add_test(NAME options_benchmarks COMMAND options_test --benchmark_min_time=0.01s)

compile_benchmark_test(basic_test)
benchmark_add_test(NAME basic_benchmark COMMAND basic_test --benchmark_min_time=0.01s)

compile_output_test(repetitions_test)
benchmark_add_test(NAME repetitions_benchmark COMMAND repetitions_test --benchmark_min_time=0.01s --benchmark_repetitions=3)

compile_benchmark_test(diagnostics_test)
benchmark_add_test(NAME diagnostics_test COMMAND diagnostics_test --benchmark_min_time=0.01s)

compile_benchmark_test(skip_with_error_test)
benchmark_add_test(NAME skip_with_error_test COMMAND skip_with_error_test --benchmark_min_time=0.01s)

compile_benchmark_test(donotoptimize_test)
# Enable errors for deprecated deprecations (DoNotOptimize(Tp const& value)).
check_cxx_compiler_flag(-Werror=deprecated-declarations BENCHMARK_HAS_DEPRECATED_DECLARATIONS_FLAG)
if (BENCHMARK_HAS_DEPRECATED_DECLARATIONS_FLAG)
  target_compile_options (donotoptimize_test PRIVATE "-Werror=deprecated-declarations")
endif()
# Some of the issues with DoNotOptimize only occur when optimization is enabled
check_cxx_compiler_flag(-O3 BENCHMARK_HAS_O3_FLAG)
if (BENCHMARK_HAS_O3_FLAG)
  set_target_properties(donotoptimize_test PROPERTIES COMPILE_FLAGS "-O3")
endif()
benchmark_add_test(NAME donotoptimize_test COMMAND donotoptimize_test --benchmark_min_time=0.01s)

compile_benchmark_test(fixture_test)
benchmark_add_test(NAME fixture_test COMMAND fixture_test --benchmark_min_time=0.01s)

compile_benchmark_test(register_benchmark_test)
benchmark_add_test(NAME register_benchmark_test COMMAND register_benchmark_test --benchmark_min_time=0.01s)

compile_benchmark_test(map_test)
benchmark_add_test(NAME map_test COMMAND map_test --benchmark_min_time=0.01s)

compile_benchmark_test(multiple_ranges_test)
benchmark_add_test(NAME multiple_ranges_test COMMAND multiple_ranges_test --benchmark_min_time=0.01s)

compile_benchmark_test(args_product_test)
benchmark_add_test(NAME args_product_test COMMAND args_product_test --benchmark_min_time=0.01s)

compile_benchmark_test_with_main(link_main_test)
benchmark_add_test(NAME link_main_test COMMAND link_main_test --benchmark_min_time=0.01s)

compile_output_test(reporter_output_test)
benchmark_add_test(NAME reporter_output_test COMMAND reporter_output_test --benchmark_min_time=0.01s)

compile_output_test(templated_fixture_test)
benchmark_add_test(NAME templated_fixture_test COMMAND templated_fixture_test --benchmark_min_time=0.01s)

compile_output_test(user_counters_test)
benchmark_add_test(NAME user_counters_test COMMAND user_counters_test --benchmark_min_time=0.01s)

compile_output_test(perf_counters_test)
benchmark_add_test(NAME perf_counters_test COMMAND perf_counters_test --benchmark_min_time=0.01s --benchmark_perf_counters=CYCLES,INSTRUCTIONS)

compile_output_test(internal_threading_test)
benchmark_add_test(NAME internal_threading_test COMMAND internal_threading_test --benchmark_min_time=0.01s)

compile_output_test(report_aggregates_only_test)
benchmark_add_test(NAME report_aggregates_only_test COMMAND report_aggregates_only_test --benchmark_min_time=0.01s)

compile_output_test(display_aggregates_only_test)
benchmark_add_test(NAME display_aggregates_only_test COMMAND display_aggregates_only_test --benchmark_min_time=0.01s)

compile_output_test(user_counters_tabular_test)
benchmark_add_test(NAME user_counters_tabular_test COMMAND user_counters_tabular_test --benchmark_counters_tabular=true --benchmark_min_time=0.01s)

compile_output_test(user_counters_thousands_test)
benchmark_add_test(NAME user_counters_thousands_test COMMAND user_counters_thousands_test --benchmark_min_time=0.01s)

compile_output_test(memory_manager_test)
benchmark_add_test(NAME memory_manager_test COMMAND memory_manager_test --benchmark_min_time=0.01s)

# MSVC does not allow to set the language standard to C++98/03.
if(NOT (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
  compile_benchmark_test(cxx03_test)
  set_target_properties(cxx03_test
      PROPERTIES
      CXX_STANDARD 98
      CXX_STANDARD_REQUIRED YES)
  # libstdc++ provides different definitions within <map> between dialects. When
  # LTO is enabled and -Werror is specified GCC diagnoses this ODR violation
  # causing the test to fail to compile. To prevent this we explicitly disable
  # the warning.
  check_cxx_compiler_flag(-Wno-odr BENCHMARK_HAS_WNO_ODR)
  check_cxx_compiler_flag(-Wno-lto-type-mismatch BENCHMARK_HAS_WNO_LTO_TYPE_MISMATCH)
  # Cannot set_target_properties multiple times here because the warnings will
  # be overwritten on each call
  set (DISABLE_LTO_WARNINGS "")
  if (BENCHMARK_HAS_WNO_ODR)
    set(DISABLE_LTO_WARNINGS "${DISABLE_LTO_WARNINGS} -Wno-odr")
  endif()
  if (BENCHMARK_HAS_WNO_LTO_TYPE_MISMATCH)
    set(DISABLE_LTO_WARNINGS "${DISABLE_LTO_WARNINGS} -Wno-lto-type-mismatch")
  endif()
  set_target_properties(cxx03_test PROPERTIES LINK_FLAGS "${DISABLE_LTO_WARNINGS}")
  benchmark_add_test(NAME cxx03 COMMAND cxx03_test --benchmark_min_time=0.01s)
endif()

compile_output_test(complexity_test)
benchmark_add_test(NAME complexity_benchmark COMMAND complexity_test --benchmark_min_time=1000000x)

###############################################################################
# GoogleTest Unit Tests
###############################################################################

if (BENCHMARK_ENABLE_GTEST_TESTS)
  macro(compile_gtest name)
    add_executable(${name} "${name}.cc")
    target_link_libraries(${name} benchmark::benchmark
        gmock_main ${CMAKE_THREAD_LIBS_INIT})
  endmacro(compile_gtest)

  macro(add_gtest name)
    compile_gtest(${name})
    benchmark_add_test(NAME ${name} COMMAND ${name})
    if(WIN32 AND BUILD_SHARED_LIBS)
      set_tests_properties(${name} PROPERTIES
        ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:$<TARGET_FILE_DIR:benchmark::benchmark>;PATH=path_list_prepend:$<TARGET_FILE_DIR:gmock_main>"
      )
    endif()
  endmacro()

  add_gtest(benchmark_gtest)
  add_gtest(benchmark_name_gtest)
  add_gtest(benchmark_random_interleaving_gtest)
  add_gtest(commandlineflags_gtest)
  add_gtest(statistics_gtest)
  add_gtest(string_util_gtest)
  add_gtest(perf_counters_gtest)
  add_gtest(time_unit_gtest)
  add_gtest(min_time_parse_gtest)
endif(BENCHMARK_ENABLE_GTEST_TESTS)

###############################################################################
# Assembly Unit Tests
###############################################################################

if (BENCHMARK_ENABLE_ASSEMBLY_TESTS)
  if (NOT LLVM_FILECHECK_EXE)
    message(FATAL_ERROR "LLVM FileCheck is required when including this file")
  endif()
  include(AssemblyTests.cmake)
  add_filecheck_test(donotoptimize_assembly_test)
  add_filecheck_test(state_assembly_test)
  add_filecheck_test(clobber_memory_assembly_test)
endif()



###############################################################################
# Code Coverage Configuration
###############################################################################

# Add the coverage command(s)
if(CMAKE_BUILD_TYPE)
  string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_LOWER)
endif()
if (${CMAKE_BUILD_TYPE_LOWER} MATCHES "coverage")
  find_program(GCOV gcov)
  find_program(LCOV lcov)
  find_program(GENHTML genhtml)
  find_program(CTEST ctest)
  if (GCOV AND LCOV AND GENHTML AND CTEST AND HAVE_CXX_FLAG_COVERAGE)
    add_custom_command(
      OUTPUT ${CMAKE_BINARY_DIR}/lcov/index.html
      COMMAND ${LCOV} -q -z -d .
      COMMAND ${LCOV} -q --no-external -c -b "${CMAKE_SOURCE_DIR}" -d . -o before.lcov -i
      COMMAND ${CTEST} --force-new-ctest-process
      COMMAND ${LCOV} -q --no-external -c -b "${CMAKE_SOURCE_DIR}" -d . -o after.lcov
      COMMAND ${LCOV} -q -a before.lcov -a after.lcov --output-file final.lcov
      COMMAND ${LCOV} -q -r final.lcov "'${CMAKE_SOURCE_DIR}/test/*'" -o final.lcov
      COMMAND ${GENHTML} final.lcov -o lcov --demangle-cpp --sort -p "${CMAKE_BINARY_DIR}" -t benchmark
      DEPENDS filter_test benchmark_test options_test basic_test fixture_test cxx03_test complexity_test
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Running LCOV"
    )
    add_custom_target(coverage
      DEPENDS ${CMAKE_BINARY_DIR}/lcov/index.html
      COMMENT "LCOV report at lcov/index.html"
    )
    message(STATUS "Coverage command added")
  else()
    if (HAVE_CXX_FLAG_COVERAGE)
      set(CXX_FLAG_COVERAGE_MESSAGE supported)
    else()
      set(CXX_FLAG_COVERAGE_MESSAGE unavailable)
    endif()
    message(WARNING
      "Coverage not available:\n"
      "  gcov: ${GCOV}\n"
      "  lcov: ${LCOV}\n"
      "  genhtml: ${GENHTML}\n"
      "  ctest: ${CTEST}\n"
      "  --coverage flag: ${CXX_FLAG_COVERAGE_MESSAGE}")
  endif()
endif()
