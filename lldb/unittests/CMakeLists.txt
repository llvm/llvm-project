add_custom_target(LLDBUnitTests)
set_target_properties(LLDBUnitTests PROPERTIES FOLDER "LLDB tests")

include_directories(${LLDB_SOURCE_ROOT})

set(LLDB_GTEST_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/gtest_common.h)
if (MSVC)
  list(APPEND LLVM_COMPILE_FLAGS /FI ${LLDB_GTEST_COMMON_INCLUDE})
else ()
  list(APPEND LLVM_COMPILE_FLAGS -include ${LLDB_GTEST_COMMON_INCLUDE})
endif ()

include(${LLDB_PROJECT_ROOT}/cmake/LLDBDependencies.cmake)

function(add_lldb_unittest test_name)
  add_unittest(LLDBUnitTests
    ${test_name}
    ${ARGN}
    )

  add_custom_command(
    TARGET ${test_name}
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/Inputs)

  lldb_link_common_libs(${test_name} EXE)
  target_link_libraries(${test_name} ${CLANG_USED_LIBS} ${LLDB_SYSTEM_LIBS})
  llvm_config(${test_name} ${LLVM_LINK_COMPONENTS})
endfunction()

function(add_unittest_inputs test_name inputs)
  foreach (INPUT ${inputs})
    add_custom_command(
      TARGET ${test_name}
      POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/${INPUT} ${CMAKE_CURRENT_BINARY_DIR}/Inputs
      COMMENT "Copying ${INPUT} to binary directory.")
  endforeach()
endfunction()

add_subdirectory(Core)
add_subdirectory(Editline)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(Language)
add_subdirectory(Process)
add_subdirectory(ScriptInterpreter)
add_subdirectory(Symbol)
add_subdirectory(SymbolFile)
add_subdirectory(Utility)
