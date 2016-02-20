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

  lldb_link_common_libs(${test_name} EXE)
  target_link_libraries(${test_name} ${SWIFT_ALL_LIBS} ${CLANG_ALL_LIBS} ${LLDB_SYSTEM_LIBS})
  llvm_config(${test_name} ${LLVM_LINK_COMPONENTS})
endfunction()

add_subdirectory(Editline)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(ScriptInterpreter)
add_subdirectory(Utility)
