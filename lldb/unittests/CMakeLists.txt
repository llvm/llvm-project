add_custom_target(LLDBUnitTests)
set_target_properties(LLDBUnitTests PROPERTIES FOLDER "lldb tests")
add_dependencies(lldb-test-deps LLDBUnitTests)

include_directories(${LLDB_SOURCE_ROOT})
include_directories(${LLDB_PROJECT_ROOT}/unittests)

set(LLDB_GTEST_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/gtest_common.h)
if (MSVC)
  list(APPEND LLVM_COMPILE_FLAGS /FI ${LLDB_GTEST_COMMON_INCLUDE})
else ()
  list(APPEND LLVM_COMPILE_FLAGS -include ${LLDB_GTEST_COMMON_INCLUDE})
endif ()

if (LLDB_BUILT_STANDALONE)
  # Build the gtest library needed for unittests, if we have LLVM sources
  # handy.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/unittest AND NOT TARGET gtest)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/unittest utils/unittest)
  endif()
  # LLVMTestingSupport library is needed for Process/gdb-remote.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      AND NOT TARGET LLVMTestingSupport)
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Testing/Support
      lib/Testing/Support)
  endif()
endif()

function(add_lldb_unittest test_name)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS;LINK_COMPONENTS"
    ${ARGN})

  list(APPEND LLVM_LINK_COMPONENTS ${ARG_LINK_COMPONENTS})

  add_unittest(LLDBUnitTests
    ${test_name}
    ${ARG_UNPARSED_ARGUMENTS}
    )

  add_custom_command(
    TARGET ${test_name}
    POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/Inputs)

  target_link_libraries(${test_name} PRIVATE ${ARG_LINK_LIBS})
endfunction()

function(add_unittest_inputs test_name inputs)
  foreach (INPUT ${inputs})
    add_custom_command(
      TARGET ${test_name}
      POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy ${CMAKE_CURRENT_SOURCE_DIR}/Inputs/${INPUT} ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/Inputs
      COMMENT "Copying ${INPUT} to binary directory.")
  endforeach()
endfunction()

add_subdirectory(TestingSupport)
add_subdirectory(Breakpoint)
add_subdirectory(Core)
add_subdirectory(DataFormatter)
add_subdirectory(Disassembler)
add_subdirectory(Editline)
add_subdirectory(Expression)
add_subdirectory(Host)
add_subdirectory(Interpreter)
add_subdirectory(Language)
add_subdirectory(ObjectFile)
add_subdirectory(Platform)
add_subdirectory(Process)
add_subdirectory(ScriptInterpreter)
add_subdirectory(Signals)
add_subdirectory(Symbol)
add_subdirectory(SymbolFile)
add_subdirectory(Target)
add_subdirectory(tools)
add_subdirectory(UnwindAssembly)
add_subdirectory(Utility)

if(LLDB_CAN_USE_DEBUGSERVER AND LLDB_TOOL_DEBUGSERVER_BUILD AND NOT LLDB_USE_SYSTEM_DEBUGSERVER)
  add_subdirectory(debugserver)
endif()
