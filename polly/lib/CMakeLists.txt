add_subdirectory(Analysis)
add_subdirectory(CodeGen)
add_subdirectory(Exchange)
add_subdirectory(Support)
add_subdirectory(JSON)

set(MODULE TRUE)
set(LLVM_NO_RTTI 1)

if (SCOPLIB_FOUND)
  set(POLLY_SCOPLIB_FILES Pocc.cpp)
endif (SCOPLIB_FOUND)

if (PLUTO_FOUND)
  set(POLLY_PLUTO_FILES Pluto.cpp)
endif (PLUTO_FOUND)

set(LLVM_USED_LIBS
  PollyAnalysis
  PollyCodeGen
  PollyExchange
  PollySupport
  PollyJSON
  )

add_polly_loadable_module(LLVMPolly
  CodePreparation.cpp
  DeadCodeElimination.cpp
  IndependentBlocks.cpp
  IndVarSimplify.cpp
  MayAliasSet.cpp
  Pocc.cpp
  RegisterPasses.cpp
  ScheduleOptimizer.cpp
  ${POLLY_SCOPLIB_FILES}
  ${POLLY_PLUTO_FILES}
  )

if (TARGET intrinsics_gen)
  # Check if we are building as part of an LLVM build
  add_dependencies(LLVMPolly intrinsics_gen)
endif()

add_dependencies(LLVMPolly
  PollyAnalysis
  PollyCodeGen
  PollyExchange
  PollySupport
  PollyJSON
  )

set_target_properties(LLVMPolly
  PROPERTIES
  LINKER_LANGUAGE CXX
  PREFIX "")

if (PLUTO_FOUND)
  target_link_libraries(LLVMPolly ${PLUTO_LIBRARY})
endif(PLUTO_FOUND)
