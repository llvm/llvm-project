foreach(t ${LLVM_TARGETS_TO_BUILD})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${t})
    add_subdirectory(${t})
  endif()
endforeach()

set(LLVM_LINK_COMPONENTS Core Target AllTargetsCodeGens AllTargetsDescs AllTargetsInfos)

add_llvm_unittest(TargetMachineCTests
  TargetMachineOptionsTest.cpp
)

set_property(TARGET TargetMachineCTests
  PROPERTY FOLDER "Tests/UnitTests/TargetTests")
