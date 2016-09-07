add_custom_target(UnitTests)
set_target_properties(UnitTests PROPERTIES FOLDER "Tests")

# People tend to run the tests _before_ installing, so we don't want the install
# rpath here.
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

function(add_llvm_unittest test_dirname)
  add_unittest(UnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(ADT)
add_subdirectory(Analysis)
add_subdirectory(AsmParser)
add_subdirectory(Bitcode)
add_subdirectory(CodeGen)
add_subdirectory(DebugInfo)
add_subdirectory(ExecutionEngine)
add_subdirectory(IR)
add_subdirectory(LineEditor)
add_subdirectory(Linker)
add_subdirectory(MC)
add_subdirectory(MI)
add_subdirectory(ObjectYAML)
add_subdirectory(Option)
add_subdirectory(ProfileData)
add_subdirectory(Support)
add_subdirectory(Transforms)
