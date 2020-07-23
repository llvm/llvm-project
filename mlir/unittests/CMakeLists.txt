add_custom_target(MLIRUnitTests)
set_target_properties(MLIRUnitTests PROPERTIES FOLDER "MLIR Tests")

function(add_mlir_unittest test_dirname)
  add_unittest(MLIRUnitTests ${test_dirname} ${ARGN})
endfunction()

if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  add_compile_options("-Wno-suggest-override")
endif()

add_subdirectory(Analysis)
add_subdirectory(Dialect)
add_subdirectory(IR)
add_subdirectory(Pass)
add_subdirectory(SDBM)
add_subdirectory(TableGen)
