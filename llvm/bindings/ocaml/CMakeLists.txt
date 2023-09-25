add_custom_target(ocaml_make_directory
  COMMAND "${CMAKE_COMMAND}" "-E" "make_directory" "${LLVM_LIBRARY_DIR}/ocaml/llvm")
add_custom_target("ocaml_all")
set_target_properties(ocaml_all PROPERTIES FOLDER "Misc")
set_target_properties(ocaml_make_directory PROPERTIES FOLDER "Misc")

add_custom_target(ocaml_copy_files)

foreach( file dune-project setup.sh discover.sh llvm.opam )
  add_custom_target(ocaml_copy_${file} ALL
    COMMAND "${CMAKE_COMMAND}" "-E" "copy"
      "${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${file}"
    COMMENT "Copying ${source} to build area")
  add_dependencies(ocaml_copy_files ocaml_copy_${file})
endforeach()

foreach ( dir
  all_backends analysis backends bitreader bitwriter debuginfo executionengine
  irreader linker llvm target transforms )
  add_custom_target(ocaml_copy_${dir} ALL
    COMMAND "${CMAKE_COMMAND}" "-E" "copy_directory"
      "${CMAKE_CURRENT_SOURCE_DIR}/${dir}" "${CMAKE_CURRENT_BINARY_DIR}/${dir}"
    DEPENDS ${MY_TARGET})
  add_dependencies(ocaml_copy_files ocaml_copy_${dir})
endforeach()

add_custom_target(dune_workspace ALL
  COMMAND "./setup.sh" "${CMAKE_BINARY_DIR}/bin/llvm-config" "static"
  DEPENDS
    ocaml_copy_files
    "${CMAKE_BINARY_DIR}/bin/llvm-config")

explicit_map_components_to_libraries(llvm_libs
  Core Support
  Analysis
  BitReader
  BitWriter
  ExecutionEngine MCJIT native
  IRReader
  Linker
  Target
  Passes
  TransformUtils)
foreach( llvm_lib ${llvm_libs} )
  add_dependencies(dune_workspace "${llvm_lib}")
endforeach()
explicit_map_components_to_libraries(llvm_libs ${LLVM_TARGETS_TO_BUILD})
foreach( llvm_lib ${llvm_libs} )
  add_dependencies(dune_workspace "${llvm_lib}")
endforeach()

add_custom_target(dune_build_all
  COMMAND "dune" "build" "--release"
  COMMAND "dune" "install" "--libdir" "${CMAKE_BINARY_DIR}/lib/ocaml"
  DEPENDS ocaml_copy_files dune_workspace)

add_dependencies(ocaml_all dune_build_all)
