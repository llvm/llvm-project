foreach( source dune-project setup.sh discover.sh llvm.opam )
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${source}"
    COMMAND
      "${CMAKE_COMMAND}" "-E" "copy"
        "${CMAKE_CURRENT_SOURCE_DIR}/${source}" "${CMAKE_CURRENT_BINARY_DIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
    COMMENT "Copying ${source} to build area")
endforeach()
foreach ( dir
  all_backends analysis backends bitreader bitwriter debuginfo executionengine
  irreader linker llvm target transforms )
  add_custom_target(ocaml_copy_${dir} ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${CMAKE_CURRENT_SOURCE_DIR}/${dir} ${CMAKE_CURRENT_BINARY_DIR}/${dir}
    DEPENDS ${MY_TARGET})
endforeach()

add_custom_target(ocaml_copied_files
  DEPENDS dune-project setup.sh discover.sh llvm.opam
    ocaml_copy_all_backends
    ocaml_copy_analysis
    ocaml_copy_backends
    ocaml_copy_bitreader
    ocaml_copy_bitwriter
    ocaml_copy_debuginfo
    ocaml_copy_executionengine
    ocaml_copy_irreader
    ocaml_copy_linker
    ocaml_copy_llvm
    ocaml_copy_target
    ocaml_copy_transforms)

add_custom_command(
  OUTPUT dune-workspace
  COMMAND
    ./setup.sh ${CMAKE_BINARY_DIR}/bin/llvm-config
  DEPENDS
    ${CMAKE_CURRENT_BINARY_DIR}/setup.sh
    ${CMAKE_BINARY_DIR}/bin/llvm-config)

add_custom_target(dune_workspace_target DEPENDS dune-workspace)

add_custom_target(dune_build_all
  COMMAND dune build @all --release
  DEPENDS ocaml_copied_files dune_workspace_target)

add_dependencies(ocaml_all dune_build_all)
