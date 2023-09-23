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
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${dir} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

add_custom_target(ocaml_copied_files
  DEPENDS dune-project setup.sh discover.sh llvm.opam)

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

#add_subdirectory(llvm)
#add_subdirectory(all_backends)
#add_subdirectory(analysis)
#add_subdirectory(backends)
#add_subdirectory(bitreader)
#add_subdirectory(bitwriter)
#add_subdirectory(debuginfo)
#add_subdirectory(irreader)
#add_subdirectory(linker)
#add_subdirectory(target)
#add_subdirectory(transforms)
#add_subdirectory(executionengine)
