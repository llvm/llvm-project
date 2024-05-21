add_custom_target(ocaml_make_directory
  COMMAND "${CMAKE_COMMAND}" "-E" "make_directory" "${LLVM_LIBRARY_DIR}/ocaml/llvm")
add_custom_target("ocaml_all")
set_target_properties(ocaml_all PROPERTIES FOLDER "Misc")
set_target_properties(ocaml_make_directory PROPERTIES FOLDER "Misc")

add_custom_target(ocaml_copy_files)

foreach( file dune-project configure.sh discover.sh llvm.opam )
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
  COMMAND "./configure.sh" "${LLVM_TOOLS_BINARY_DIR}/llvm-config" "static"
  DEPENDS ocaml_copy_files "${LLVM_TOOLS_BINARY_DIR}/llvm-config")

add_custom_target(ocaml_llvm_deps)
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
  TransformUtils
  AllTargetsCodeGens
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  AllTargetsMCAs)
foreach( llvm_lib ${llvm_libs} )
  add_dependencies(ocaml_llvm_deps "${llvm_lib}")
endforeach()

add_custom_target(dune_build_all
  COMMAND "${DUNE}" "build" "--release"
  COMMAND "${DUNE}" "install" "--libdir" "${LLVM_LIBRARY_DIR}/ocaml"
  DEPENDS dune_workspace ocaml_copy_files ocaml_llvm_deps)

add_dependencies(ocaml_all dune_build_all)
