add_ocaml_library(llvm_linker
  OCAML    llvm_linker
  OCAMLDEP llvm
  C        linker_ocaml
  LLVM     Linker)
