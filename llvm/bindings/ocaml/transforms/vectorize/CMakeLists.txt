add_ocaml_library(llvm_vectorize
  OCAML    llvm_vectorize
  OCAMLDEP llvm
  C        vectorize_ocaml
  LLVM     Vectorize)
