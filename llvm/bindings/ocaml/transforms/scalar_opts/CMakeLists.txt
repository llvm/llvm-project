add_ocaml_library(llvm_scalar_opts
  OCAML    llvm_scalar_opts
  OCAMLDEP llvm
  C        scalar_opts_ocaml
  LLVM     Scalar)
