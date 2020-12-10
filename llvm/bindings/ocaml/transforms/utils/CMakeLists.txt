add_ocaml_library(llvm_transform_utils
  OCAML    llvm_transform_utils
  OCAMLDEP llvm
  C        transform_utils_ocaml
  LLVM     TransformUtils)
