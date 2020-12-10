add_ocaml_library(llvm_bitreader
  OCAML    llvm_bitreader
  OCAMLDEP llvm
  C        bitreader_ocaml
  LLVM     BitReader)
