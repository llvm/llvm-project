add_ocaml_library(llvm_bitwriter
  OCAML    llvm_bitwriter
  OCAMLDEP llvm
  C        bitwriter_ocaml
  LLVM     BitWriter)
