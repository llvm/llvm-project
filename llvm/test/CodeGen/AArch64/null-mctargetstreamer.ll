; RUN: llc -mtriple=aarch64-unknown-unknown -filetype=null %s

define dso_local aarch64_vector_pcs void @foo() {
  ret void
}
