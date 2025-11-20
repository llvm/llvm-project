; RUN: llc -mtriple aarch64 -O0 < %s

define hidden void @foo() "target-features"="+harden-sls-blr" {
entry:
  ret void
}
