# RUN: llvm-mca -mtriple=arm64-apple-macos -mcpu=apple-m4 -iterations=1 < %s

.text
.subsections_via_symbols
.globl _foo
_foo:
  ret
