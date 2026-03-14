# REQUIRES: aarch64-registered-target

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t
# RUN: llvm-objcopy %t %t.copy
# RUN: cmp %t %t.copy

.text
.globl _foo, _bar
_foo:
  ## ARM64_RELOC_ADDEND and ARM64_RELOC_BRANCH26
  bl _bar + 123

_bar:
  ret

.subsections_via_symbols
