// RUN: llvm-mc -triple aarch64-windows -filetype obj -o %t.obj %s
// RUN: llvm-rtdyld -triple aarch64-windows -target-addr-start=40960000000000 -verify -check %s %t.obj

// A symbol defined in a zero-sized section (e.g. a zero-sized object, such as
// an empty struct in C, in .bss) must still get a valid address.

  .text
  .globl f
  .p2align 2
f:
adrp_empty:
  adrp x0, empty_var
add_empty:
  add x0, x0, :lo12:empty_var
  ret

  .bss
  .globl empty_var
empty_var:

# rtdyld-check: decode_operand(adrp_empty, 1) = (empty_var[32:12] - adrp_empty[32:12])
# rtdyld-check: decode_operand(add_empty, 2) = empty_var[11:0]
# rtdyld-check: empty_var = section_addr(COFF_AArch64_empty_section.s.tmp.obj, .bss)
