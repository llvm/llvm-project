# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=powerpc64le-unknown-linux-gnu --filetype=obj -o \
# RUN:   %t/elf_reloc.o %s
# RUN: llvm-jitlink --noexec \
# RUN:              --abs external_data=0xdeadbeef \
# RUN:              --abs external_func=0xcafef00d \
# RUN:              --abs external_func_notoc=0x88880000 \
# RUN:              --check %s %t/elf_reloc.o

# jitlink-check: section_addr(elf_reloc.o, $__GOT) + 0x8000 = __TOC__
  .text
  .abiversion 2
  .global main
  .p2align 4
  .type main,@function
main:
  li 3, 0
  blr
  .size main, .-main

# Check R_PPC64_REL16_HA and R_PPC64_REL16_LO
# jitlink-check: decode_operand(test_rel16, 2) & 0xffff = \
# jitlink-check:   (((__TOC__ - test_rel16) + 0x8000) >> 16) & 0xffff
# jitlink-check: decode_operand(test_rel16 + 4, 2) & 0xffff = \
# jitlink-check:   (__TOC__ - test_rel16) & 0xffff
  .global test_rel16
  .p2align 4
  .type test_re16,@function
test_rel16:
  .Ltest_rel16_begin:
  addis 2, 12, .TOC.-.Ltest_rel16_begin@ha
  addi 2, 2, .TOC.-.Ltest_rel16_begin@l
  li 3, 0
  blr
  .size test_rel16, .-test_rel16

# Check R_PPC64_ADDR64, R_PPC64_TOC16_HA and R_PPC64_TOC16_LO
# jitlink-check: *{8}(got_addr(elf_reloc.o, external_data)) = external_data
# jitlink-check: decode_operand(test_tocrel16, 2) & 0xffff = \
# jitlink-check:   (((got_addr(elf_reloc.o, external_data) - __TOC__) + 0x8000) >> 16) & 0xffff
# jitlink-check: decode_operand(test_tocrel16 + 4, 1) & 0xffff = \
# jitlink-check:   (got_addr(elf_reloc.o, external_data) - __TOC__) & 0xffff
  .global test_tocrel16
  .p2align 4
  .type test_tocrel16,@function
test_tocrel16:
  addis 3, 2, .LC0@toc@ha
  ld 3, .LC0@toc@l(3)
  blr
  .size test_tocrel16, .-test_tocrel16

# Check R_PPC64_REL24
# jitlink-check: *{8}(got_addr(elf_reloc.o, external_func)) = external_func
# jitlink-check: decode_operand(test_external_call, 0) = \
# jitlink-check:   (stub_addr(elf_reloc.o, external_func) - test_external_call) >> 2
# Check r2 is saved.
# jitlink-check: *{4}(stub_addr(elf_reloc.o, external_func)) = 0xf8410018
  .global test_external_call
  .p2align 4
  .type test_external_call,@function
test_external_call:
  bl external_func
  nop
  blr
  .size test_external_call, .-test_external_call

# FIXME: Current implementation allows only one plt call stub for a target function,
# so we can't re-use `external_func` as target here.
# Check R_PPC64_REL24_NOTOC
# jitlink-check: *{8}(got_addr(elf_reloc.o, external_func_notoc)) = external_func_notoc
# jitlink-check: decode_operand(test_external_call_notoc, 0) = \
# jitlink-check:   (stub_addr(elf_reloc.o, external_func_notoc) - test_external_call_notoc) >> 2
# jitlink-check: (*{4}(stub_addr(elf_reloc.o, external_func_notoc) + 16)) & 0xffff = \
# jitlink-check:   ((((got_addr(elf_reloc.o, external_func_notoc) - stub_addr(elf_reloc.o, external_func_notoc)) - 8) + 0x8000) >> 16) & 0xffff
# jitlink-check: (*{4}(stub_addr(elf_reloc.o, external_func_notoc) + 20)) & 0xffff = \
# jitlink-check:   ((got_addr(elf_reloc.o, external_func_notoc) - stub_addr(elf_reloc.o, external_func_notoc)) - 8) & 0xffff
  .global test_external_call_notoc
  .p2align 4
  .type test_external_call_notoc,@function
test_external_call_notoc:
  bl external_func_notoc@notoc
  blr
  .size test_external_call_notoc, .-test_external_call_notoc

 .section .toc,"aw",@progbits
.LC0:
  .tc external_data[TC],external_data
