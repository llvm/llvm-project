# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc --triple=powerpc64le-unknown-linux-gnu --filetype=obj -o \
# RUN:   %t/elf_reloc.o --defsym LE=1 %s
# RUN: llvm-jitlink --noexec \
# RUN:              --abs external_data=0xdeadbeef \
# RUN:              --abs external_func=0xcafef00d \
# RUN:              --abs external_func_notoc=0x88880000 \
# RUN:              --abs external_addr14_func=0x0880 \
# RUN:              --abs external_addr16_data=0x6000 \
# RUN:              --abs external_addr32_data=0x36668840 \
# RUN:              --abs pcrel_external_var=0x36668860 \
# RUN:              --abs pcrel_external_tls=0x36668880 \
# RUN:              --check %s %t/elf_reloc.o
# RUN: llvm-mc --triple=powerpc64-unknown-linux-gnu --filetype=obj -o \
# RUN:   %t/elf_reloc.o %s
# RUN: llvm-jitlink --noexec \
# RUN:              --abs external_data=0xdeadbeef \
# RUN:              --abs external_func=0xcafef00d \
# RUN:              --abs external_func_notoc=0x88880000 \
# RUN:              --abs external_addr14_func=0x0880 \
# RUN:              --abs external_addr16_data=0x6000 \
# RUN:              --abs external_addr32_data=0x36668840 \
# RUN:              --abs pcrel_external_var=0x36668860 \
# RUN:              --abs pcrel_external_tls=0x36668880 \
# RUN:              --check %s %t/elf_reloc.o

# jitlink-check: section_addr(elf_reloc.o, $__GOT) + 0x8000 = __TOC__
  .text
  .abiversion 2
  .global external_addr32_data
  .global external_addr16_data
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

# Check R_PPC64_PCREL34
# jitlink-check: (section_addr(elf_reloc.o, .rodata.str1.1) - test_pcrel34)[33:0] = \
# jitlink-check:   ((((*{4}(test_pcrel34)) & 0x3ffff) << 16) | ((*{4}(test_pcrel34 + 4)) & 0xffff))[33:0]
  .global test_pcrel34
  .p2align 4
  .type test_pcrel34,@function
test_pcrel34:
  paddi 3, 0, .L.str@PCREL, 1
  blr
  .size test_pcrel34, .-test_pcrel34

# Check R_PPC64_ADDR14
# jitlink-check: decode_operand(reloc_addr14, 2) << 2 = external_addr14_func
  .global reloc_addr14
  .p2align 4
  .type reloc_addr14,@function
reloc_addr14:
  bca 21, 30, external_addr14_func
  .size reloc_addr14, .-reloc_addr14

# Check R_PPC64_TOC16
# jitlink-check: decode_operand(reloc_toc16, 1) & 0xffff = \
# jitlink-check:   (section_addr(elf_reloc.o, .rodata.str1.1) - __TOC__) & 0xffff
# jitlink-check: decode_operand(reloc_toc16 + 4, 1) & 0xffff = \
# jitlink-check:   ((section_addr(elf_reloc.o, .rodata.str1.1) - __TOC__) >> 16) & 0xffff
# jitlink-check: decode_operand(reloc_toc16 + 8, 1) & 0xffff = \
# jitlink-check:   (((section_addr(elf_reloc.o, .rodata.str1.1) - __TOC__) + 0x8000) >> 16) & 0xffff
# jitlink-check: decode_operand(reloc_toc16 + 12, 1) & 0xffff = \
# jitlink-check:   (section_addr(elf_reloc.o, .rodata.str1.1) - __TOC__) & 0xffff
  .global reloc_toc16
  .p2align 4
  .type reloc_toc16,@function
reloc_toc16:
.ifdef LE
  li 3, 0
  .reloc reloc_toc16, R_PPC64_TOC16, .L.str
  li 3, 0
  .reloc reloc_toc16+4, R_PPC64_TOC16_HI, .L.str
  li 3, 0
  .reloc reloc_toc16+8, R_PPC64_TOC16_HA, .L.str
  li 3, 0
  .reloc reloc_toc16+12, R_PPC64_TOC16_DS, .L.str
.else
  li 3, 0
  .reloc reloc_toc16+2, R_PPC64_TOC16, .L.str
  li 3, 0
  .reloc reloc_toc16+6, R_PPC64_TOC16_HI, .L.str
  li 3, 0
  .reloc reloc_toc16+10, R_PPC64_TOC16_HA, .L.str
  li 3, 0
  .reloc reloc_toc16+14, R_PPC64_TOC16_DS, .L.str
.endif
  blr
  .size reloc_toc16, .-reloc_toc16

# Check R_PPC64_ADDR16*
# R_PPC64_ADDR16_DS
# jitlink-check: decode_operand(reloc_addr16, 1) & 0xffff = \
# jitlink-check: external_addr16_data
# R_PPC64_ADDR16_LO
# jitlink-check: decode_operand(reloc_addr16 + 4, 1) & 0xffff = \
# jitlink-check: external_addr32_data & 0xffff
# R_PPC64_ADDR16_LO_DS
# jitlink-check: decode_operand(reloc_addr16 + 8, 1) & 0xffff = \
# jitlink-check: external_addr32_data & 0xffff
# R_PPC64_ADDR16
# jitlink-check: decode_operand(reloc_addr16 + 12, 1) & 0xffff = \
# jitlink-check: external_addr16_data
# R_PPC64_ADDR16_HI
# jitlink-check: decode_operand(reloc_addr16 + 16, 1) & 0xffff = \
# jitlink-check: (external_addr32_data >> 16) & 0xffff
# R_PPC64_ADDR16_HA
# jitlink-check: decode_operand(reloc_addr16 + 20, 1) & 0xffff = \
# jitlink-check: ((external_addr32_data + 0x8000) >> 16) & 0xffff
# R_PPC64_ADDR16_HIGH
# jitlink-check: decode_operand(reloc_addr16 + 24, 1) & 0xffff = \
# jitlink-check: (external_addr32_data >> 16) & 0xffff
# R_PPC64_ADDR16_HIGHA
# jitlink-check: decode_operand(reloc_addr16 + 28, 1) & 0xffff = \
# jitlink-check: ((external_addr32_data + 0x8000) >> 16) & 0xffff
# R_PPC64_ADDR16_HIGHER
# jitlink-check: decode_operand(reloc_addr16 + 32, 1) & 0xffff = \
# jitlink-check: (external_addr32_data >> 32) & 0xffff
# R_PPC64_ADDR16_HIGHERA
# jitlink-check: decode_operand(reloc_addr16 + 36, 1) & 0xffff = \
# jitlink-check: ((external_addr32_data + 0x8000) >> 32) & 0xffff
# R_PPC64_ADDR16_HIGHEST
# jitlink-check: decode_operand(reloc_addr16 + 40, 1) & 0xffff = \
# jitlink-check: (external_addr32_data >> 48) & 0xffff
# R_PPC64_ADDR16_HIGHESTA
# jitlink-check: decode_operand(reloc_addr16 + 44, 1) & 0xffff = \
# jitlink-check: ((external_addr32_data + 0x8000) >> 48) & 0xffff
  .global reloc_addr16
  .p2align 4
  .type reloc_addr16,@function
reloc_addr16:
.ifdef LE
  li 3, 0
  .reloc reloc_addr16, R_PPC64_ADDR16_DS, external_addr16_data
  li 3, 0
  .reloc reloc_addr16+4, R_PPC64_ADDR16_LO, external_addr32_data
  li 3, 0
  .reloc reloc_addr16+8, R_PPC64_ADDR16_LO_DS, external_addr32_data
  li 3, 0
  .reloc reloc_addr16+12, R_PPC64_ADDR16, external_addr16_data
  li 3, 0
  .reloc reloc_addr16+16, R_PPC64_ADDR16_HI, external_addr32_data
.else
  li 3, 0
  .reloc reloc_addr16+2, R_PPC64_ADDR16_DS, external_addr16_data
  li 3, 0
  .reloc reloc_addr16+6, R_PPC64_ADDR16_LO, external_addr32_data
  li 3, 0
  .reloc reloc_addr16+10, R_PPC64_ADDR16_LO_DS, external_addr32_data
  li 3, 0
  .reloc reloc_addr16+14, R_PPC64_ADDR16, external_addr16_data
  li 3, 0
  .reloc reloc_addr16+18, R_PPC64_ADDR16_HI, external_addr32_data
.endif
  li 3, external_addr32_data@ha
  li 3, external_addr32_data@high
  li 3, external_addr32_data@higha
  li 3, external_addr32_data@higher
  li 3, external_addr32_data@highera
  li 3, external_addr32_data@highest
  li 3, external_addr32_data@highesta
  blr
  .size reloc_addr16, .-reloc_addr16

# Check R_PPC64_REL16*
# jitlink-check: decode_operand(reloc_rel16, 1) & 0xffff = \
# jitlink-check:   (__TOC__ - reloc_rel16) & 0xffff
# jitlink-check: decode_operand(reloc_rel16 + 4, 1) & 0xffff = \
# jitlink-check:   ((__TOC__ - reloc_rel16) >> 16) & 0xffff
  .global reloc_rel16
  .p2align 4
  .type reloc_rel16,@function
reloc_rel16:
  li 3, .TOC.-reloc_rel16
  li 3, .TOC.-reloc_rel16@h
  blr
  .size reloc_rel16, .-reloc_rel16

# Check R_PPC64_GOT_PCREL34
# jitlink-check: (got_addr(elf_reloc.o, pcrel_external_var) - reloc_got_pcrel34)[33:0] = \
# jitlink-check:   ((((*{4}(reloc_got_pcrel34)) & 0x3ffff) << 16) | ((*{4}(reloc_got_pcrel34 + 4)) & 0xffff))[33:0]
  .global reloc_got_pcrel34
  .p2align 4
  .type reloc_got_pcrel34,@function
reloc_got_pcrel34:
  pld 3,pcrel_external_var@got@pcrel(0),1
.Lpcrel0:
  .reloc .Lpcrel0-8,R_PPC64_PCREL_OPT,.-(.Lpcrel0-8)
  blr
  .size reloc_got_pcrel34,.-reloc_got_pcrel34

  .global reloc_tlsgd_pcrel34
  .p2align 4
  .type reloc_tlsgd_pcrel34,@function
reloc_tlsgd_pcrel34:
  mflr 0
  std 0, 16(1)
  stdu 1, -32(1)
  paddi 3, 0, pcrel_external_tls@got@tlsgd@pcrel, 1
  bl __tls_get_addr@notoc(a@tlsgd)
  lwa 3, 0(3)
  addi 1, 1, 32
  ld 0, 16(1)
  mtlr 0
  blr
  .size reloc_tlsgd_pcrel34,.-reloc_tlsgd_pcrel34

  .type	.L.str,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Hey!"
	.size	.L.str, 5

 .section .toc,"aw",@progbits
.LC0:
  .tc external_data[TC],external_data
