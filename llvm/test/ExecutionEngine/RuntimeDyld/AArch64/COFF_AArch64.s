// RUN: llvm-mc -triple aarch64-windows -filetype obj -o %t.obj %s
// RUN: llvm-rtdyld -triple aarch64-windows -dummy-extern dummy=0x79563413 -dummy-extern dummyA=0x78566879 -target-addr-start=40960000000000 -verify -check %s %t.obj

  .text
  .def _bnamed
  .scl 2
	.type 32
  .endef

  .globl _bnamed
  .align 2
_bnamed:
  ret

  .def _foo
  .scl 2
	.type 32
  .endef
  .globl _foo
  .align 2
_foo:
  movz  w0, #0
  ret

  .globl _test_adr_relocation
  .align 2

# IMAGE_REL_ARM64_REL21
# rtdyld-check:  decode_operand(adr1, 1) = (_const[20:0] - adr1[20:0])
_test_adr_relocation:
adr1:
  adr x0, _const
  ret

  .globl _test_branch26_reloc
  .align 2

# IMAGE_REL_ARM64_BRANCH26, test long branch
# rtdyld-check:  decode_operand(brel, 0)[25:0] = (stub_addr(COFF_AArch64.s.tmp.obj/.text, dummy) - brel)[27:2]
_test_branch26_reloc:
brel:
  b dummy
  ret

  .globl _test_branch19_reloc
  .align 2

# IMAGE_REL_ARM64_BRANCH19
# rtdyld-check:  decode_operand(bcond, 1)[18:0] = (_foo - bcond)[20:2]
_test_branch19_reloc:
  mov x0, #3
  cmp x0, #2
bcond:
  bne _foo
  ret

  .globl _test_branch14_reloc
  .align 2

# IMAGE_REL_ARM64_BRANCH14
# rtdyld-check:  decode_operand(tbz_branch, 2)[13:0] = (_bnamed - tbz_branch)[15:2]
_test_branch14_reloc:
  mov x1, #0
tbz_branch:
  tbz x1, #0, _bnamed
  ret

  .globl  _test_adrp_ldr_reloc
  .align  2

# IMAGE_REL_ARM64_PAGEBASE_REL21
# rtdyld-check:  decode_operand(adrp1, 1) = (_const[32:12] - adrp1[32:12])
_test_adrp_ldr_reloc:
adrp1:
  adrp x0, _const

# IMAGE_REL_ARM64_PAGEOFFSET_12L
# rtdyld-check:  decode_operand(ldr1, 2) = _const[11:3]
ldr1:
  ldr  x0, [x0, #:lo12:_const]
  ret

  .globl  _test_add_reloc
  .align  2

# IMAGE_REL_ARM64_PAGEOFFSET_12A
# rtdyld-check: decode_operand(add1, 2) = (tgt+4)[11:0]
_test_add_reloc:
add1:
  add x0, x0, tgt@PAGEOFF+4
  ret

  .section .data
  .globl _test_addr64_reloc
  .align 2

# IMAGE_REL_ARM64_ADDR64
# rtdyld-check:  *{8}addr64 = tgt+4
_test_addr64_reloc:
addr64:
  .quad tgt+4

# IMAGE_REL_ARM64_ADDR32
# rtdyld-check:  *{4}_test_addr32_reloc = 0x78566879
_test_addr32_reloc:
  .long dummyA

  .globl _relocations
  .align 2

# IMAGE_REL_ARM64_ADDR32NB, RVA of the target
# rtdyld-check:  *{4}_relocations = _foo - 40960000000000
_relocations:
  .long _foo@IMGREL

# IMAGE_REL_ARM64_ADDR32NB
# rtdyld-check:  *{4}imgrel2 = _string - 40960000000000+5
imgrel2:
  .long _string@IMGREL+5

# IMAGE_REL_ARM64_SECTION
# rtdyld-check: *{2}secindex = 1
secindex:
  .secidx _test_addr32_reloc

# IMAGE_REL_ARM64_SECREL
# rtdyld-check: *{4}secrel = string - section_addr(COFF_AArch64.s.tmp.obj, .data)
secrel:
  .secrel32 string

  .globl _const
  .align 3
_const:
  .quad 4614256650576692846

tgt:
  .word 1
  .word 2
  .word 3
  .word 4
  .word 5

  .globl string
  .align 2
string:
  .asciz "Hello World\n"

  .section .rdata,"dr"
  .globl _string
  .align 2
_string:
  .asciz "Hello World\n"
