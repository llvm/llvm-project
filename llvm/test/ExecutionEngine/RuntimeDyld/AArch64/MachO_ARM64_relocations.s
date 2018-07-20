# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-apple-ios7.0.0 -filetype=obj -o %t/foo.o %s
# RUN: llvm-rtdyld -triple=arm64-apple-ios7.0.0 -map-section foo.o,__text=0x10bc0 -dummy-extern _dummy1=0x100000 -verify -check=%s %t/foo.o

    .section  __TEXT,__text,regular,pure_instructions
    .ios_version_min 7, 0
    .globl  _foo
    .align  2
_foo:
    movz  w0, #0
    ret

    .globl  _test_branch_reloc
    .align  2


# Test ARM64_RELOC_BRANCH26 relocation. The branch instruction only encodes 26
# bits of the 28-bit possible branch range. The lower two bits are always zero
# and therefore ignored.
# rtdyld-check:  decode_operand(br1, 0)[25:0] = (_foo - br1)[27:2]
_test_branch_reloc:
br1:
    b _foo
    ret

    .globl  _test_branch_reloc_bl
    .align  2
# rtdyld-check:  decode_operand(br2, 0)[25:0] = (_foo - br2)[27:2]
_test_branch_reloc_bl:
br2:
    bl _foo
    ret	

# Test ARM64_RELOC_PAGE21 and ARM64_RELOC_PAGEOFF12 relocation. adrp encodes
# the PC-relative page (4 KiB) difference between the adrp instruction and the
# variable ptr. ldr encodes the offset of the variable within the page. The ldr
# instruction perfroms an implicit shift on the encoded immediate (imm<<3).
# rtdyld-check:  decode_operand(adrp1, 1) = (_ptr[32:12] - adrp1[32:12])
# rtdyld-check:  decode_operand(ldr1, 2) = _ptr[11:3]
    .globl  _test_adrp_ldr
    .align  2
_test_adrp_ldr:
adrp1:
    adrp x0, _ptr@PAGE
ldr1:
    ldr  x0, [x0, _ptr@PAGEOFF]
    ret

# Test ARM64_RELOC_GOT_LOAD_PAGE21 and ARM64_RELOC_GOT_LOAD_PAGEOFF12
# relocation. adrp encodes the PC-relative page (4 KiB) difference between the
# adrp instruction and the GOT entry for ptr. ldr encodes the offset of the GOT
# entry within the page. The ldr instruction perfroms an implicit shift on the
# encoded immediate (imm<<3).
# rtdyld-check:  *{8}(stub_addr(foo.o, __text, _ptr)) = _ptr
# rtdyld-check:  decode_operand(adrp2, 1) = (stub_addr(foo.o, __text, _ptr)[32:12] - adrp2[32:12])
# rtdyld-check:  decode_operand(ldr2, 2) = stub_addr(foo.o, __text, _ptr)[11:3]
    .globl  _test_adrp_ldr
    .align  2
_test_got_adrp_ldr:
adrp2:
    adrp x0, _ptr@GOTPAGE
ldr2:
    ldr  x0, [x0, _ptr@GOTPAGEOFF]
    ret

# rtdyld-check: decode_operand(add1, 2) = (tgt+8)[11:2] << 2
    .globl  _test_explicit_addend_reloc
    .align  4
_test_explicit_addend_reloc:
add1:
    add x0, x0, tgt@PAGEOFF+8

    .align  3
tgt:
    .long 0
    .long 0
    .long 7

# Test ARM64_RELOC_UNSIGNED relocation. The absolute 64-bit address of the
# function should be stored at the 8-byte memory location.
# rtdyld-check: *{8}_ptr = _foo
    .section  __DATA,__data
    .globl  _ptr
    .align  3
    .fill 4096, 1, 0
_ptr:
    .quad _foo

# Test ARM64_RELOC_SUBTRACTOR.
# rtdyld-check: *{8}_subtractor_result = _test_branch_reloc - _foo
_subtractor_result:
    .quad _test_branch_reloc - _foo

# Test 32-bit relative ARM64_RELOC_POINTER_TO_GOT
# rtdyld-check: *{4}_pointer_to_got_32_rel = (stub_addr(foo.o, __data, _dummy1) - _pointer_to_got_32_rel)
_pointer_to_got_32_rel:
    .long _dummy1@got - .

# Test 64-bit absolute ARM64_RELOC_POINTER_TO_GOT
# rtdyld-check: *{8}_pointer_to_got_64_abs = stub_addr(foo.o, __data, _dummy1)
_pointer_to_got_64_abs:
    .quad _dummy1@got

.subsections_via_symbols