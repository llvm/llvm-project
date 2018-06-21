# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=thumbv7s-apple-ios7.0.0 -filetype=obj -o %t/MachO_Thumb.o %s
# RUN: llvm-rtdyld -triple=thumbv7s-apple-ios7.0.0 -verify -check=%s %t/MachO_Thumb.o

        .section        __TEXT,__text,regular,pure_instructions
        .syntax unified

# Add 'aaa' to the common symbols to make sure 'baz' isn't at the start of the
# section. This ensures that we test VANILLA relocation addends correctly.
        .comm   aaa, 4, 2
        .comm   baz, 4, 2


        .globl  bar
        .p2align        1
        .code   16                      @ @bar
        .thumb_func     bar

bar:
# Check lower 16-bits of section difference relocation
# rtdyld-check: decode_operand(insn1, 1) = (foo-(nextPC+8))[15:0]
insn1:
        movw    r0, :lower16:(foo-(nextPC+8))
# Check upper 16-bits of section difference relocation
# rtdyld-check: decode_operand(insn2, 2) = (foo-(nextPC+8))[31:16]
insn2:
        movt    r0, :upper16:(foo-(nextPC+8))
nextPC:
        nop

# Check stub generation for external symbols by referencing a common symbol, 'baz'.
# Check both the content of the stub, and the reference to the stub.
# Stub should contain '0xf000f8df' (ldr.w pc, [pc]), followed by the target.
#
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, baz)) = 0xf000f8df
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, baz) + 4) = baz
#
# rtdyld-check: decode_operand(insn3, 0) = stub_addr(MachO_Thumb.o, __text, baz) - (insn3 + 4)
insn3:
        bl      baz

# Check stub generation for internal symbols by referencing 'bar'.
# rtdyld-check: *{4}(stub_addr(MachO_Thumb.o, __text, bar) + 4) = bar & 0xfffffffffffffffe
insn4:
        bl      bar

        .section	__DATA,__data
  	.align	2
foo:
	.long	0

.subsections_via_symbols
