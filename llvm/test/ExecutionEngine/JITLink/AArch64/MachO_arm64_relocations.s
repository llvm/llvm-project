# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=arm64-apple-darwin19 -filetype=obj -o %t/macho_reloc.o %s
# RUN: llvm-jitlink -noexec -define-abs external_data=0xdeadbeef -define-abs external_func=0xcafef00d -check=%s %t/macho_reloc.o

        .section        __TEXT,__text,regular,pure_instructions

        .p2align  2
Lanon_func:
        ret

        .globl  named_func
        .p2align  2
named_func:
        ret

# Check ARM64_RELOC_BRANCH26 handling with a call to a local function.
# The branch instruction only encodes 26 bits of the 28-bit possible branch
# range, since the low 2 bits will always be zero.
#
# jitlink-check: decode_operand(test_local_call, 0)[25:0] = (named_func - test_local_call)[27:2]
        .globl  test_local_call
        .p2align  2
test_local_call:
        bl   named_func

        .globl  _main
        .p2align  2
_main:
        ret

# Check ARM64_RELOC_GOTPAGE21 / ARM64_RELOC_GOTPAGEOFF12 handling with a
# reference to an external symbol. Validate both the reference to the GOT entry,
# and also the content of the GOT entry.
#
# For the GOTPAGE21/ADRP instruction we have the 21-bit delta to the 4k page
# containing the GOT entry for external_data.
#
# For the GOTPAGEOFF/LDR instruction we have the 12-bit offset of the entry
# within the page.
#
# jitlink-check: *{8}(got_addr(macho_reloc.o, external_data)) = external_data
# jitlink-check: decode_operand(test_gotpage21_external, 1) = \
# jitlink-check:     (got_addr(macho_reloc.o, external_data)[32:12] - \
# jitlink-check:        test_gotpage21_external[32:12])
# jitlink-check: decode_operand(test_gotpageoff12_external, 2) = \
# jitlink-check:     got_addr(macho_reloc.o, external_data)[11:3]
        .globl  test_gotpage21_external
        .p2align  2
test_gotpage21_external:
        adrp  x0, external_data@GOTPAGE
        .globl  test_gotpageoff12_external
test_gotpageoff12_external:
        ldr   x0, [x0, external_data@GOTPAGEOFF]

# Check ARM64_RELOC_GOTPAGE21 / ARM64_RELOC_GOTPAGEOFF12 handling with a
# reference to a defined symbol. Validate both the reference to the GOT entry,
# and also the content of the GOT entry.
# jitlink-check: *{8}(got_addr(macho_reloc.o, named_data)) = named_data
# jitlink-check: decode_operand(test_gotpage21_defined, 1) = \
# jitlink-check:     (got_addr(macho_reloc.o, named_data)[32:12] - \
# jitlink-check:        test_gotpage21_defined[32:12])
# jitlink-check: decode_operand(test_gotpageoff12_defined, 2) = \
# jitlink-check:     got_addr(macho_reloc.o, named_data)[11:3]
        .globl  test_gotpage21_defined
        .p2align  2
test_gotpage21_defined:
        adrp  x0, named_data@GOTPAGE
        .globl  test_gotpageoff12_defined
test_gotpageoff12_defined:
        ldr   x0, [x0, named_data@GOTPAGEOFF]

# Check ARM64_RELOC_PAGE21 / ARM64_RELOC_PAGEOFF12 handling with a reference to
# a local symbol.
#
# For the PAGE21/ADRP instruction we have the 21-bit delta to the 4k page
# containing the global.
#
# For the GOTPAGEOFF12 relocation we test the ADD instruction, all LDR/GPR
# variants and all LDR/Neon variants.
#
# jitlink-check: decode_operand(test_page21, 1) = ((named_data + 256) - test_page21)[32:12]
# jitlink-check: decode_operand(test_pageoff12add, 2) = (named_data + 256)[11:0]
# jitlink-check: decode_operand(test_pageoff12gpr8, 2) = (named_data + 256)[11:0]
# jitlink-cherk: decode_operand(test_pageoff12gpr8s, 2) = (named_data + 256)[11:0]
# jitlink-check: decode_operand(test_pageoff12gpr16, 2) = (named_data + 256)[11:1]
# jitlink-check: decode_operand(test_pageoff12gpr16s, 2) = (named_data + 256)[11:1]
# jitlink-check: decode_operand(test_pageoff12gpr32, 2) = (named_data + 256)[11:2]
# jitlink-check: decode_operand(test_pageoff12gpr64, 2) = (named_data + 256)[11:3]
# jitlink-check: decode_operand(test_pageoff12neon8, 2) = (named_data + 256)[11:0]
# jitlink-check: decode_operand(test_pageoff12neon16, 2) = (named_data + 256)[11:1]
# jitlink-check: decode_operand(test_pageoff12neon32, 2) = (named_data + 256)[11:2]
# jitlink-check: decode_operand(test_pageoff12neon64, 2) = (named_data + 256)[11:3]
# jitlink-check: decode_operand(test_pageoff12neon128, 2) = (named_data + 256)[11:4]
        .globl  test_page21
        .p2align  2
test_page21:
        adrp  x0, named_data@PAGE + 256

        .globl  test_pageoff12add
test_pageoff12add:
        add   x0, x0, named_data@PAGEOFF + 256

        .globl  test_pageoff12gpr8
test_pageoff12gpr8:
        ldrb  w0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12gpr8s
test_pageoff12gpr8s:
        ldrsb w0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12gpr16
test_pageoff12gpr16:
        ldrh  w0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12gpr16s
test_pageoff12gpr16s:
        ldrsh w0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12gpr32
test_pageoff12gpr32:
        ldr   w0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12gpr64
test_pageoff12gpr64:
        ldr   x0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12neon8
test_pageoff12neon8:
        ldr   b0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12neon16
test_pageoff12neon16:
        ldr   h0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12neon32
test_pageoff12neon32:
        ldr   s0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12neon64
test_pageoff12neon64:
        ldr   d0, [x0, named_data@PAGEOFF + 256]

        .globl  test_pageoff12neon128
test_pageoff12neon128:
        ldr   q0, [x0, named_data@PAGEOFF + 256]

# Check that calls to external functions trigger the generation of stubs and GOT
# entries.
#
# jitlink-check: decode_operand(test_external_call, 0) = (stub_addr(macho_reloc.o, external_func) - test_external_call)[27:2]
# jitlink-check: *{8}(got_addr(macho_reloc.o, external_func)) = external_func
        .globl  test_external_call
        .p2align  2
test_external_call:
        bl   external_func

        .section        __DATA,__data

# Storage target for non-extern ARM64_RELOC_SUBTRACTOR relocs.
        .p2align  3
Lanon_data:
        .quad   0x1111111111111111

# Check ARM64_RELOC_SUBTRACTOR Quad/Long in anonymous storage with anonymous
# minuend: "LA: .quad LA - B + C". The anonymous subtrahend form
# "LA: .quad B - LA + C" is not tested as subtrahends are not permitted to be
# anonymous.
#
# Note: +8 offset in expression below to accounts for sizeof(Lanon_data).
# jitlink-check: *{8}(section_addr(macho_reloc.o, __DATA,__data) + 8) = \
# jitlink-check:     (section_addr(macho_reloc.o, __DATA,__data) + 8) - named_data + 2
        .p2align  3
Lanon_minuend_quad:
        .quad Lanon_minuend_quad - named_data + 2

# Note: +16 offset in expression below to accounts for sizeof(Lanon_data) + sizeof(Lanon_minuend_long).
# jitlink-check: *{4}(section_addr(macho_reloc.o, __DATA,__data) + 16) = \
# jitlink-check:     ((section_addr(macho_reloc.o, __DATA,__data) + 16) - named_data + 2)[31:0]
        .p2align  2
Lanon_minuend_long:
        .long Lanon_minuend_long - named_data + 2

# Named quad storage target (first named atom in __data).
# Align to 16 for use as 128-bit load target.
        .globl named_data
        .p2align  4
named_data:
        .quad   0x2222222222222222
        .quad   0x3333333333333333

# An alt-entry point for named_data
        .globl named_data_alt_entry
        .p2align  3
        .alt_entry named_data_alt_entry
named_data_alt_entry:
        .quad   0

# Check ARM64_RELOC_UNSIGNED / quad / extern handling by putting the address of
# a local named function into a quad symbol.
#
# jitlink-check: *{8}named_func_addr_quad = named_func
        .globl  named_func_addr_quad
        .p2align  3
named_func_addr_quad:
        .quad   named_func

# Check ARM64_RELOC_UNSIGNED / quad / non-extern handling by putting the
# address of a local anonymous function into a quad symbol.
#
# jitlink-check: *{8}anon_func_addr_quad = \
# jitlink-check:     section_addr(macho_reloc.o, __TEXT,__text)
        .globl  anon_func_addr_quad
        .p2align  3
anon_func_addr_quad:
        .quad   Lanon_func

# ARM64_RELOC_SUBTRACTOR Quad/Long in named storage with anonymous minuend
#
# jitlink-check: *{8}anon_minuend_quad1 = \
# jitlink-check:     section_addr(macho_reloc.o, __DATA,__data) - anon_minuend_quad1 + 2
# Only the form "B: .quad LA - B + C" is tested. The form "B: .quad B - LA + C" is
# invalid because the subtrahend can not be local.
        .globl  anon_minuend_quad1
        .p2align  3
anon_minuend_quad1:
        .quad Lanon_data - anon_minuend_quad1 + 2

# jitlink-check: *{4}anon_minuend_long1 = \
# jitlink-check:     (section_addr(macho_reloc.o, __DATA,__data) - anon_minuend_long1 + 2)[31:0]
        .globl  anon_minuend_long1
        .p2align  2
anon_minuend_long1:
        .long Lanon_data - anon_minuend_long1 + 2

# Check ARM64_RELOC_SUBTRACTOR Quad/Long in named storage with minuend and subtrahend.
# Both forms "A: .quad A - B + C" and "A: .quad B - A + C" are tested.
#
# Check "A: .quad B - A + C".
# jitlink-check: *{8}subtrahend_quad2 = (named_data - subtrahend_quad2 - 2)
        .globl  subtrahend_quad2
        .p2align  3
subtrahend_quad2:
        .quad named_data - subtrahend_quad2 - 2

# Check "A: .long B - A + C".
# jitlink-check: *{4}subtrahend_long2 = (named_data - subtrahend_long2 - 2)[31:0]
        .globl  subtrahend_long2
        .p2align  2
subtrahend_long2:
        .long named_data - subtrahend_long2 - 2

# Check "A: .quad A - B + C".
# jitlink-check: *{8}minuend_quad3 = (minuend_quad3 - named_data - 2)
        .globl  minuend_quad3
        .p2align  3
minuend_quad3:
        .quad minuend_quad3 - named_data - 2

# Check "A: .long B - A + C".
# jitlink-check: *{4}minuend_long3 = (minuend_long3 - named_data - 2)[31:0]
        .globl  minuend_long3
        .p2align  2
minuend_long3:
        .long minuend_long3 - named_data - 2

# Check ARM64_RELOC_SUBTRACTOR handling for exprs of the form
# "A: .quad/long B - C + D", where 'B' or 'C' is at a fixed offset from 'A'
# (i.e. is part of an alt_entry chain that includes 'A').
#
# Check "A: .long B - C + D" where 'B' is an alt_entry for 'A'.
# jitlink-check: *{4}subtractor_with_alt_entry_minuend_long = (subtractor_with_alt_entry_minuend_long_B - named_data + 2)[31:0]
        .globl  subtractor_with_alt_entry_minuend_long
        .p2align  2
subtractor_with_alt_entry_minuend_long:
        .long subtractor_with_alt_entry_minuend_long_B - named_data + 2

        .globl  subtractor_with_alt_entry_minuend_long_B
        .p2align  2
        .alt_entry subtractor_with_alt_entry_minuend_long_B
subtractor_with_alt_entry_minuend_long_B:
        .long 0

# Check "A: .quad B - C + D" where 'B' is an alt_entry for 'A'.
# jitlink-check: *{8}subtractor_with_alt_entry_minuend_quad = (subtractor_with_alt_entry_minuend_quad_B - named_data + 2)
        .globl  subtractor_with_alt_entry_minuend_quad
        .p2align  3
subtractor_with_alt_entry_minuend_quad:
        .quad subtractor_with_alt_entry_minuend_quad_B - named_data + 2

        .globl  subtractor_with_alt_entry_minuend_quad_B
        .p2align  3
        .alt_entry subtractor_with_alt_entry_minuend_quad_B
subtractor_with_alt_entry_minuend_quad_B:
        .quad 0

# Check "A: .long B - C + D" where 'C' is an alt_entry for 'A'.
# jitlink-check: *{4}subtractor_with_alt_entry_subtrahend_long = (named_data - subtractor_with_alt_entry_subtrahend_long_B + 2)[31:0]
        .globl  subtractor_with_alt_entry_subtrahend_long
        .p2align  2
subtractor_with_alt_entry_subtrahend_long:
        .long named_data - subtractor_with_alt_entry_subtrahend_long_B + 2

        .globl  subtractor_with_alt_entry_subtrahend_long_B
        .p2align  2
        .alt_entry subtractor_with_alt_entry_subtrahend_long_B
subtractor_with_alt_entry_subtrahend_long_B:
        .long 0

# Check "A: .quad B - C + D" where 'B' is an alt_entry for 'A'.
# jitlink-check: *{8}subtractor_with_alt_entry_subtrahend_quad = (named_data - subtractor_with_alt_entry_subtrahend_quad_B + 2)
        .globl  subtractor_with_alt_entry_subtrahend_quad
        .p2align  3
subtractor_with_alt_entry_subtrahend_quad:
        .quad named_data - subtractor_with_alt_entry_subtrahend_quad_B + 2

        .globl  subtractor_with_alt_entry_subtrahend_quad_B
        .p2align  3
        .alt_entry subtractor_with_alt_entry_subtrahend_quad_B
subtractor_with_alt_entry_subtrahend_quad_B:
        .quad 0

# Check ARM64_POINTER_TO_GOT handling.
# ARM64_POINTER_TO_GOT is a delta-32 to a GOT entry.
#
# jitlink-check: *{4}test_got = (got_addr(macho_reloc.o, external_data) - test_got)[31:0]
        .globl test_got
        .p2align  2
test_got:
        .long   external_data@got - .

# Check that unreferenced atoms in no-dead-strip sections are not dead stripped.
# We need to use a local symbol for this as any named symbol will end up in the
# ORC responsibility set, which is automatically marked live and would couse
# spurious passes.
#
# jitlink-check: *{8}section_addr(macho_reloc.o, __DATA,__nds_test_sect) = 0
        .section        __DATA,__nds_test_sect,regular,no_dead_strip
        .quad 0

# Check that unreferenced local symbols that have been marked no-dead-strip are
# not dead-striped.
#
# jitlink-check: *{8}section_addr(macho_reloc.o, __DATA,__nds_test_nlst) = 0
        .section       __DATA,__nds_test_nlst,regular
        .no_dead_strip no_dead_strip_test_symbol
no_dead_strip_test_symbol:
        .quad 0

# Check that explicit zero-fill symbols are supported
# jitlink-check: *{8}zero_fill_test = 0
        .globl zero_fill_test
.zerofill __DATA,__zero_fill_test,zero_fill_test,8,3

# Check that section alignments are respected.
# We test this by introducing two segments with alignment 8, each containing one
# byte of data. We require both symbols to have an aligned address.
#
# jitlink-check: section_alignment_check1[2:0] = 0
# jitlink-check: section_alignment_check2[2:0] = 0
        .section        __DATA,__sec_align_chk1
        .p2align 3

        .globl section_alignment_check1
section_alignment_check1:
        .byte 0

        .section        __DATA,__sec_align_chk2
        .p2align 3

        .globl section_alignment_check2
section_alignment_check2:
        .byte 0

.subsections_via_symbols
