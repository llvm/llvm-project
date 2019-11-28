# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/macho_reloc.o %s
# RUN: llvm-jitlink -noexec -define-abs external_data=0xdeadbeef -define-abs external_func=0xcafef00d -check=%s %t/macho_reloc.o

        .section        __TEXT,__text,regular,pure_instructions

        .align  4, 0x90
Lanon_func:
        retq

        .globl  named_func
        .align  4, 0x90
named_func:
        xorq    %rax, %rax
        retq

# Check X86_64_RELOC_BRANCH handling with a call to a local function.
#
# jitlink-check: decode_operand(test_local_call, 0) = named_func - next_pc(test_local_call)
        .globl  test_local_call
        .align  4, 0x90
test_local_call:
        callq   named_func
        retq

        .globl  _main
        .align  4, 0x90
_main:
        retq

# Check X86_64_RELOC_GOTPCREL handling with a load from an external symbol.
# Validate both the reference to the GOT entry, and also the content of the GOT
# entry.
#
# jitlink-check: decode_operand(test_gotld, 4) = got_addr(macho_reloc.o, external_data) - next_pc(test_gotld)
# jitlink-check: *{8}(got_addr(macho_reloc.o, external_data)) = external_data
        .globl  test_gotld
        .align  4, 0x90
test_gotld:
        movq    external_data@GOTPCREL(%rip), %rax
        retq


# Check X86_64_RELOC_GOTPCREL handling with cmp instructions, which have
# negative addends.
#
# jitlink-check: decode_operand(test_gotcmpq, 3) = got_addr(macho_reloc.o, external_data) - next_pc(test_gotcmpq)
        .globl  test_gotcmpq
        .align  4, 0x90
test_gotcmpq:
        cmpq    $0, external_data@GOTPCREL(%rip)
        retq

# Check that calls to external functions trigger the generation of stubs and GOT
# entries.
#
# jitlink-check: decode_operand(test_external_call, 0) = stub_addr(macho_reloc.o, external_func) - next_pc(test_external_call)
# jitlink-check: *{8}(got_addr(macho_reloc.o, external_func)) = external_func
        .globl  test_external_call
        .align  4, 0x90
test_external_call:
        callq   external_func
        retq

# Check signed relocation handling:
#
# X86_64_RELOC_SIGNED / Extern -- movq address of linker global
# X86_64_RELOC_SIGNED1 / Extern -- movb immediate byte to linker global
# X86_64_RELOC_SIGNED2 / Extern -- movw immediate word to linker global
# X86_64_RELOC_SIGNED4 / Extern -- movl immediate long to linker global
#
# X86_64_RELOC_SIGNED / Anon -- movq address of linker private into register
# X86_64_RELOC_SIGNED1 / Anon -- movb immediate byte to linker private
# X86_64_RELOC_SIGNED2 / Anon -- movw immediate word to linker private
# X86_64_RELOC_SIGNED4 / Anon -- movl immediate long to linker private
signed_reloc_checks:
        .globl signed
# jitlink-check: decode_operand(signed, 4) = named_data - next_pc(signed)
signed:
        movq named_data(%rip), %rax

        .globl signed1
# jitlink-check: decode_operand(signed1, 3) = named_data - next_pc(signed1)
signed1:
        movb $0xAA, named_data(%rip)

        .globl signed2
# jitlink-check: decode_operand(signed2, 3) = named_data - next_pc(signed2)
signed2:
        movw $0xAAAA, named_data(%rip)

        .globl signed4
# jitlink-check: decode_operand(signed4, 3) = named_data - next_pc(signed4)
signed4:
        movl $0xAAAAAAAA, named_data(%rip)

        .globl signedanon
# jitlink-check: decode_operand(signedanon, 4) = section_addr(macho_reloc.o, __data) - next_pc(signedanon)
signedanon:
        movq Lanon_data(%rip), %rax

        .globl signed1anon
# jitlink-check: decode_operand(signed1anon, 3) = section_addr(macho_reloc.o, __data) - next_pc(signed1anon)
signed1anon:
        movb $0xAA, Lanon_data(%rip)

        .globl signed2anon
# jitlink-check: decode_operand(signed2anon, 3) = section_addr(macho_reloc.o, __data) - next_pc(signed2anon)
signed2anon:
        movw $0xAAAA, Lanon_data(%rip)

        .globl signed4anon
# jitlink-check: decode_operand(signed4anon, 3) = section_addr(macho_reloc.o, __data) - next_pc(signed4anon)
signed4anon:
        movl $0xAAAAAAAA, Lanon_data(%rip)



        .section        __DATA,__data

# Storage target for non-extern X86_64_RELOC_SIGNED_(1/2/4) relocs.
        .p2align  3
Lanon_data:
        .quad   0x1111111111111111

# Check X86_64_RELOC_SUBTRACTOR Quad/Long in anonymous storage with anonymous
# minuend: "LA: .quad LA - B + C". The anonymous subtrahend form
# "LA: .quad B - LA + C" is not tested as subtrahends are not permitted to be
# anonymous.
#
# Note: +8 offset in expression below to accounts for sizeof(Lanon_data).
# jitlink-check: *{8}(section_addr(macho_reloc.o, __data) + 8) = (section_addr(macho_reloc.o, __data) + 8) - named_data - 2
        .p2align  3
Lanon_minuend_quad:
        .quad Lanon_minuend_quad - named_data - 2

# Note: +16 offset in expression below to accounts for sizeof(Lanon_data) + sizeof(Lanon_minuend_long).
# jitlink-check: *{4}(section_addr(macho_reloc.o, __data) + 16) = ((section_addr(macho_reloc.o, __data) + 16) - named_data - 2)[31:0]
        .p2align  2
Lanon_minuend_long:
        .long Lanon_minuend_long - named_data - 2

# Named quad storage target (first named atom in __data).
        .globl named_data
        .p2align  3
named_data:
        .quad   0x2222222222222222

# An alt-entry point for named_data
        .globl named_data_alt_entry
        .p2align  3
        .alt_entry named_data_alt_entry
named_data_alt_entry:
        .quad   0

# Check X86_64_RELOC_UNSIGNED / quad / extern handling by putting the address of
# a local named function into a quad symbol.
#
# jitlink-check: *{8}named_func_addr_quad = named_func
        .globl  named_func_addr_quad
        .p2align  3
named_func_addr_quad:
        .quad   named_func

# Check X86_64_RELOC_UNSIGNED / long / extern handling by putting the address of
# an external function (defined to reside in the low 4Gb) into a long symbol.
#
# jitlink-check: *{4}named_func_addr_long = external_func
        .globl  named_func_addr_long
        .p2align  2
named_func_addr_long:
        .long   external_func

# Check X86_64_RELOC_UNSIGNED / quad / non-extern handling by putting the
# address of a local anonymous function into a quad symbol.
#
# jitlink-check: *{8}anon_func_addr_quad = section_addr(macho_reloc.o, __text)
        .globl  anon_func_addr_quad
        .p2align  3
anon_func_addr_quad:
        .quad   Lanon_func

# X86_64_RELOC_SUBTRACTOR Quad/Long in named storage with anonymous minuend
#
# jitlink-check: *{8}anon_minuend_quad1 = section_addr(macho_reloc.o, __data) - anon_minuend_quad1 + 2
# Only the form "B: .quad LA - B + C" is tested. The form "B: .quad B - LA + C" is
# invalid because the subtrahend can not be local.
        .globl  anon_minuend_quad1
        .p2align  3
anon_minuend_quad1:
        .quad Lanon_data - anon_minuend_quad1 + 2

# jitlink-check: *{4}anon_minuend_long1 = (section_addr(macho_reloc.o, __data) - anon_minuend_long1 + 2)[31:0]
        .globl  anon_minuend_long1
        .p2align  2
anon_minuend_long1:
        .long Lanon_data - anon_minuend_long1 + 2

# Check X86_64_RELOC_SUBTRACTOR Quad/Long in named storage with minuend and subtrahend.
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

# Check X86_64_RELOC_SUBTRACTOR handling for exprs of the form
# "A: .quad/long B - C + D", where 'B' or 'C' is at a fixed offset from 'A'
# (i.e. is part of an alt_entry chain that includes 'A').
#
# Check "A: .long B - C + D" where 'B' is an alt_entry for 'A'.
# jitlink-check: *{4}subtractor_with_alt_entry_minuend_long = (subtractor_with_alt_entry_minuend_long_B - named_data - 2)[31:0]
        .globl  subtractor_with_alt_entry_minuend_long
        .p2align  2
subtractor_with_alt_entry_minuend_long:
        .long subtractor_with_alt_entry_minuend_long_B - named_data - 2

        .globl  subtractor_with_alt_entry_minuend_long_B
        .p2align  2
        .alt_entry subtractor_with_alt_entry_minuend_long_B
subtractor_with_alt_entry_minuend_long_B:
        .long 0

# Check "A: .quad B - C + D" where 'B' is an alt_entry for 'A'.
# jitlink-check: *{8}subtractor_with_alt_entry_minuend_quad = (subtractor_with_alt_entry_minuend_quad_B - named_data - 2)
        .globl  subtractor_with_alt_entry_minuend_quad
        .p2align  3
subtractor_with_alt_entry_minuend_quad:
        .quad subtractor_with_alt_entry_minuend_quad_B - named_data - 2

        .globl  subtractor_with_alt_entry_minuend_quad_B
        .p2align  3
        .alt_entry subtractor_with_alt_entry_minuend_quad_B
subtractor_with_alt_entry_minuend_quad_B:
        .quad 0

# Check "A: .long B - C + D" where 'C' is an alt_entry for 'A'.
# jitlink-check: *{4}subtractor_with_alt_entry_subtrahend_long = (named_data - subtractor_with_alt_entry_subtrahend_long_B - 2)[31:0]
        .globl  subtractor_with_alt_entry_subtrahend_long
        .p2align  2
subtractor_with_alt_entry_subtrahend_long:
        .long named_data - subtractor_with_alt_entry_subtrahend_long_B - 2

        .globl  subtractor_with_alt_entry_subtrahend_long_B
        .p2align  2
        .alt_entry subtractor_with_alt_entry_subtrahend_long_B
subtractor_with_alt_entry_subtrahend_long_B:
        .long 0

# Check "A: .quad B - C + D" where 'B' is an alt_entry for 'A'.
# jitlink-check: *{8}subtractor_with_alt_entry_subtrahend_quad = (named_data - subtractor_with_alt_entry_subtrahend_quad_B - 2)
        .globl  subtractor_with_alt_entry_subtrahend_quad
        .p2align  3
subtractor_with_alt_entry_subtrahend_quad:
        .quad named_data - subtractor_with_alt_entry_subtrahend_quad_B - 2

        .globl  subtractor_with_alt_entry_subtrahend_quad_B
        .p2align  3
        .alt_entry subtractor_with_alt_entry_subtrahend_quad_B
subtractor_with_alt_entry_subtrahend_quad_B:
        .quad 0

# Check X86_64_RELOC_GOT handling.
# X86_64_RELOC_GOT is the data-section counterpart to X86_64_RELOC_GOTLD. It is
# handled exactly the same way, including having an implicit PC-rel offset of -4
# (despite this not making sense in a data section, and requiring an explicit
# +4 addend to cancel it out and get the correct result).
#
# jitlink-check: *{4}test_got = (got_addr(macho_reloc.o, external_data) - test_got)[31:0]
        .globl test_got
        .p2align  2
test_got:
        .long   external_data@GOTPCREL + 4

# Check that unreferenced atoms in no-dead-strip sections are not dead stripped.
# We need to use a local symbol for this as any named symbol will end up in the
# ORC responsibility set, which is automatically marked live and would couse
# spurious passes.
#
# jitlink-check: *{8}section_addr(macho_reloc.o, __nds_test_sect) = 0
        .section        __DATA,__nds_test_sect,regular,no_dead_strip
        .quad 0

# Check that unreferenced local symbols that have been marked no-dead-strip are
# not dead-striped.
#
# jitlink-check: *{8}section_addr(macho_reloc.o, __nds_test_nlst) = 0
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
