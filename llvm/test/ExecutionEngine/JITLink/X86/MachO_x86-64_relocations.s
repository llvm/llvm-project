# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/test_x86-64.o %s
# RUN: llvm-jitlink -noexec -define-abs external_data=0xdeadbeef -define-abs external_func=0xcafef00d -check=%s %t/test_x86-64.o

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
# jitlink-check: decode_operand(test_gotld, 4) = got_addr(test_x86-64.o, external_data) - next_pc(test_gotld)
# jitlink-check: *{8}(got_addr(test_x86-64.o, external_data)) = external_data
        .globl  test_gotld
        .align  4, 0x90
test_gotld:
        movq    external_data@GOTPCREL(%rip), %rax
        retq

# Check that calls to external functions trigger the generation of stubs and GOT
# entries.
#
# jitlink-check: decode_operand(test_external_call, 0) = stub_addr(test_x86-64.o, external_func) - next_pc(test_external_call)
# jitlink-check: *{8}(got_addr(test_x86-64.o, external_func)) = external_func
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
# jitlink-check: decode_operand(signedanon, 4) = section_addr(test_x86-64.o, __data) - next_pc(signedanon)
signedanon:
        movq Lanon_data(%rip), %rax

        .globl signed1anon
# jitlink-check: decode_operand(signed1anon, 3) = section_addr(test_x86-64.o, __data) - next_pc(signed1anon)
signed1anon:
        movb $0xAA, Lanon_data(%rip)

        .globl signed2anon
# jitlink-check: decode_operand(signed2anon, 3) = section_addr(test_x86-64.o, __data) - next_pc(signed2anon)
signed2anon:
        movw $0xAAAA, Lanon_data(%rip)

        .globl signed4anon
# jitlink-check: decode_operand(signed4anon, 3) = section_addr(test_x86-64.o, __data) - next_pc(signed4anon)
signed4anon:
        movl $0xAAAAAAAA, Lanon_data(%rip)



        .section        __DATA,__data

# Storage target for non-extern X86_64_RELOC_SIGNED_(1/2/4) relocs.
        .p2align  3
Lanon_data:
        .quad   0x1111111111111111

# Check X86_64_RELOC_SUBTRACTOR Quad/Long in anonymous storage with anonymous minuend
# Only the form "LA: .quad LA - B + C" is tested. The form "LA: .quad B - LA + C" is
# invalid because the minuend can not be local.
#
# Note: +8 offset in expression below to accounts for sizeof(Lanon_data).
# jitlink-check: *{8}(section_addr(test_x86-64.o, __data) + 8) = (section_addr(test_x86-64.o, __data) + 8) - named_data + 2
        .p2align  3
Lanon_minuend_quad:
        .quad Lanon_minuend_quad - named_data + 2

# Note: +16 offset in expression below to accounts for sizeof(Lanon_data) + sizeof(Lanon_minuend_long).
# jitlink-check: *{4}(section_addr(test_x86-64.o, __data) + 16) = ((section_addr(test_x86-64.o, __data) + 16) - named_data + 2)[31:0]
        .p2align  2
Lanon_minuend_long:
        .long Lanon_minuend_long - named_data + 2


# Named quad storage target (first named atom in __data).
        .globl named_data
        .p2align  3
named_data:
        .quad   0x2222222222222222

# Check X86_64_RELOC_UNSIGNED / extern handling by putting the address of a
# local named function in a pointer variable.
#
# jitlink-check: *{8}named_func_addr = named_func
        .globl  named_func_addr
        .p2align  3
named_func_addr:
        .quad   named_func

# Check X86_64_RELOC_UNSIGNED / non-extern handling by putting the address of a
# local anonymous function in a pointer variable.
#
# jitlink-check: *{8}anon_func_addr = section_addr(test_x86-64.o, __text)
        .globl  anon_func_addr
        .p2align  3
anon_func_addr:
        .quad   Lanon_func

# X86_64_RELOC_SUBTRACTOR Quad/Long in named storage with anonymous minuend
#
# jitlink-check: *{8}minuend_quad1 = section_addr(test_x86-64.o, __data) - minuend_quad1 + 2
# Only the form "B: .quad LA - B + C" is tested. The form "B: .quad B - LA + C" is
# invalid because the minuend can not be local.
        .globl  minuend_quad1
        .p2align  3
minuend_quad1:
        .quad Lanon_data - minuend_quad1 + 2

# jitlink-check: *{4}minuend_long1 = (section_addr(test_x86-64.o, __data) - minuend_long1 + 2)[31:0]
        .globl  minuend_long1
        .p2align  2
minuend_long1:
        .long Lanon_data - minuend_long1 + 2

# Check X86_64_RELOC_SUBTRACTOR Quad/Long in named storage with minuend and subtrahend.
# Both forms "A: .quad A - B + C" and "A: .quad B - A + C" are tested.
#
# Check "A: .quad B - A + C".
# jitlink-check: *{8}minuend_quad2 = (named_data - minuend_quad2 + 2)
        .globl  minuend_quad2
        .p2align  3
minuend_quad2:
        .quad named_data - minuend_quad2 + 2

# Check "A: .long B - A + C".
# jitlink-check: *{4}minuend_long2 = (named_data - minuend_long2 + 2)[31:0]
        .globl  minuend_long2
        .p2align  2
minuend_long2:
        .long named_data - minuend_long2 + 2

# Check "A: .quad A - B + C".
# jitlink-check: *{8}minuend_quad3 = (minuend_quad3 - named_data + 2)
        .globl  minuend_quad3
        .p2align  3
minuend_quad3:
        .quad minuend_quad3 - named_data + 2

# Check "A: .long B - A + C".
# jitlink-check: *{4}minuend_long3 = (minuend_long3 - named_data + 2)[31:0]
        .globl  minuend_long3
        .p2align  2
minuend_long3:
        .long minuend_long3 - named_data + 2

.subsections_via_symbols
