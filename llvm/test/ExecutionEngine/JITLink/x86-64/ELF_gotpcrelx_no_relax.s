# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/gotpcrelx_no_relax.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0x7fff00000000 -slab-page-size 4096 \
# RUN:     -abs extern_low=0x00401000 \
# RUN:     -check %s %t/gotpcrelx_no_relax.o

        .text
        .globl  main
        .type   main,@function
main:   retq
        .size   main, .-main

# When JIT code is at 0x7fff00000000 and extern is at 0x00401000:
#   - isUInt<32>(0x00401000) = true   -> old code would relax (BUG)
#   - isInt<32>(displacement) = false -> new code does NOT relax (CORRECT)
#
# Verify the call was NOT relaxed -- indirect opcode ff 15 preserved:
# jitlink-check: *{1}test_call_no_relax = 0xff
# jitlink-check: *{1}test_call_no_relax+1 = 0x15
        .globl test_call_no_relax
        .p2align 4, 0x90
        .type  test_call_no_relax,@function
test_call_no_relax:
        call   *extern_low@GOTPCREL(%rip)
        .size  test_call_no_relax, .-test_call_no_relax

# Same for jmp -- verify indirect opcode ff 25 preserved:
# jitlink-check: *{1}test_jmp_no_relax = 0xff
# jitlink-check: *{1}test_jmp_no_relax+1 = 0x25
        .globl test_jmp_no_relax
        .p2align 4, 0x90
        .type  test_jmp_no_relax,@function
test_jmp_no_relax:
        jmp    *extern_low@GOTPCREL(%rip)
        .size  test_jmp_no_relax, .-test_jmp_no_relax
