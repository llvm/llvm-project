// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t

// This is a test that we don't crash. We used to do so by going in a infinite
// recursion trying to compute the size of a MCDwarfLineAddrFragment.

       .section        .debug_line,"",@progbits
       .text
       .file 1 "Disassembler.ii"
       .section foo
       .loc 1 1 0
       ret
