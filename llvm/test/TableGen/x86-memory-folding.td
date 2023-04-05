// RUN: llvm-tblgen -gen-x86-fold-tables -asmwriternum=1 %p/../../lib/Target/X86/X86.td -I %p/../../lib/Target/X86 -I %p/../../include -o %t1
// RUN: cmp %p/../../lib/Target/X86/X86MemFoldTables.inc %t1
