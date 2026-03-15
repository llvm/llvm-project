// waiting for upstream to fix ...
// XFAIL:* 

// RUN: llvm-tblgen -gen-x86-fold-tables -asmwriternum=1 %p/../../lib/Target/X86/X86.td -I %p/../../include -I %p/../../lib/Target/X86/ -I %p/../../include/ -I %p/../../lib/Target/ --write-if-changed  -o %t1
// RUN: cmp %p/../../lib/Target/X86/X86MemFoldTables.inc %t1
