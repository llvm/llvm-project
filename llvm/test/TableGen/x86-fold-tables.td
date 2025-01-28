// Steps to fix this test if it fails:
//   1. Update X86ManualFoldTables.def to manually correct the entries in fold tables, or
//      fix the vulnerable rules in X86FoldTablesEmitter.cpp until the diff is reasonable
//   2. cp <generated_file> x86-fold-tables.inc

// RUN: llvm-tblgen -gen-x86-fold-tables -asmwriternum=1 %p/../../lib/Target/X86/X86.td -I %p/../../lib/Target/X86 -I %p/../../include -o %t
// RUN: diff --strip-trailing-cr %p/x86-fold-tables.inc %t
