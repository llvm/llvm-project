// Check that parsing the preprocessor output creates exactly the same records
// as parsing the original files directly.

// RUN: llvm-tblgen %p/../../lib/Target/X86/X86.td -I %p/../../lib/Target/X86 -I %p/../../include -o %t -E
// RUN: llvm-tblgen %p/../../lib/Target/X86/X86.td -I %p/../../lib/Target/X86 -I %p/../../include -o %t1
// RUN: llvm-tblgen %t -o %t2
// RUN: cmp %t1 %t2
