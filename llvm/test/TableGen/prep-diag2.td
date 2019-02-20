// RUN: not llvm-tblgen -DDIAG1 -I %p %s 2>&1 | FileCheck --check-prefixes=DIAG1 %s
// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck --check-prefixes=DIAG2 %s

#ifdef DIAG1
// DIAG1: error: Only comments are supported after #else

// Invalid #else below should be detected even if DIAG1 is not defined.
// DIAG2: error: Only comments are supported after #else
#ifdef DIAG2//DIAG2
#else/*
*/class C;
#endif
#endif // DIAG1

