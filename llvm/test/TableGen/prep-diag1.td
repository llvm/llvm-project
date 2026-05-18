// RUN: not llvm-tblgen -DDIAG1 -I %p %s 2>&1 | FileCheck --check-prefixes=DIAG1 %s
// RUN: not llvm-tblgen -DDIAG4 -I %p %s 2>&1 | FileCheck --check-prefixes=DIAG4 %s
// RUN: not llvm-tblgen -DDIAG2 -I %p %s 2>&1 | FileCheck --check-prefixes=DIAG2 %s
// RUN: not llvm-tblgen -I %p %s 2>&1 | FileCheck --check-prefixes=DIAG3 %s

#ifdef DIAG1
// DIAG1: error: only comments are supported after #define NAME
#define ENABLED1/*
*/class C;
#endif // DIAG1

#ifdef DIAG4
// DIAG4: warning: duplicate definition of macro: ENABLED1
#define ENABLED1
#define ENABLED1
#endif // DIAG4

#ifdef DIAG2
// DIAG2: error: only comments are supported after #ifdef NAME

// Invalid #ifdef below should be detected even if DIAG2 is not defined.
// DIAG3: error: only comments are supported after #ifdef NAME
#ifdef DIAG2/*
*/class C;
#endif
#endif // DIAG2
