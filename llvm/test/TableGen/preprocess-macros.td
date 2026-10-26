// Check that -E evaluates preprocessor directives and only outputs the tokens
// in active regions. Macros can be defined on the command line or by #define.

// RUN: llvm-tblgen -E %s | FileCheck %s --check-prefix=NONE --match-full-lines --strict-whitespace
// RUN: llvm-tblgen -E %s -DA | FileCheck %s --check-prefix=A --match-full-lines --strict-whitespace
// RUN: llvm-tblgen -E %s -DA -DB | FileCheck %s --check-prefix=AB --match-full-lines --strict-whitespace

// NONE:{{^}}def not_a ;
// NONE-NEXT:def not_c ;
// NONE-NOT:{{.}}

// A:{{^}}def a ;
// A-NEXT:def a_not_b ;
// A-NEXT:def c ;
// A-NOT:{{.}}

// AB:{{^}}def a ;
// AB-NEXT:def a_b ;
// AB-NEXT:def c ;
// AB-NOT:{{.}}

#ifdef A
def a;
#ifdef B
def a_b;
#else
def a_not_b;
#endif
#define C
#else
def not_a;
#endif

// Directives in inactive regions have no effect.
#ifdef UNDEFINED
#define C
#endif

#ifndef C
def not_c;
#else
def c;
#endif
