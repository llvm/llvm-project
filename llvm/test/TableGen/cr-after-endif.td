// LineEnd ::=  newline | return | EOF
// Preprocessor must treat the \r after `hello` as a line end so that it
// doesn't skip the first #endif thinking that it's part of the #ifdef body.
// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// CHECK: :[[@LINE+4]]:2: error: #endif without #ifdef
// CHECK: :[[@LINE+3]]:2: error: Unexpected token at top level
#ifdef FOO
hello#endif
#endif
