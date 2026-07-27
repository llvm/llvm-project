// Do not modify this file from a text editor.
// It has no whitespace, newline, or carriage return after #endif.
// It tests that the lexer doesn't crash on EOF at the end of a preprocessor
// directive.
// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// CHECK: :[[@LINE+4]]:7: error: reached EOF without matching #endif
// CHECK: :[[@LINE+2]]:8: error: the latest preprocessor control is here
// CHECK: :[[@LINE+2]]:7: error: Unexpected token at top level
#ifdef FOO
#endif