// LineEnd ::=  newline | return | EOF
// Preprocessor must treat all the \r's as line ends.
// This means, for example, not skipping the first #endif thinking that it's
// part of the #ifdef body.
// RUN: printf "#ifdef FOO\rhello\r#endif\r#endif\r" | not llvm-tblgen 2>&1 | FileCheck %s
// CHECK: :1:26: error: #endif without #ifdef
// CHECK-NOT: error:
