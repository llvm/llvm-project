// Check that -E reports errors from the lexer and preprocessor.

// RUN: echo '/*' | not llvm-tblgen -E 2>&1 | FileCheck %s --check-prefix=COMMENT
// COMMENT: error: unterminated comment

// RUN: echo 'include "foo"' | not llvm-tblgen -E 2>&1 | FileCheck %s --check-prefix=INCLUDE
// INCLUDE: error: could not find include file 'foo'

// RUN: echo '!foo' | not llvm-tblgen -E 2>&1 | FileCheck %s --check-prefix=OPERATOR
// OPERATOR: error: unknown operator

// RUN: echo '#else' | not llvm-tblgen -E 2>&1 | FileCheck %s --check-prefix=ELSE
// ELSE: error: #else without #ifdef or #ifndef
