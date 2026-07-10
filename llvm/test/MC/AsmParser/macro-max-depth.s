// RUN: llvm-mc -triple x86_64 -defsym DEPTH=30 %s \
// RUN:   | FileCheck %s --check-prefix=PASS
// RUN: llvm-mc -triple x86_64 -defsym DEPTH=99 %s \
// RUN:   | FileCheck %s --check-prefix=PASS
// RUN: not llvm-mc -triple x86_64 -defsym DEPTH=100 %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOODEEP-DEFAULT

/// -asm-macro-max-nesting-depth overrides the default.
// RUN: llvm-mc -triple x86_64 -asm-macro-max-nesting-depth=42 \
// RUN:   -defsym DEPTH=41 %s | FileCheck %s --check-prefix=PASS
// RUN: not llvm-mc -triple x86_64 -asm-macro-max-nesting-depth=42 \
// RUN:   -defsym DEPTH=42 %s 2>&1 | FileCheck %s --check-prefix=TOODEEP-FLAG

.macro rec n
 .if \n > 0
 rec "(\n - 1)"
 .else
 .long 42
 .endif
.endm

rec DEPTH

/// DEPTH=n nests n+1 deep, counting the outermost invocation.
// PASS: .long 42
// TOODEEP-DEFAULT: error: macros cannot be nested more than 100 levels deep. Use -asm-macro-max-nesting-depth to increase this limit.
// TOODEEP-FLAG: error: macros cannot be nested more than 42 levels deep. Use -asm-macro-max-nesting-depth to increase this limit.
