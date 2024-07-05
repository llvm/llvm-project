// RUN: not llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s 2>&1 | FileCheck %s

// When the intel syntax is enabled, to parse an operand, X86AsmParser doesn't use the method parseExpression from AsmParser
// but ParseIntelExpr which was not processing well an end of statement.

// CHECK: LLVM ERROR: SmallVector unable to grow. Requested capacity (4294967296) is larger than maximum value for size type (4294967295)
test:
i-
