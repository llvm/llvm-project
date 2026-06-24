// RUN: not llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s 2>&1 | FileCheck %s

// When the intel syntax is enabled, to parse an operand, X86AsmParser doesn't use the method parseExpression from AsmParser
// but ParseIntelExpr which was not processing well an end of statement.

// CHECK: error: unknown token in expression
test:
i-
