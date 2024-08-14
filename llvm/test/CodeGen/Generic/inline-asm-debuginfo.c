// RUN: clang -emit-llvm -c %s -g -o %t
// RUN: llc %t 2>&1 | FileCheck %s
void bad_asm() {
  asm volatile ("BAD SYNTAX$%"); // CHECK: inline-asm-debuginfo.c:4:{{[0-9]+}}: <inline asm>:1:14: error: unknown token in expression
}

void bad_multi_asm() {
  asm ( ";"
        "BAD SYNTAX;"   // CHECK: inline-asm-debuginfo.c:8:{{[0-9]+}}: <inline asm>:1:3: error: invalid instruction mnemonic 'bad'
        ";" );
}

void bad_multi_asm_linechg() {
  asm ( ";\n"
        "BAD SYNTAX;\n" // CHECK: inline-asm-debuginfo.c:15:{{[0-9]+}}: <inline asm>:2:1: error: invalid instruction mnemonic 'bad'
        ";\n" );
}
