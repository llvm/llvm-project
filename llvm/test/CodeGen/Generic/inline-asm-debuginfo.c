// RUN: set +o pipefail; clang -emit-llvm -c %s -g -o /dev/stdout | llc -o /dev/null 2>&1 | FileCheck %s
void bad_asm() {
  asm volatile ("BAD SYNTAX$%"); // CHECK: inline-asm-debuginfo.c:3:16: error: unknown token in expression
}

void good_asm() {
  asm volatile ("movq $0xdeadbeef, %rax");
}

void bad_multi_asm() {
  asm ( "movl $10, %eax;"
        "BAD SYNTAX;"   // CHECK: inline-asm-debuginfo.c:11:19: error: invalid instruction mnemonic 'bad'
        "subl %ebx, %eax;" );
}

void bad_multi_asm_linechg() {
  asm ( "movl $10, %eax;\n"
        "BAD SYNTAX;\n" // CHECK: inline-asm-debuginfo.c:18:3: error: invalid instruction mnemonic 'bad'
        "subl %ebx, %eax;\n" );
}

void good_multi_asm_linechg() {
  asm ( "movl $10, %eax;\n"
        "test %rax, %rax;\n"
        "subl %ebx, %eax;\n" );
}

void bad_multi_asm_op() {
  unsigned val=1, i=0;
  asm ( "movl %1, %%eax;\n"
        "BAD SYNTAX;\n" // CHECK: inline-asm-debuginfo.c:31:3: error: invalid instruction mnemonic 'bad'
        "subl %0, %%eax;\n" : "=r" (val) : "r" (i) : );
}
