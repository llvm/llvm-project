// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


char * foo() { return "\\begin{"; }
