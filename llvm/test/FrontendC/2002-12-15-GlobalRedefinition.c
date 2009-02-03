// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

extern char algbrfile[9];
char algbrfile[9] = "abcdefgh";

