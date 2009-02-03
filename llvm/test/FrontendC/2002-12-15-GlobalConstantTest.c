// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


const char *W = "foo";
const int X = 7;
int Y = 8;
const char * const Z = "bar";

