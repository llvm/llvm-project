// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

void query_newnamebuf(void) { ((void)"query_newnamebuf"); }

