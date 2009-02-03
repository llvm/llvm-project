// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


typedef union {
   long (*ap)[4];
} ptrs;

void DoAssignIteration() {
  ptrs abase;
  abase.ap+=27;
  Assignment(*abase.ap);
}


