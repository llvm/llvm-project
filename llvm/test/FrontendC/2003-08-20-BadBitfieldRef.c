// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

void foo()
{
  char *ap;
  ap[1] == '-' && ap[2] == 0;
}

