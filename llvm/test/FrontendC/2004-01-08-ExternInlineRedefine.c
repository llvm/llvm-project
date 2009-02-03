// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


extern __inline long int
__strtol_l (int a)
{
  return 0;
}

long int
__strtol_l (int a)
{
  return 0;
}
