// RUN: not %llvmgcc -S %s -o /dev/null |& grep "error: assignment of read-only location"
// PR 1603
int func()
{
   const int *arr;
   arr[0] = 1;
}

