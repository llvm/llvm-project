// RUN: %llvmgcc -S %s -o - | grep volatile
// PR1647

void foo(volatile int *p)
{
p[0] = 0;
}
