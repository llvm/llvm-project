// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* This code used to break GCC's SSA computation code.  It would create
   uses of B & C that are not dominated by their definitions.  See:
   http://gcc.gnu.org/ml/gcc/2002-03/msg00697.html
 */
int bar();
int foo()
{
  int a,b,c;

  a = b + c;
  b = bar();
  c = bar();
  return a + b + c;
}

