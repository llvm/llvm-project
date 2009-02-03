// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

#include <vector>
std::vector<int> my_method ();

int
main ()
{
  my_method ();
  return 0;
}

