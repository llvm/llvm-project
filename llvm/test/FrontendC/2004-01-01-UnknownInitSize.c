// RUN: %llvmgcc -S %s -o /dev/null

/*
 * This regression test ensures that the C front end can compile initializers
 * even when it cannot determine the size (as below).
*/
struct one
{
  int a;
  int values [];
};

struct one hobbit = {5, {1, 2, 3}};

