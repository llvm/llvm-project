// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* Testcase for when struct tag conflicts with typedef name... grr */

typedef struct foo {
  struct foo *X;
  int Y;
} * foo;

foo F1;
struct foo *F2;

enum bar { test1, test2 };

typedef float bar;

enum bar B1;
bar B2;

