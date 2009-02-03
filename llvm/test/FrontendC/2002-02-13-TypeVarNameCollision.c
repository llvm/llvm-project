// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* This testcase causes a symbol table collision.  Type names and variable
 * names should be in distinct namespaces
 */

typedef struct foo {
  int X, Y;
} FOO;

static FOO foo[100];

int test() {
  return foo[4].Y;
}

