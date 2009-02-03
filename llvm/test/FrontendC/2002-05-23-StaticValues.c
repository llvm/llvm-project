// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null

/* Make sure the frontend is correctly marking static stuff as internal! */

int X;
static int Y = 12;

static void foo(int Z) {
  Y = Z;
}

void *test() {
  foo(12);
  return &Y;
}
