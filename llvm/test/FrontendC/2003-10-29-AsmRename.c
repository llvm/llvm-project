// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


struct foo { int X; };
struct bar { int Y; };

extern int Func(struct foo*) __asm__("Func64");
extern int Func64(struct bar*);

int Func(struct foo *F) {
  return 1;
}

int Func64(struct bar* B) {
  return 0;
}


int test() {
  Func(0);    /* should be renamed to call Func64 */
  Func64(0);
}
