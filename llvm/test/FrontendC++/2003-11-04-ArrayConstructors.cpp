// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null


struct Foo {
  Foo(int);
  ~Foo();
};
void foo() {
  struct {
    Foo name;
  } Int[] =  { 1 };
}
