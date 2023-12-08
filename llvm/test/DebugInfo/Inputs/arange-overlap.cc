void call();

struct S {
  static void foo() { call(); call(); }
  static void bar() { call(); call(); }
  static void baz() {}
};

#ifdef FILE1
# define FUNC_NAME func1
# define FUNC_BODY \
    S::foo(); S::bar(); S::baz();
#else
# define FUNC_NAME func2
# define FUNC_BODY \
    S::bar();
#endif

void FUNC_NAME() {
  FUNC_BODY
}

// Build instructions:
// $ clang -g -fPIC -c -DFILE1 arange-overlap.cc -o obj1.o
// $ clang -g -fPIC -c arange-overlap.cc -o obj2.o
// $ clang -shared obj1.o obj2.o -o <output>
