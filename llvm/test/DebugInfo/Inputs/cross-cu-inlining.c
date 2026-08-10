// To generate the test file:
// clang cross-cu-inlining.c -DA_C -g -emit-llvm -S -o a.ll
// clang cross-cu-inlining.c -DB_C -g -emit-llvm -S -o b.ll
// llvm-link a.ll b.ll -o ab.bc
// opt -inline ab.bc -o cross-cu-inlining.bc
// clang -c cross-cu-inlining.bc -o cross-cu-inlining.o
#ifdef A_C
int i;
int func(int);
int main() {
  return func(i);
}
#endif
#ifdef B_C
int __attribute__((always_inline)) func(int x) {
  return x * 2;
}
#endif
