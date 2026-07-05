// clang -c dsym-test.c -g
// clang dsym-test.o -g -o dsym-test-exe
// dsymutil dsym-test-exe
// clang dsym-test.o -g -o dsym-test-exe-second
// dsymutil dsym-test-exe-second -o dsym-test-exe-differentname.dSYM
int main() {
  return 0;
}
