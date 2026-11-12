void bar(int const *foo) {
  __builtin_trap(); // Set break point at this line.
}

int main() {
  int foo = 5;
  bar(&foo);
}
