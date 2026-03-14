void foo(int x) {}

struct FooBar {
  int i;
};

int main() {
  FooBar f;
  foo(1);
  return 0; // Break here
}
