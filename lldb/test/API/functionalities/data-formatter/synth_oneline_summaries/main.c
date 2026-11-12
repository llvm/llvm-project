struct MyString {
  const char *guts;
};

struct S {
  struct MyString a;
  struct MyString b;
};
void stop() {}
int main() {
  struct S s;
  s.a.guts = "hello";
  s.b.guts = "world";
  stop(); // break here
  return 0;
}
