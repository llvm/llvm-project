struct A {
  int i = 1;
  int member_method() {
    return i; // break in member function
  }
};
int main() {
  A a;
  int i = a.member_method();
  return i; // break in main
}
