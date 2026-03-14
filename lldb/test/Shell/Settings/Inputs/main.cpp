class MyClass {
public:
  MyClass() {}
  void foo();
};

void MyClass::foo() {
  return; // Set break point at this line.
}

int main() {
  MyClass mc;
  mc.foo();
  return 0;
}
