int g = 0;

void greet() { g++; } // greet decl

struct Test {
  void foo() {} // foo decl
  virtual void bar() {}
};

int main(void) {
  int var1 = 1;                         // var1 decl
  void (*func_ptr)() = &greet;          // func_ptr decl
  void (&func_ref)() = greet;           // func_ref decl
  auto member_ptr = &Test::foo;         // member_ptr decl
  auto virtual_member_ptr = &Test::bar; // virtual_member_ptr decl
  return 0;                             // break here
}
