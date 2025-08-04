#include <cstdio>

struct Tagged {
  [[gnu::abi_tag("Default")]] Tagged() { std::puts(__func__); }
  [[gnu::abi_tag("Copy")]] Tagged(const Tagged &) { std::puts(__func__); }
  [[gnu::abi_tag("CopyAssign")]] Tagged &operator=(const Tagged &) {
    std::puts(__func__);
    return *this;
  }
  [[gnu::abi_tag("Dtor")]] ~Tagged() { std::puts(__func__); }
};

struct Base {
  virtual ~Base() { std::puts(__func__); }
  virtual int func() { return 5; }
};

struct HasVirtualDtor : public Base {
  int func() override { return 10; }

  [[gnu::abi_tag("VirtualDtor")]] ~HasVirtualDtor() override {
    std::puts(__func__);
  }
};

int main() {
  Tagged t1;
  Tagged t2(t1);
  t1 = t2;

  Base b;
  HasVirtualDtor vdtor;
  vdtor.func();

  std::puts("Break here");

  return 0;
}
