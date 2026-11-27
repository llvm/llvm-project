#include <cstdio>

struct Tagged {
  [[gnu::abi_tag("Default")]] Tagged() : x(15) { std::puts(__func__); }
  [[gnu::abi_tag("Value")]] Tagged(int val) : x(val) { std::puts(__func__); }
  [[gnu::abi_tag("Copy")]] Tagged(const Tagged &lhs) : x(lhs.x) {
    std::puts(__func__);
  }
  [[gnu::abi_tag("CopyAssign")]] Tagged &operator=(const Tagged &) {
    std::puts(__func__);
    return *this;
  }
  [[gnu::abi_tag("Dtor")]] ~Tagged() { std::puts(__func__); }

  int x;
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

struct HasNestedCtor {
  HasNestedCtor() {
    struct TaggedLocal {
      [[gnu::abi_tag("Local")]] TaggedLocal() { std::puts(__func__); }
    };

    struct Local {
      Local() { std::puts(__func__); }
    };

    TaggedLocal l1;
    Local l2;
    std::puts("Break nested");
  }
};

int main() {
  Tagged t;
  Tagged t1(10);
  Tagged t2(t1);
  t1 = t2;

  Base b;
  HasVirtualDtor vdtor;
  vdtor.func();

  std::puts("Break here");

  HasNestedCtor nested;

  return 0;
}
