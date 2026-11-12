class FooBase {
public:
  virtual void bar();

protected:
  FooBase();

  int x;
};

namespace ns {
class Foo2Base {
public:
  virtual void bar();

protected:
  Foo2Base();

  int x;
};

} // namespace ns
