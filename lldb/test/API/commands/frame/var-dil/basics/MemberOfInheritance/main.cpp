int main(int argc, char** argv)
{
  struct A {
    int a_;
  } a{1};

  struct B {
    int b_;
  } b{2};

  struct C : A, B {
    int c_;
  } c;
  //  } c{{1}, {2}, 3};
  c.a_ = 1;
  c.b_ = 2;
  c.c_ = 3;

  struct D : C {
    int d_;
    A fa_;
  } d;
  //  } d{{{1}, {2}, 3}, 4, {5}};
  d.a_ = 1;
  d.b_ = 2;
  d.c_ = 3;
  d.d_ = 4;
  d.fa_.a_ = 5;

  // Virtual inheritance example.
  struct Animal {
    virtual ~Animal() = default;
    int weight_;
  };
  struct Mammal : virtual Animal {};
  struct WingedAnimal : virtual Animal {};
  struct Bat : Mammal, WingedAnimal {
  } bat;
  bat.weight_ = 10;

 // Empty bases example.
  struct IPlugin {
    virtual ~IPlugin() {}
  };
  struct Plugin : public IPlugin {
    int x;
    int y;
  };
  Plugin plugin;
  plugin.x = 1;
  plugin.y = 2;

  struct ObjectBase {
    int x;
  };
  struct Object : ObjectBase {};
  struct Engine : Object {
    int y;
    int z;
  };

  Engine engine;
  engine.x = 1;
  engine.y = 2;
  engine.z = 3;

  // Empty multiple inheritance with empty base.
  struct Base {
    int x;
    int y;
    virtual void Do() = 0;
    virtual ~Base() {}
  };
  struct Mixin {};
  struct Parent : private Mixin, public Base {
    int z;
    virtual void Do(){};
  };
  Parent obj;
  obj.x = 1;
  obj.y = 2;
  obj.z = 3;
  Base* parent_base = &obj;
  Parent* parent = &obj;

  return 0; // Set a breakpoint here
}
