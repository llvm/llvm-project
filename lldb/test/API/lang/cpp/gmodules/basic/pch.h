template<typename T>
class GenericContainer {
  private:
    T storage;

  public:
    GenericContainer(T value) {
      storage = value;
    };
};

typedef GenericContainer<int> IntContainer;

struct Foo {
  class Bar;
  Bar *bar;
};
