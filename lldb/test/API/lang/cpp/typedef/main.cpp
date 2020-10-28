template<typename T>
struct S {
  typedef T V;

  V value;
};

typedef S<float> SF;

namespace ns {
typedef S<float> SF;
}
struct ST {
  typedef S<float> SF;
};

int main (int argc, char const *argv[]) {
  SF s{ .5 };
  ns::SF in_ns;
  ST::SF in_struct;
  return 0; // Set a breakpoint here
}
