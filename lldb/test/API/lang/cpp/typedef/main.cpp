template<typename T>
struct S {
  typedef T V;

  V value;
};

typedef S<float> SF;

int main (int argc, char const *argv[]) {
  SF s{ .5 };
  return 0; // Set a breakpoint here
}
