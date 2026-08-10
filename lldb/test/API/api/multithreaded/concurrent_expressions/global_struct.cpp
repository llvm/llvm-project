// This is a test program with a global struct that is inspected from several
// threads at once by test_concurrent_expressions.cpp.

enum Color { eRed = 1, eGreen = 2, eBlue = 3 };

struct Base {
  int base_field = 1;
};

struct Data : Base {
  struct Inner {
    int inner_field = 20;
  };

  int i = 42;
  Inner inner;
  Color color = eGreen;
  int array[3] = {1, 2, 3};
  const char *str = "global struct";

  static int static_field;

  int GetI() const { return i; }
};

int Data::static_field = 4711;

Data g_data;

int main() {
  // Touch the members so that nothing is dropped from the debug info.
  return g_data.GetI() + Data::static_field - 4753;
}
