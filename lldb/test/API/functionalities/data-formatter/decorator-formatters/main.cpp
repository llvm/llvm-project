struct IntContainer {
  int items[3];
  int size;
};

template <typename T> struct Container {
  T items[3];
  int size;
};

int main() {
  IntContainer ic = {{10, 20, 0}, 2};
  Container<float> fc = {{10.5, 20.25, 0}, 2};
  return 0; // break here
}
