// A bare-bones llvm::Optional reimplementation.

template <typename T> struct MyOptionalStorage {
  MyOptionalStorage(T val) : value(val), hasVal(true) {}
  MyOptionalStorage() {}
  T value;
  bool hasVal = false;
};

template <typename T> struct MyOptional {
  MyOptionalStorage<T> Storage;
  MyOptional(T val) : Storage(val) {}
  MyOptional() {}
  T &operator*() { return Storage.value; }
};

void stop() {}

int main(int argc, char **argv) {
  MyOptional<int> x, y = 42;
  stop(); // break here
  return *y;
}
