struct Wrapper {
  int value;
};

struct Outer {
  struct Wrapper inner;
};

int main() {
  struct Wrapper wrapper = {47};
  struct Outer outer = {{48}};
  return wrapper.value + outer.inner.value; // break here
}
