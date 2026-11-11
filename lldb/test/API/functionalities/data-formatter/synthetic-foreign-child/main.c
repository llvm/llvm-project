struct Wrapper {
  int value;
};

int main() {
  struct Wrapper wrapper = {47};
  return wrapper.value; // break here
}
