struct Pair {
  int first;
  int second;
};

struct Container {
  int items[3];
  int size;
};

int main() {
  Pair p = {1, 2};
  Container c = {{10, 20, 30}, 3};
  return 0; // break here
}
