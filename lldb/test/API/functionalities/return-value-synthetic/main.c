struct Pair {
  int first;
  int second;
};

struct Pair make_pair(void) {
  struct Pair pair = {1, 2};
  return pair; // break here
}

int main(void) {
  struct Pair pair = make_pair();
  return pair.first + pair.second;
}
