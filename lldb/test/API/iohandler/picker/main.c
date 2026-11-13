int inner(int x) {
  return x + 1; // break here
}

int outer(int x) { return inner(x) * 2; }

int main() { return outer(1); }
