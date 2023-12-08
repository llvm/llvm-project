bool bar(int const *foo) {
  return foo != 0; // Set break point at this line.
}

int main() {
  int foo[] = {1,2,3};
  return bar(foo);
}
