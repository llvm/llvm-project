int main(int argc, char const *argv[]) {
  // Test for data breakpoint
  int x = 0;
  int arr[4] = {1, 2, 3, 4};
  for (int i = 0; i < 5; ++i) { // first loop breakpoint
    if (i == 1) {
      x = i + 1;
    } else if (i == 2) {
      arr[i] = 42;
    }
  }

  x = 1;
  for (int i = 0; i < 10; ++i) { // second loop breakpoint
    ++x;
  }
}
