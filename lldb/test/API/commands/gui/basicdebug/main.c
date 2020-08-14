extern int func();

int main(int argc, char **argv) {
  func(); // Break here
  func(); // Second
  return 0;
}
