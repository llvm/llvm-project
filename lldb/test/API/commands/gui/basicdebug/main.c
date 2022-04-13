extern int func();
extern void func_up();

int main(int argc, char **argv) {
  int dummy;
  func();    // Break here
  func();    // Second
  dummy = 1; // Dummy command 1

  func_up(); // First func1 call
  dummy = 2; // Dummy command 2

  return 0;
}
