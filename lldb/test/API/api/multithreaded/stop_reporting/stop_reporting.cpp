int bkpt_function(int input) {
  return input * 10; // Set a breakpoint here:
}

int call_me_function(int input) { return input * 20; }

int main() {
  int counter = 0; // Set a breakpoint in main
  counter += call_me_function(5);
  counter += bkpt_function(10);
  counter += bkpt_function(15);
  return counter;
}
