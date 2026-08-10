int func() {
  return 1; // In function
}

void func_down() {
  int dummy = 1; // In func_down
  (void)dummy;
}

void func_up() {
  func_down(); // In func_up
}
