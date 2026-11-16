static int f(int x) {
  int y = x + 1; // Set a breakpoint here.
  return y;
}

int main() {
  int a = f(1);
  a += f(2);
  return a;
}
