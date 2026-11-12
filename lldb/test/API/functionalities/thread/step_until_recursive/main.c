volatile int g;

__attribute__((noinline)) void rec(int n) {
  if (n > 0)
    rec(n - 1); // recursive call
  else
    g = 0; // base case
  g = n;   // after recursive call
}

int main(void) {
  rec(2);
  return 0;
}
