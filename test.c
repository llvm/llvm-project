int main() {
  int a = 5;
  int b = 10;
  int c = a + b;

  // Dead code that should be eliminated
  int dead1 = 100;
  int dead2 = 200;
  int dead3 = dead1 + dead2; // This is never used

  // Redundant calculations
  int x = a + b;  // Same as c
  int y = 5 + 10; // Constant folding opportunity

  // Unused loop
  for (int i = 0; i < 10; i++) {
    int temp = i * 2; // Dead code in loop
  }

  return c;
}
