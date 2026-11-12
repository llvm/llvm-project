int func1(int in) { return in + 1; }

int func2(int in) { return in * 2; }

int func3(int in) { return in - 3; }

int func4(int in) { return in * 4; }

int func5(int in) { return in + 5; }

int main() {
  int val = 10;
  val = func1(val);
  val = func2(val);
  val = func3(val);
  val = func4(val);
  val = func5(val); // break here
  return val;
}
