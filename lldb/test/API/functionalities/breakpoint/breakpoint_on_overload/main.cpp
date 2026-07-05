int a_function(int x) {
  return x * x;
}

int a_function(double x) {
  return static_cast<int>(x * x);
}

int a_function(double x, int y) {
  return y * y;
}

int a_function(int x, double y) {
  return static_cast<int>(y * y);
}

int main(int argc, char const *argv[]) {
  // This is a random comment.

  int int_val = 20;
  double double_val = 20.0;

  int result = a_function(int_val);
  result += a_function(double_val);
  result += a_function(double_val, int_val);
  result += a_function(int_val, double_val);

  return result;
}
