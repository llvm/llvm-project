int main() {
  // Scalar variables of every floating point type. The values are all exactly
  // representable so their printed value is unambiguous.
  float the_float = 3.5f;
  double the_double = 6.25;
  long double the_long_double = 10.75;

  // Edge-case values: zero, -1 and a negative value.
  float float_zero = 0.0f;
  float float_neg_one = -1.0f;
  float float_neg = -2.5f;
  double double_zero = 0.0;
  double double_neg_one = -1.0;
  double double_neg = -2.5;
  long double long_double_zero = 0.0;
  long double long_double_neg_one = -1.0;

  return 0; // break here
}
