struct Number {
  Number(double value) : value_(value) {}
  double value_;
};

struct Calc {
  void add(const Number &number) { result_ += number.value_; }

  Number result() const { return Number(result_); }

private:
  double result_ = 0;
};

extern "C" {
bool test_sum(double v1, double v2, double expected) {
  Calc calc;
  calc.add(Number(v1));
  calc.add(Number(v2));
  Number result = calc.result();
  return expected == result.value_;
}
}