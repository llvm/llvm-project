int __attribute__((vectorcall)) func(double a) {
  return (int)a;
}

int main() {
  return func(1.0); // break here
}
