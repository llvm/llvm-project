double eval(double a, double b, int op) {
  switch (op) {
  case 0: return a+b;
  case 1: return a-b;
  case 2: return a/b;
  case 3: return a*b;
  default: return 0;
  }
}

int main (int argc, char const *argv[])
{
    double a = 42.0, b = 2.0;
    float f = 42.0, q = 2.0;
    return 0; //// Set break point at this line.
}
