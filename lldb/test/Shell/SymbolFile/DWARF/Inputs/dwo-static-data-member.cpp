struct NoCtor {
  NoCtor();
  static int i;
};

int NoCtor::i = 15;

int main() { return NoCtor::i; }
