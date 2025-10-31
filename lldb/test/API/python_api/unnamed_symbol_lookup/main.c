__attribute__((nodebug)) int stripped_function(int val) { return val * val; }

int main(void) {
  stripped_function(10);
  return 0;
}
