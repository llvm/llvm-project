static volatile int num = 1;

bool hw_break_function (int i) {
  return num == i;
}

int main (int argc, char const *argv[]) {
  return hw_break_function(argc) ? 0 : 1;
}
