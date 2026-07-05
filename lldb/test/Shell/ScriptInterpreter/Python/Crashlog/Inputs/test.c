void foo() {
  int *i = 0;
  *i = 1;
}

void bar() { foo(); }

int main(int argc, char **argv) { bar(); }
