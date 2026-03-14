static void bar(char *boom) {
  char F = 'b';
  *boom = 47; // Frame bar
}

static void foo(char *boom, void (*boomer)(char *)) {
  char F = 'f';
  boomer(boom); // Frame foo
}

void main(void) {
  char F = 'm';
  foo(0, bar); // Frame main
}
