extern int consume(int);

void foo(void) {
  consume(1);
  consume(2);
}

void bar(void) {
  consume(3);
}

__attribute__((section("a")))
void buz(void) {
  consume(4);
}

__attribute__((section("b")))
void quux(void) {
  consume(5);
}
