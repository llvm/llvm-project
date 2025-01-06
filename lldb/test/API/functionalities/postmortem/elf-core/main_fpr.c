static void bar(float *boom) {
  float F = 98.0;
  *boom = 47.0; // Frame bar
}

static void foo(float *boom, void (*boomer)(float *)) {
  float F = 102.0;
  boomer(boom); // Frame foo
}

void _start(void) {
  float F = 95.0;
  foo(0, bar); // Frame _start
}
