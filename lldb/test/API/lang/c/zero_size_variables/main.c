struct Empty {};

__attribute__((noinline)) int take(struct Empty param, int x) {
  struct Empty local = param;
  int zero_len[0];
  (void)local;
  (void)zero_len;
  return x + 1;
}

int main(void) {
  struct Empty empty = {};
  return take(empty, 41);
}
