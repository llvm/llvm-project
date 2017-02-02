static volatile int do_mul;
static volatile int do_inc;

int main () {
  int x = 1;
  if (do_mul) x *= 2; else x /= 2;
  return do_inc ? ++x : --x;
}
