struct Opaque;
struct Opaque *getOpaque();
void puts(const char *);

int main() {
  struct Opaque *x = getOpaque();
  puts("break here\n");
  return (int)x;
}
