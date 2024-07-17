struct Structure {
  int number;
};

struct Opaque;
int puts(const char *s);

int main(int argc, char **argv) {
  struct Structure s;
  s.number = 30;
  struct Opaque *opaque = &s;
  puts("break here");
  return 0;
}
