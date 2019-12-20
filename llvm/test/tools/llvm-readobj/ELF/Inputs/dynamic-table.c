// clang -target mipsel-linux-gnu -shared -fPIC -lc dynamic-table.c \
//       -o dynamic-table-so.mips
// clang -target mipsel-linux-gnu -lc dynamic-table.c \
//       -o dynamic-table-exe.mips
int puts(const char *);

__thread int foo;

int main(void) {
  puts("Hello, World");
  foo = 0;
}
