// clang -target x86_64-linux-gnu -shared -fPIC -lc dynamic-table.c \
//       -o dynamic-table-so.x86 -Wl,-f,aux.so -Wl,-F,filter.so
// clang -target mipsel-linux-gnu -shared -fPIC -lc dynamic-table.c \
//       -o dynamic-table-so.mips
// clang -target mipsel-linux-gnu -lc dynamic-table.c \
//       -o dynamic-table-exe.mips
// clang -target aarch64-linux-gnu -fPIC -shared dynamic-table.c\
//       -o dynamic-table-so.aarch64
int puts(const char *);

__thread int foo;

int main(void) {
  puts("Hello, World");
  foo = 0;
}
