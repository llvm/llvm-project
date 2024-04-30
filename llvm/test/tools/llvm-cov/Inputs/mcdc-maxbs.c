#define RANGE(a,b,c) ((a) <= (b) && (b) <= (c))

int sub(int c) {
  if (RANGE('0', c, '9')) return 1;
  return (('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z'));
}

extern void __llvm_profile_write_file(void);

int main(int c, char **v)
{
  return (c > 1 ? sub(c) : 0);
}
