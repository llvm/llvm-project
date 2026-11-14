unsigned long long xxh3(const unsigned char *in, unsigned long len);
void loclist(void);

unsigned long long mid(const unsigned char *in, unsigned long len) {
  loclist();
  return xxh3(in, len);
}
