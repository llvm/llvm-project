unsigned long long mid(const unsigned char *in, unsigned long len);
long top_data = 1;

unsigned long long top(const unsigned char *in, unsigned long len) {
  return mid(in, len) + top_data++;
}
