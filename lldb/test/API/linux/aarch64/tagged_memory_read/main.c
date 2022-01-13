#include <stddef.h>

static char *set_non_address_bits(char *ptr, size_t tag) {
  // Set top byte tag (AArch64 Linux always enables top byte ignore)
  return (char *)((size_t)ptr | (tag << 56));
}

int main(int argc, char const *argv[]) {
  char buf[32];

  char *ptr1 = set_non_address_bits(buf, 0x34);
  char *ptr2 = set_non_address_bits(buf, 0x56);

  return 0; // Set break point at this line.
}
