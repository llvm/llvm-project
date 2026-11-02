// The width of 'long' depends on the data model. Expose its size to the test
// as a global so it can select the right expectations from the target itself
// instead of matching architecture names.
unsigned long_size = sizeof(long);

int main() {
  long the_long = -1100110100l;
  unsigned long the_unsigned_long = 1100110100ul;
  long long the_long_long = -110011001100ll;
  unsigned long long the_unsigned_long_long = 110011001100ull;

  // Use a size-independent way to compute min/max of long which has a size that
  // depends on the data model.
  long long_zero = 0;
  long long_neg_one = -1;
  unsigned long ulong_zero = 0;
  long long_max = (long)(~0ul >> 1);
  long long_min = -long_max - 1;
  unsigned long ulong_max = ~0ul;

  long long llong_min = -9223372036854775807ll - 1;
  long long llong_max = 9223372036854775807ll;
  long long llong_zero = 0;
  long long llong_neg_one = -1;
  unsigned long long ullong_zero = 0;
  unsigned long long ullong_max = 18446744073709551615ull;

  return 0; // break here
}
