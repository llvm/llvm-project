int main() {
  short the_short = -31987;
  unsigned short the_unsigned_short = 65000;
  int the_int = -1100110;
  unsigned int the_unsigned_int = 4000000000u;

  // Edge-case values: smallest, largest, zero and -1.
  short short_min = -32768;
  short short_max = 32767;
  short short_zero = 0;
  short short_neg_one = -1;
  unsigned short ushort_zero = 0;
  unsigned short ushort_max = 65535;

  int int_min = -2147483647 - 1;
  int int_max = 2147483647;
  int int_zero = 0;
  int int_neg_one = -1;
  unsigned int uint_zero = 0;
  unsigned int uint_max = 4294967295u;

  return 0; // break here
}
