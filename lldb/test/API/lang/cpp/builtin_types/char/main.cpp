int main() {
  char the_char = 'a';
  signed char the_signed_char = 'B';
  unsigned char the_unsigned_char = 'Z';

  // Edge-cases.
  char char_zero = 0;
  char char_neg_one = -1;
  char char_high_bit = (char)0x80;
  char char_low_max = (char)0x7f;

  signed char schar_neg_one = -1;
  signed char schar_min = -128;
  signed char schar_max = 127;
  unsigned char uchar_zero = 0;
  unsigned char uchar_max = 255;

  return 0; // break here
}
