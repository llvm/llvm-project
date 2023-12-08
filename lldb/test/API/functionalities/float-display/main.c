float f_neg3 = 1.234567 / 1e3;
float f_neg4 = 1.234567 / 1e4;
float f_neg5 = 1.234567 / 1e5;
float f_neg6 = 1.234567 / 1e6;
float f_neg7 = 1.234567 / 1e7;
float f_neg8 = 1.234567 / 1e8;
float f_neg20 = 1.234567 / 1e20;
float f_neg30 = 1.234567 / 1e30;

float f_3 = 1.234567 * 1e3;
float f_4 = 1.234567 * 1e4;
float f_5 = 1.234567 * 1e5;
float f_6 = 1.234567 * 1e6;
float f_7 = 1.234567 * 1e7;
float f_8 = 1.234567 * 1e8;
float f_20 = 1.234567 * 1e20;
float f_30 = 1.234567 * 1e30;

double d_neg3 = 1.234567 / 1e3;
double d_neg4 = 1.234567 / 1e4;
double d_neg5 = 1.234567 / 1e5;
double d_neg6 = 1.234567 / 1e6;
double d_neg7 = 1.234567 / 1e7;
double d_neg8 = 1.234567 / 1e8;
double d_neg20 = 1.234567 / 1e20;
double d_neg30 = 1.234567 / 1e30;
double d_neg50 = 1.234567 / 1e50;
double d_neg250 = 1.234567 / 1e250;

double d_3 = 1.234567 * 1e3;
double d_4 = 1.234567 * 1e4;
double d_5 = 1.234567 * 1e5;
double d_6 = 1.234567 * 1e6;
double d_7 = 1.234567 * 1e7;
double d_8 = 1.234567 * 1e8;
double d_20 = 1.234567 * 1e20;
double d_30 = 1.234567 * 1e30;
double d_50 = 1.234567 * 1e50;
double d_250 = 1.234567 * 1e250;

int main (int argc, char const *argv[]) {
  //% # Default setting should be 6.
  //% self.expect("frame variable f_neg3", substrs=["0.00123456"])
  //% self.expect("frame variable f_neg4", substrs=["0.000123456"])
  //% self.expect("frame variable f_neg5", substrs=["0.0000123456"])
  //% self.expect("frame variable f_neg6", substrs=["0.00000123456"])
  //% self.expect("frame variable f_neg7", substrs=["1.234567", "E-7"])
  //% self.expect("frame variable f_neg8", substrs=["1.23456", "E-8"])
  //% self.expect("frame variable f_neg20", substrs=["E-20"])
  //% self.expect("frame variable f_neg30", substrs=["E-30"])
  //% self.expect("frame variable f_3", substrs=["1234.56"])
  //% self.expect("frame variable f_4", substrs=["12345.6"])
  //% self.expect("frame variable f_5", substrs=["123456"])
  //% self.expect("frame variable f_6", substrs=["123456"])
  //% self.expect("frame variable f_7", substrs=["123456"])
  //% self.expect("frame variable f_8", substrs=["123456"])
  //% self.expect("frame variable f_20", substrs=["E+20"])
  //% self.expect("frame variable f_30", substrs=["E+30"])
  //% self.expect("frame variable d_neg3", substrs=["0.00123456"])
  //% self.expect("frame variable d_neg4", substrs=["0.000123456"])
  //% self.expect("frame variable d_neg5", substrs=["0.0000123456"])
  //% self.expect("frame variable d_neg6", substrs=["0.00000123456"])
  //% self.expect("frame variable d_neg7", substrs=["1.23456", "E-7"])
  //% self.expect("frame variable d_neg8", substrs=["1.23456", "E-8"])
  //% self.expect("frame variable d_neg20", substrs=["1.23456", "E-20"])
  //% self.expect("frame variable d_neg30", substrs=["1.23456", "E-30"])
  //% self.expect("frame variable d_neg50", substrs=["1.23456", "E-50"])
  //% self.expect("frame variable d_neg250", substrs=["E-250"])
  //% self.expect("frame variable d_3", substrs=["1234.56"])
  //% self.expect("frame variable d_4", substrs=["12345.6"])
  //% self.expect("frame variable d_5", substrs=["123456"])
  //% self.expect("frame variable d_6", substrs=["1234567"])
  //% self.expect("frame variable d_7", substrs=["1234567"])
  //% self.expect("frame variable d_8", substrs=["1234567"])
  //% self.expect("frame variable d_20", substrs=["1.23456", "E+20"])
  //% self.expect("frame variable d_30", substrs=["1.23456", "E+30"])
  //% self.expect("frame variable d_50", substrs=["1.23456", "E+50"])
  //% self.expect("frame variable d_250", substrs=["1.23456", "E+250"])
  //% # Now change the setting to print all the zeroes.
  //% # Note that changing this setting should invalidate the data visualizer
  //% # cache so that the new setting is used in the following calls.
  //% self.runCmd("settings set target.max-zero-padding-in-float-format 9999")
  //% self.expect("frame variable  f_neg3", substrs=["0.00123456"])
  //% self.expect("frame variable  f_neg4", substrs=["0.000123456"])
  //% self.expect("frame variable  f_neg5", substrs=["0.0000123456"])
  //% self.expect("frame variable  f_neg6", substrs=["0.00000123456"])
  //% self.expect("frame variable  f_neg7", substrs=["0.000000123456"])
  //% self.expect("frame variable  f_neg8", substrs=["0.0000000123456"])
  //% self.expect("frame variable  f_neg20", substrs=["0.0000000000000000000123456"])
  //% self.expect("frame variable  f_neg30", substrs=["0.00000000000000000000000000000123456"])
  //% self.expect("frame variable  f_3", substrs=["1234.56"])
  //% self.expect("frame variable  f_4", substrs=["12345.6"])
  //% self.expect("frame variable  f_5", substrs=["123456"])
  //% self.expect("frame variable  f_6", substrs=["1234567"])
  //% self.expect("frame variable  f_7", substrs=["1234567"])
  //% self.expect("frame variable  f_8", substrs=["1234567"])
  //% self.expect("frame variable  f_20", substrs=["E+20"])
  //% self.expect("frame variable  f_30", substrs=["E+30"])
  //% self.expect("frame variable  d_neg3", substrs=["0.00123456"])
  //% self.expect("frame variable  d_neg4", substrs=["0.000123456"])
  //% self.expect("frame variable  d_neg5", substrs=["0.0000123456"])
  //% self.expect("frame variable  d_neg6", substrs=["0.00000123456"])
  //% self.expect("frame variable  d_neg7", substrs=["0.000000123456"])
  //% self.expect("frame variable  d_neg8", substrs=["0.0000000123456"])
  //% self.expect("frame variable  d_neg20", substrs=["0.0000000000000000000123456"])
  //% self.expect("frame variable  d_neg30", substrs=["0.000000000000000000000000000001234567"])
  //% self.expect("frame variable  d_neg50", substrs=["0.0000000000000000000000000000000000000000000000000123456"])
  //% self.expect("frame variable  d_neg250", substrs=["0.000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000123456"])
  //% self.expect("frame variable  d_3", substrs=["1234.56"])
  //% self.expect("frame variable  d_4", substrs=["12345.6"])
  //% self.expect("frame variable  d_5", substrs=["123456"])
  //% self.expect("frame variable  d_6", substrs=["1234567"])
  //% self.expect("frame variable  d_7", substrs=["1234567"])
  //% self.expect("frame variable  d_8", substrs=["1234567"])
  //% # Positive numbers are not affected by this setting.
  //% self.expect("frame variable  d_20", substrs=["1.23456", "E+20"])
  //% self.expect("frame variable  d_30", substrs=["1.23456", "E+30"])
  //% self.expect("frame variable  d_50", substrs=["1.23456", "E+50"])
  //% self.expect("frame variable  d_250", substrs=["1.23456", "E+250"])
  return 0;
}
