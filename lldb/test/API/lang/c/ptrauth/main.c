#if __aarch64__
#define VALID_DATA_KEY 2
#else
#error Provide these constants if you port this test
#endif

int *__ptrauth(VALID_DATA_KEY) valid0;

typedef int *intp;

int nonConstantGlobal = 5;

int *__ptrauth(VALID_DATA_KEY) valid0;
int *__ptrauth(VALID_DATA_KEY) * valid1;
__ptrauth(VALID_DATA_KEY) intp valid2;
__ptrauth(VALID_DATA_KEY) intp *valid3;
intp __ptrauth(VALID_DATA_KEY) valid4;
intp __ptrauth(VALID_DATA_KEY) * valid5;
int *__ptrauth(VALID_DATA_KEY, 0) valid6;
int *__ptrauth(VALID_DATA_KEY, 1) valid7;
int *__ptrauth(VALID_DATA_KEY, (_Bool)1) valid8;
int *__ptrauth(VALID_DATA_KEY, 1, 0) valid9;
int *__ptrauth(VALID_DATA_KEY, 1, 65535) valid10;

void test_code(intp p) {
  __ptrauth(VALID_DATA_KEY) intp pSpecial = p;
  //% self.expect("fr var pSpecial",
  //%     substrs=['(__ptrauth(2,0,0) intp)', 'pSpecial =', '(actual=0x'])
  //% self.expect("fr var *pSpecial", substrs=['5'])
  //% self.expect("p pSpecial",
  //%     substrs=['(__ptrauth(2,0,0) intp)', '(actual=0x'])
  pSpecial = p;
  intp pNormal = pSpecial;
  pNormal = pSpecial;

  intp __ptrauth(VALID_DATA_KEY) *ppSpecial0 = &pSpecial;
  intp *ppNormal1 = &pNormal;
}

void test_array(void) {
  intp __ptrauth(VALID_DATA_KEY) pSpecialArray[10];
  intp __ptrauth(VALID_DATA_KEY) *ppSpecial0 = pSpecialArray;
  intp __ptrauth(VALID_DATA_KEY) *ppSpecial1 = &pSpecialArray[0];
}

int printf(const char *, ...);

int main(int argc, char **argv) {
  valid0 = &nonConstantGlobal;
  valid1 = &valid0;
  valid2 = &nonConstantGlobal;
  valid3 = &valid2;
  valid4 = &nonConstantGlobal;
  //% self.expect("target var valid0",
  //%     substrs=['(int *__ptrauth(2,0,0))', 'valid0 =', '(actual=0x'])
  //% self.expect("target var valid1",
  //%     substrs=['(int *__ptrauth(2,0,0) *)', 'valid1 ='])
  //% self.expect("target var valid2",
  //%     substrs=['(__ptrauth(2,0,0) intp)', 'valid2 =', '(actual=0x'])
  //% self.expect("target var valid3",
  //%     substrs=['(__ptrauth(2,0,0) intp *)', 'valid3 ='])
  //% self.expect("target var valid4",
  //%     substrs=['(__ptrauth(2,0,0) intp)', 'valid4 =', '(actual=0x'])
  int (*f)(int, char **) = main;
  test_code(valid4);
  test_code(valid4);
  test_array();
  return 0;
}
