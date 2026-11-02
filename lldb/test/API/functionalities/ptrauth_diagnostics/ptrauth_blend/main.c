#define auth __ptrauth(2, 1, 9484)

struct mytype {
  int *auth p;
};

int main() {
  int x = 123;
  struct mytype s;
  s.p = &x;
  struct mytype *s_ptr = &s;
  return 0; // break here
}
