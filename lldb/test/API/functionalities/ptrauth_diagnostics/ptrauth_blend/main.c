#define auth __ptrauth(2, 1, 9484)

struct mytype {
  int *auth p;
};

int main() {
  int x = 123;
  struct mytype s;
  s.p = &x;
  struct mytype *s_ptr = &s;
  //% self.filecheck("expr -j 0 -- *s_ptr->p", "main.c")
  // CHECK: (int) {{.*}} = 123
  return 0;
}
