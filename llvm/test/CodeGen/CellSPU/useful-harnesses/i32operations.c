#include <stdio.h>

typedef unsigned int  		uint32_t;
typedef int           		int32_t;

const char *boolstring(int val) {
  return val ? "true" : "false";
}

int i32_eq(int32_t a, int32_t b) {
  return (a == b);
}

int i32_neq(int32_t a, int32_t b) {
  return (a != b);
}

int32_t i32_eq_select(int32_t a, int32_t b, int32_t c, int32_t d) {
  return ((a == b) ? c : d);
}

int32_t i32_neq_select(int32_t a, int32_t b, int32_t c, int32_t d) {
  return ((a != b) ? c : d);
}

struct pred_s {
  const char *name;
  int (*predfunc)(int32_t, int32_t);
  int (*selfunc)(int32_t, int32_t, int32_t, int32_t);
};

struct pred_s preds[] = {
  { "eq",  i32_eq,  i32_eq_select },
  { "neq", i32_neq, i32_neq_select }
};

int main(void) {
  int i;
  int32_t a = 1234567890;
  int32_t b =  345678901;
  int32_t c = 1234500000;
  int32_t d =      10001;
  int32_t e =      10000;

  printf("a = %12d (0x%08x)\n", a, a);
  printf("b = %12d (0x%08x)\n", b, b);
  printf("c = %12d (0x%08x)\n", c, c);
  printf("d = %12d (0x%08x)\n", d, d);
  printf("e = %12d (0x%08x)\n", e, e);
  printf("----------------------------------------\n");

  for (i = 0; i < sizeof(preds)/sizeof(preds[0]); ++i) {
    printf("a %s a = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, a)));
    printf("a %s a = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, a)));
    printf("a %s b = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, b)));
    printf("a %s c = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(a, c)));
    printf("d %s e = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(d, e)));
    printf("e %s e = %s\n", preds[i].name, boolstring((*preds[i].predfunc)(e, e)));

    printf("a %s a ? c : d = %d\n", preds[i].name, (*preds[i].selfunc)(a, a, c, d));
    printf("a %s a ? c : d == c (%s)\n", preds[i].name, boolstring((*preds[i].selfunc)(a, a, c, d) == c));
    printf("a %s b ? c : d = %d\n", preds[i].name, (*preds[i].selfunc)(a, b, c, d));
    printf("a %s b ? c : d == d (%s)\n", preds[i].name, boolstring((*preds[i].selfunc)(a, b, c, d) == d));

    printf("----------------------------------------\n");
  }

  return 0;
}
