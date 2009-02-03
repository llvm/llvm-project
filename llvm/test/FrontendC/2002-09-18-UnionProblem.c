// RUN: %llvmgcc -S %s -o - | llvm-as -f -o /dev/null


struct DWstruct {
  char high, low;
};

typedef union {
  struct DWstruct s;
  short ll;
} DWunion;

short __udivmodhi4 (char n1, char bm) {
  DWunion rr;

  if (bm == 0)
    {
      rr.s.high = n1;
    }
  else
    {
      rr.s.high = bm;
    }

  return rr.ll;
}
