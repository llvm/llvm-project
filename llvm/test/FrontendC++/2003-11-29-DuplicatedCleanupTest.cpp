// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null


void doesntThrow() throw();
struct F {
  ~F() { doesntThrow(); }
};

void atest() {
  F A;
lab:
  F B;
  goto lab;
}

void test(int val) {
label: {
   F A;
   F B;
   if (val == 0) goto label;
   if (val == 1) goto label;
}
}

void test3(int val) {
label: {
   F A;
   F B;
   if (val == 0) { doesntThrow(); goto label; }
   if (val == 1) { doesntThrow(); goto label; }
}
}

void test4(int val) {
label: {
   F A;
   F B;
   if (val == 0) { F C; goto label; }
   if (val == 1) { F D; goto label; }
}
}
