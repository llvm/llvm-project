// RUN: %llvmgcc -xc %s -S -o /dev/null |& not grep warning

struct item {
  short delta[4];
};

int TEST(int nt) {
 register struct item *aa;
 aa[nt].delta;
 return 1;
}

