// RUN: %llvmgcc %s -S -o -
char p (int n) {
  struct f {
    char w; char x[n]; char y[n];
  } F;

  return F.x[0];
}
