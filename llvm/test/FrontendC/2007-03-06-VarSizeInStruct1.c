// RUN: %llvmgcc %s -w -S -o -
void* p (int n) {
  struct f {
    char w; char x[n]; char z[];
  } F;
  F.x[0]='x';
  return &F;
}
