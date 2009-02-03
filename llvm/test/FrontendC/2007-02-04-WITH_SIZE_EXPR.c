// RUN: %llvmgcc %s -O3 -S -o - -emit-llvm
// PR1174

void zzz (char *s1, char *s2, int len, int *q)
{
  int z = 5;
  unsigned int i,  b;
  struct { char a[z]; } x;
          
  for (i = 0; i < len; i++)
    s1[i] = s2[i];

  b = z & 0x3;

  len += (b == 0 ? 0 : 1) + z;
    
  *q = len;

  foo (x, x);
}

