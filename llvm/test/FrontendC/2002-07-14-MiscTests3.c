// RUN: %llvmgcc -w -S %s -o - | llvm-as -f -o /dev/null



void *malloc(unsigned);

//#include <stdio.h>
int puts(const char *s);

struct FunStructTest {
  int Test1;
  char *Pointer;
  int Array[12];
};

struct SubStruct {
  short X, Y;
};

struct Quad {
  int w;
  struct SubStruct SS;
  struct SubStruct *SSP;
  char c;
  int y;
};

struct Quad GlobalQuad = { 4, {1, 2}, 0, 3, 156 };

typedef int (*FuncPtr)(int);

unsigned PtrFunc(int (*Func)(int), int X) {
  return Func(X);
}

char PtrFunc2(FuncPtr FuncTab[30], int Num) {
  return FuncTab[Num]('b');
}

extern char SmallArgs2(char w, char x, long long Zrrk, char y, char z);
extern int SomeFunc(void);
char SmallArgs(char w, char x, char y, char z) {
  SomeFunc();
  return SmallArgs2(w-1, x+1, y, z, w);
}

static int F0(struct Quad Q, int i) {              /* Pass Q by value */
  struct Quad R;
  if (i) R.SS = Q.SS;
  Q.SSP = &R.SS;
  Q.w = Q.y = Q.c = 1;
  return Q.SS.Y + i + R.y - Q.c;
}

int F1(struct Quad *Q, int i) {             /* Pass Q by address */
  struct Quad R;
#if 0
  if (i) R.SS = Q->SS;
#else
  if (i) R = *Q;
#endif
  Q->w = Q->y = Q->c = 1;
  return Q->SS.Y+i+R.y-Q->c;
}


int BadFunc(float Val) {
  int Result;
  if (Val > 12.345) Result = 4;
  return Result;     /* Test use of undefined value */
}

int RealFunc(void) {
  return SomeUndefinedFunction(1, 4, 5);
}

extern int EF1(int *, char *, int *);

int Func(int Param, long long Param2) {
  int Result = Param;

  {{{{
      char c; int X;
      EF1(&Result, &c, &X);
    }}}

    {   // c & X are duplicate names!
      char c; int X;
      EF1(&Result, &c, &X);
    }

  }
  return Result;
}


short FunFunc(long long x, char z) {
  return x+z;
}

unsigned castTest(int X) { return X; }

double TestAdd(double X, float Y) {
  return X+Y+.5;
}

int func(int i, int j) {
  while (i != 20)
    i += 2;

  j += func(2, i);
  return (i * 3 + j*2)*j;
}

int SumArray(int Array[], int Num) {
  int i, Result = 0;
  for (i = 0; i < Num; ++i)
    Result += Array[i];

  return Result;
}

int ArrayParam(int Values[100]) {
  return EF1((int*)Values[50], (char*)1, &Values[50]);
}

int ArrayToSum(void) {
  int A[100], i;
  for (i = 0; i < 100; ++i)
    A[i] = i*4;

  return A[A[0]]; //SumArray(A, 100);
}


int ExternFunc(long long, unsigned*, short, unsigned char);

int main(int argc, char *argv[]) {
  unsigned i;
  puts("Hello world!\n");

  ExternFunc(-1, 0, (short)argc, 2);
  //func(argc, argc);

  for (i = 0; i < 10; i++)
    puts(argv[3]);
  return 0;
}

double MathFunc(double X, double Y, double Z,
                double AA, double BB, double CC, double DD,
                double EE, double FF, double GG, double HH,
                double aAA, double aBB, double aCC, double aDD,
                double aEE, double aFF) {
  return X + Y + Z + AA + BB + CC + DD + EE + FF + GG + HH
       + aAA + aBB + aCC + aDD + aEE + aFF;
}



void strcpy(char *s1, char *s2) {
  while (*s1++ = *s2++);
}

void strcat(char *s1, char *s2) {
  while (*s1++);
  s1--;
  while (*s1++ = *s2++);
}

int strcmp(char *s1, char *s2) {
  while (*s1++ == *s2++);
  if (*s1 == 0) {
    if (*s2 == 0) {
      return 0;
    } else {
      return -1;
    }
  } else {
    if (*s2 == 0) {
      return 1;
    } else {
      return (*(--s1) - *(--s2));
    }
  }
}

