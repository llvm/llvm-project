// RUN: %llvmgcc -S -fnested-functions -O0 -o - -emit-llvm %s 
// PR915

extern void abort(void);

void nest(int n)
{
  int a = 0;
  int b = 5;
  int c = 0;
  int d = 7;

  void o(int i, int j)
  {
    if (i!=j)
      abort();
  }

  void f(x)
    int x; /* K&R style */
  {
    int e = 0;
    int f = 2;
    int g = 0;

    void y(void)
    {
      c = n;
      e = 1;
      g = x;
    }

    void z(void)
    {
      a = 4;
      g = 3;
    }

    a = 5;
    y();
    c = x;
    z();
    o(1,e);
    o(2,f);
    o(3,g);
  }

  c = 2;
  f(6);
  o(4,a);
  o(5,b);
  o(6,c);
  o(7,d);
}
