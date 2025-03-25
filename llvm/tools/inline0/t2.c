int foo(int a, int b);

int goo(int c, int d) {
  int s= 0;
  for(int i = c; i < d; ++i)
    s+=foo(i, d);
  for(int i = c*s; i < d*s*3; ++i)
    s+=i*3;
  return s;
}
