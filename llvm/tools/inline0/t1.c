#ifdef ALWAYS_INLINE 
__attribute__((always_inline))
#endif
int foo(int a, int b) {
  int s= 0;
  for(int i = a; i < b; ++i)
    s+=i;
  return s*3;
}
