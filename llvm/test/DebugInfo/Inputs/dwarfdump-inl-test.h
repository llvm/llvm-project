inline int inlined_h() {
  volatile int z = 0;
  return z;
}

inline int inlined_g() {
  volatile int y = inlined_h();
  return y;
}
