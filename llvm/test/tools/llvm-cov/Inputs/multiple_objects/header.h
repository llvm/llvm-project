static inline void f1() {
#ifdef DEF
  if (false && false)
    return;

  if (true || false || true)
    return;

  if (true && false)
    return;
#endif
}

template<typename T>
void f2(T **x) {
#ifdef DEF
  if (false && false)
    *x = nullptr;

  if (true || false || true)
    *x = nullptr;

  if (true && false)
    *x = nullptr;
#endif
}

static inline void f3() {
}
