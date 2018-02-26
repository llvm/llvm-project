template<typename T>
T pow(T b, T p) {
  if (!p)
    return 1;

  while (--p) {
    b *= b;
  }

  return b;
}
