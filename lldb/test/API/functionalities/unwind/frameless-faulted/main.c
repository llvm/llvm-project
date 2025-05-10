#if defined(__APPLE__)
int to_be_interrupted(int);
#else
int _to_be_interrupted(int);
#endif

int main() {
  int c = 10;
#if defined(__APPLE__)
  c = to_be_interrupted(c);
#else
  c = _to_be_interrupted(c);
#endif

  return c;
}
