bool process(bool err1, bool err2) {
  {
    int fd = 0;
    (void)fd;
    if (err1) {
      return false;
    }
    fd = 1;
    (void)fd;
  }
  int result = 42;
  (void)result;
  {
    int fd2 = 0;
    (void)fd2;
    if (err2) {
      return false;
    }
    fd2 = 1;
    (void)fd2;
  }
  return true;
}

int main() {
  process(false, false);
  return 0;
}
