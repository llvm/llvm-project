int frame3() { return 3; }

int frame2() { return frame3(); }

int frame1() { return frame2(); }

int main() {
  int result = 0;
  for (int i = 0; i < 25; ++i)
    result += frame1();
  return result;
}
