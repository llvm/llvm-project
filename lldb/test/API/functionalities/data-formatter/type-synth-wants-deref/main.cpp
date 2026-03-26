struct Wrapper {
  int x;
  int y;
};

int main() {
  Wrapper w{10, 20};
  Wrapper *wp = &w;
  Wrapper &wr = w;
  (void)wp;
  (void)wr;
  return 0;
}
