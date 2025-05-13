int
main(int argc, char**argv)
{
  int x = 2;
  struct Sx {
    int x;
    int& r;
    char y;
  } s{1, x, 2};

  Sx& sr = s;
  Sx* sp = &s;

  Sx sarr[2] = {{5, x, 2}, {1, x, 3}};

  using SxAlias = Sx;
  SxAlias sa{3, x, 4};

  return 0; // Set a breakpoint here
}
