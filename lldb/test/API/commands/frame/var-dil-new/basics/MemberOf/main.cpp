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

/*
  EXPECT_THAT(Eval("s.x"), IsEqual("1"));
  EXPECT_THAT(Eval("s.r"), IsEqual("2"));
  EXPECT_THAT(Eval("s.r + 1"), IsEqual("3"));
  EXPECT_THAT(Eval("sr.x"), IsEqual("1"));
  EXPECT_THAT(Eval("sr.r"), IsEqual("2"));
  EXPECT_THAT(Eval("sr.r + 1"), IsEqual("3"));
  EXPECT_THAT(Eval("sp->x"), IsEqual("1"));
  EXPECT_THAT(Eval("sp->r"), IsEqual("2"));
  EXPECT_THAT(Eval("sp->r + 1"), IsEqual("3"));
  EXPECT_THAT(Eval("sarr->x"), IsEqual("5"));
  EXPECT_THAT(Eval("sarr->r"), IsEqual("2"));
  EXPECT_THAT(Eval("sarr->r + 1"), IsEqual("3"));
  EXPECT_THAT(Eval("(sarr + 1)->x"), IsEqual("1"));

  EXPECT_THAT(
      Eval("sp->4"),
      IsError(
          "<expr>:1:5: expected 'identifier', got: <'4' (numeric_constant)>\n"
          "sp->4\n"
          "    ^"));
  EXPECT_THAT(Eval("sp->foo"), IsError("no member named 'foo' in 'Sx'"));
  EXPECT_THAT(
      Eval("sp->r / (void*)0"),
      IsError("invalid operands to binary expression ('int' and 'void *')"));

  EXPECT_THAT(Eval("sp.x"), IsError("member reference type 'Sx *' is a "
                                    "pointer; did you mean to use '->'"));
  EXPECT_THAT(
      Eval("sarr.x"),
      IsError(
          "member reference base type 'Sx[2]' is not a structure or union"));

  // Test for record typedefs.
  EXPECT_THAT(Eval("sa.x"), IsEqual("3"));
  EXPECT_THAT(Eval("sa.y"), IsEqual("'\\x04'"));

*/
