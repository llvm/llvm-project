int
main(int argc, char **argv)
{
  int a = 1;
  int b = 2;

  char c = -3;
  unsigned short s = 4;

  return 0; // Set a breakpoint here
}

/*
TEST_F(EvalTest, TestLocalVariables) {
  EXPECT_THAT(Eval("a"), IsEqual("1"));
  EXPECT_THAT(Eval("b"), IsEqual("2"));
  EXPECT_THAT(Eval("a + b"), IsEqual("3"));

  EXPECT_THAT(Eval("c + 1"), IsEqual("-2"));
  EXPECT_THAT(Eval("s + 1"), IsEqual("5"));
  EXPECT_THAT(Eval("c + s"), IsEqual("1"));

  EXPECT_THAT(Eval("__test_non_variable + 1"),
              IsError("use of undeclared identifier '__test_non_variable'"));
}
*/
