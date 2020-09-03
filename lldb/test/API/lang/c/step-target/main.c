void
lotsOfArgs
(
  int firstArg,
  int secondArg,
  int thirdArg,
  int fourthArg
)
{
  int x = firstArg + secondArg + thirdArg + fourthArg;
}

int
modifyInt(int incoming)
{
  return incoming % 2;
}

int
main (int argc, char **argv)
{
  if (argc > 0)
    {
      int var_makes_block = argc + 1;
      int dummy = 0; // Break here to try targetted stepping.
      lotsOfArgs(var_makes_block,
                 modifyInt(20),
                 30,
                 modifyInt(40));
      int abc = 0; // Done calling lotsOfArgs.
    }
  return 0; // All done.
}
