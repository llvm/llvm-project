extern int call_through_indirect(int);

int
fake_call_through_reexport(int value)
{
  return value + 10;
}
