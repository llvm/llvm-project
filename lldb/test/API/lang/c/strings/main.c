int main()
{
  const char a[] = "abcde";
  const char *z = "vwxyz";

  return *a + *z; // breakpoint 1
}
