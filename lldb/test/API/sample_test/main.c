#include <stdio.h>

int global_test_var = 10;

int
main()
{
  int test_var = 10;
  printf ("Set a breakpoint here: %d.\n", test_var);
  //% test_var = self.frame().FindVariable("test_var")
  //% test_value = test_var.GetValueAsUnsigned()
  //% self.assertSuccess(test_var.GetError(), "Failed to fetch test_var")
  //% self.assertEqual(test_value, 10, "Failed to get the right value for test_var")
  return global_test_var;
}
