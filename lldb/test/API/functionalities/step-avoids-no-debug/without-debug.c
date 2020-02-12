typedef int (*debug_callee) (int);

int
no_debug_caller_intermediate(int input, debug_callee callee)
{
  int return_value = 0;
  return_value = callee(input);
  return return_value;
}

int
no_debug_caller (int input, debug_callee callee)
{
  int return_value = 0;
  return_value = no_debug_caller_intermediate (input, callee);
  return return_value;
}
