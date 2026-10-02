#include <stdio.h>

void function_prologue_outlined(void);
void function_body_outlined(void);
void function_epilogue_outlined(void);

void call_external_functions() {
  puts("there");
  function_prologue_outlined();
  function_body_outlined();
  function_epilogue_outlined();
  puts("done");
}

void sub_main_function() {
  puts("in subfunc");
  call_external_functions();
}

int main() {
  puts("HI");
  sub_main_function();
  puts("exiting");
  return 0;
}
