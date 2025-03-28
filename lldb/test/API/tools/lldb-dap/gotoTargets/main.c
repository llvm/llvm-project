
int test_execute_again() {
  int var_3 = 10; // goto 2

  var_3 = 99;

  return var_3; // breakpoint 2
}

int main() {

  int var_1 = 10;

  var_1 = 20; // breakpoint 1

  int var_2 = 40; // goto 1

  int result = test_execute_again();
  return 0;
}
