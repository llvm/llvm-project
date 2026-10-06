//Algebraic identity examples
int compute (int a, int b)
{
  int result = (a/a); // result = 1

  result *= (b/b); // result = 1 * 1
  result += (b-b); // result = 1 + 0
  result /= result; // result = 1 / 1
  result -= result; // result = 1 - 1
  result += 23; //constant folding
  return result;
}

// Expected (checked by run_tests.sh): no arithmetic left.
// CHECK-NOT: sdiv
// CHECK:     ret i32 23
