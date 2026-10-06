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

// ---------------------------------------------------------------------------
// Expected results, checked by run_tests.sh with FileCheck.
//
// TEST-OPTS: constprop;instcombine
// Every arithmetic instruction disappears: a/a, b/b -> 1, b-b -> 0,
// r/r -> 1, r-r -> 0, then 0 + 23 is folded.
// ONLY-LABEL: define {{.*}} @compute(
// ONLY-NOT:   = {{sdiv|mul|sub|add}} 
// ONLY:       ret i32 23
//
// ALL-LABEL:  define {{.*}} @compute(
// ALL-NEXT:   entry:
// ALL-NEXT:     ret i32 23
