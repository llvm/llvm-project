// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s

#ifdef ERROR1
// Refer to a variable we haven't defined *yet*, expecting an error.
// ERROR1: [[@LINE+1]]:22: error: Variable not defined: 'myvar'
def bad { dag x = (? myvar); }
#endif

// Define a global variable.
defvar myvar = "foo";

#ifdef ERROR2
// Demonstrate an error when a global variable is redefined.
// ERROR2: [[@LINE+1]]:8: error: def or global variable of this name already exists
defvar myvar = "another value";
#endif

// These variables should be overrided by template arguments.
defvar a = 2333;
defvar b = 2333;
class VarScopeTest<int a, int b> {
  defvar c = !add(a, b);
  int value = !add(c, c);
}

// CHECK: def aaa_scope_test {
// CHECK-NEXT: int value = 10;
def aaa_scope_test: VarScopeTest<2, 3>;

multiclass Test<int x> {
  // Refer to a global variable, while inside a local scope like a multiclass.
  def _with_global_string { string s = myvar; }

  // Define some variables local to this multiclass, and prove we can refer to
  // those too.
  defvar myvar = !add(x, 100);
  defvar myvar2 = "string of " # myvar;
  def _with_local_int { int i = myvar; string s = myvar2; }

#ifdef ERROR3
  // Demonstrate an error when a local variable is redefined.
  // ERROR3: [[@LINE+1]]:10: error: local variable of this name already exists
  defvar myvar = "another value";
#endif
}

// Instantiate the above multiclass, and expect all the right outputs.

// CHECK: def aaa_with_global_string {
// CHECK-NEXT: string s = "foo";
// CHECK: def aaa_with_local_int {
// CHECK-NEXT: int i = 101;
// CHECK-NEXT: string s = "string of 101";
// CHECK: def bbb_with_global_string {
// CHECK-NEXT: string s = "foo";
// CHECK: def bbb_with_local_int {
// CHECK-NEXT: int i = 102;
// CHECK-NEXT: string s = "string of 102";
defm aaa: Test<1>;
defm bbb: Test<2>;

// Test that local variables can be defined inside a foreach block, and inside
// an object body.
//
// The scopes nest (you can refer to variables in an outer block from an inner
// one), and the variables go out of scope again at the end of the block (in
// particular, you don't get a redefinition error the next time round the
// loop).

// CHECK: def nest_f1_s3 {
// CHECK-NEXT: int member = 113;
// CHECK-NEXT: }
// CHECK: def nest_f1_s4 {
// CHECK-NEXT: int member = 114;
// CHECK-NEXT: }
// CHECK: def nest_f2_s3 {
// CHECK-NEXT: int member = 123;
// CHECK-NEXT: }
// CHECK: def nest_f2_s4 {
// CHECK-NEXT: int member = 124;
// CHECK-NEXT: }
foreach first = [ 1, 2 ] in {
  defvar firstStr = "f" # first;
  foreach second = [ 3, 4 ] in {
    defvar secondStr = "s" # second;
    def "nest_" # firstStr # "_" # secondStr {
      defvar defLocalVariable = !add(!mul(first, 10), second);
      int member = !add(100, defLocalVariable);
    }
  }
}
defvar firstStr = "now define this at the top level and still expect no error";

// Test that you can shadow an outer declaration with an inner one. Here, we
// expect all the shadowOuter records (both above and below the inner foreach)
// to get the value 1 from the outer definition of shadowedVariable, and the
// shadowInner ones to get 2 from the inner definition.

// CHECK: def shadowInner11 {
// CHECK-NEXT: int var = 2;
// CHECK: def shadowInner12 {
// CHECK-NEXT: int var = 2;
// CHECK: def shadowInner21 {
// CHECK-NEXT: int var = 2;
// CHECK: def shadowInner22 {
// CHECK-NEXT: int var = 2;
// CHECK: def shadowInnerIf1 {
// CHECK-NEXT: int var = 3;
// CHECK: def shadowOuterAbove1 {
// CHECK-NEXT: int var = 1;
// CHECK: def shadowOuterAbove2 {
// CHECK-NEXT: int var = 1;
// CHECK: def shadowOuterBelowForeach1 {
// CHECK-NEXT: int var = 1;
// CHECK: def shadowOuterBelowForeach2 {
// CHECK-NEXT: int var = 1;
// CHECK: def shadowOuterBelowIf1 {
// CHECK-NEXT: int var = 1;
// CHECK: def shadowOuterBelowIf2 {
// CHECK-NEXT: int var = 1;

foreach first = [ 1, 2 ] in {
  defvar shadowedVariable = 1;
  def shadowOuterAbove # first { int var = shadowedVariable; }

  // The foreach statement opens a new scope, in which a new variable of the
  // same name can be defined without clashing with the outer one.
  foreach second = [ 1, 2 ] in {
    defvar shadowedVariable = 2;
    def shadowInner # first # second { int var = shadowedVariable; }
  }

  // Now the outer variable is back in scope.
  def shadowOuterBelowForeach # first { int var = shadowedVariable; }

  // An if statement also opens a new scope.
  if !eq(first, 1) then {
    defvar shadowedVariable = 3;
    def shadowInnerIf # first { int var = shadowedVariable; }
  }

  // Now the outer variable is back in scope again.
  def shadowOuterBelowIf # first { int var = shadowedVariable; }
}

// Test that a top-level let statement also makes a variable scope (on the
// general principle of consistency, because it defines a braced sub-block).

let someVariable = "some value" in {
  defvar myvar = "override the definition from above and expect no error";
}
// CHECK: def topLevelLetTest {
// CHECK-NEXT: string val = "foo";
def topLevelLetTest { string val = myvar; }
