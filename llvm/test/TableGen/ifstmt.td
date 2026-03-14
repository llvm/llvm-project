// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s

// Test at top level.

// CHECK-NOT: def aNo
// CHECK: def aYes
if 0 then def aNo;
if 1 then def aYes;

// Test inside a foreach, with condition based on the iteration variable.

// CHECK: def bNotThree1
// CHECK: def bNotThree2
// CHECK: def bNotThree4
// CHECK: def bThree3
foreach i = 1...4 in {
  if !eq(i, 3) then {
    def "bThree" # i;
  } else {
    def "bNotThree" # i;
  }
}

// Test inside a multiclass, with condition based on a multiclass parameter.

multiclass Multi<int i> {
  def Unconditional;

  if !eq(i, 2) then
    def Cond;

  if !ge(i, 3) then
    def ThenRec;
  else
    def ElseRec;
}

// CHECK-NOT: def c1Cond
// CHECK: def c1ElseRec
// CHECK-NOT: def c1ThenRec
// CHECK: def c1Unconditional
defm c1: Multi<1>;

// CHECK: def c2Cond
// CHECK: def c2ElseRec
// CHECK-NOT: def c2ThenRec
// CHECK: def c2Unconditional
defm c2: Multi<2>;

// CHECK-NOT: def c3Cond
// CHECK-NOT: def c3ElseRec
// CHECK: def c3ThenRec
// CHECK: def c3Unconditional
defm c3: Multi<3>;

// Test resolution of the dangling-else ambiguity.

// CHECK: def dThenElse00
// CHECK-NOT: def dThenElse1
// CHECK-NOT: def dThenElse11
// CHECK: def dThenThen01
foreach i = 0...1 in
  foreach j = 0...1 in
    if !eq(i,0) then
      if !eq(j,1) then
        def "dThenThen"#i#j;
      else // binds to the inner if, not the outer one
        def "dThenElse"#i#j;

// Error tests: ensure you can't put an if inside a def or class.

#ifdef ERROR1
def baddef {
  int x = 3;
  // ERROR1: [[@LINE+1]]:3: error: Unknown token when expecting a type
  if 1 then {
    int y = 4;
  }
}
#endif

#ifdef ERROR2
class badclass<int i> {
  int x = 3;
  // ERROR2: [[@LINE+1]]:3: error: Unknown token when expecting a type
  if !eq(i, 5) then {
    int y = 4;
  }
}
#endif
