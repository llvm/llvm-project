// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s
// RUN: not llvm-tblgen -DERROR4 %s 2>&1 | FileCheck --check-prefix=ERROR4 %s
// RUN: not llvm-tblgen -DERROR5 %s 2>&1 | FileCheck --check-prefix=ERROR5 %s
// RUN: not llvm-tblgen -DERROR6 %s 2>&1 | FileCheck --check-prefix=ERROR6 %s
// RUN: not llvm-tblgen -DERROR7 %s 2>&1 | FileCheck --check-prefix=ERROR7 %s
// RUN: not llvm-tblgen -DERROR8 %s 2>&1 | FileCheck --check-prefix=ERROR8 %s
// RUN: not llvm-tblgen -DERROR9 %s 2>&1 | FileCheck --check-prefix=ERROR9 %s
// RUN: not llvm-tblgen -DERROR10 %s 2>&1 | FileCheck --check-prefix=ERROR10 %s

// !setop and !getop are deprecated in favor of !setdagop and !getdagop.
// Two tests retain the old names just to be sure they are still supported.

class Base;
class OtherBase;

class Super : Base;

def foo: Base;
def bar: Base;
def qux: OtherBase;

def alice : Super;
def bob : Super;

def test {
  dag orig = (foo 1, 2:$a, $b);
  dag another = (qux "hello", $world);

  // CHECK: dag replaceWithBar = (bar 1, 2:$a, ?:$b);
  dag replaceWithBar = !setop(orig, bar);

  // CHECK: dag replaceWithBaz = (qux 1, 2:$a, ?:$b);
  dag replaceWithBaz = !setdagop(orig, qux);

  // CHECK: Base getopWithCast = foo;
  Base getopWithCast = !getop<Base>(orig);

  // CHECK: dag getopToSetop = (foo "hello", ?:$world);
  dag getopToSetop = !setdagop(another, !getdagop(orig));

  // CHECK: dag getopToBangDag = (foo 1:$a, 2:$b, 3:$c);
  dag getopToBangDag = !dag(!getdagop(orig), [1, 2, 3], ["a", "b", "c"]);

  // CHECK: dag getopToDagInit = (foo "it worked");
  dag getopToDagInit = (!getdagop(orig) "it worked");

#ifdef ERROR1
  // !getdagop(...) has a static type of 'any record at all, with no
  // required superclasses'. That's too general to use in an
  // assignment whose LHS demands an instance of Base, so we expect a
  // static (parse-time) type-checking error.

  // ERROR1: error: Field 'noCast' of type 'Base' is incompatible with value '!getdagop(orig)' of type '{}'
  Base noCast = !getdagop(orig);
#endif

#ifdef ERROR2
  // Here, we expect a _dynamic_ type error, when it turns out at
  // evaluation time that the operator of 'another' is a record that
  // isn't an instance of the specified base class.

  // ERROR2: error: Expected type 'Base', got 'OtherBase' in: !getdagop((qux "hello", ?:$world))
  Base badCast = !getdagop<Base>(another);
#endif

#ifdef ERROR3
  // Obviously, you shouldn't be able to give any type to !getdagop that
  // isn't a class type.

  // ERROR3: error: type for !getdagop must be a record type
  int ridiculousCast = !getdagop<int>(orig);
#endif

  dag in1 = (foo 1:$a, 2:$b, 3:$c);
  // CHECK: list<string> in1Names = ["a", "b", "c"];
  list<string> in1Names = !foreach(i, !range(!size(in1)), !getdagname(in1, i));
  // CHECK: list<int> in1Args = [1, 2, 3];
  list<int> in1Args = !foreach(i, !range(!size(in1)), !getdagarg<int>(in1, i));

  dag in2 = (foo 1:$a, (bar "x":$x, (qux foo:$s1, bar:$s2):$y, 7:$z):$b, 3:$c);
  // CHECK: dag in2NestedDag = (qux foo:$s1, bar:$s2);
  dag in2NestedDag = !getdagarg<dag>(!getdagarg<dag>(in2, 1), "y");
  // CHECK: Base in2NestedArg = foo;
  Base in2NestedArg = !getdagarg<Base>(!getdagarg<dag>(!getdagarg<dag>(in2, 1), "y"), "s1");

  dag in3 = (foo 1:$a, ?:$b, 3);
  // CHECK: list<string> in3Names = ["a", "b", ?];
  list<string> in3Names = !foreach(i, !range(!size(in3)), !getdagname(in3, i));
  // CHECK: list<int> in3Args = [1, ?, 3];
  list<int> in3Args = !foreach(i, !range(!size(in3)), !getdagarg<int>(in3, i));

#ifdef ERROR4
  // ERROR4: error: !getdagarg index -1 is negative
  int outOfRange = !getdagarg<int>(in1, -1);
#endif

#ifdef ERROR5
  // ERROR5: error: !getdagarg index 3 is out of range (dag has 3 arguments)
  int outOfRange = !getdagarg<int>(in1, 3);
#endif

#ifdef ERROR6
  // ERROR6: error: !getdagarg key 'x' is not found
  int notFound = !getdagarg<int>(in1, "x");
#endif

  dag in4 = (foo "arg1":$a, "arg2":$b, "arg3":$c);
  // CHECK: int misMatchType1 = ?;
  int misMatchType1 = !getdagarg<int>(in4, 0);

  dag in5 = (foo foo:$a, bar:$b, foo:$c);
  // CHECK: OtherBase misMatchType2 = ?;
  OtherBase misMatchType2 = !getdagarg<OtherBase>(in5, 1);

  dag in6 = (foo alice:$a, bob:$b);
  // CHECK: Base base = bob;
  Base base = !getdagarg<Base>(in6, 1);

  // CHECK: dag orig_set_val = (foo 1, 2:$a, "val":$b);
  dag orig_set_val = !setdagarg(orig, 2, "val");
  // CHECK: dag orig_set_val_by_name = (foo 1, 2:$a, "aval":$b);
  dag orig_set_val_by_name = !setdagarg(orig, "b", "aval");
  // CHECK: dag orig_set_dag_val = (foo 1, 2:$a, (bar foo:$p, qux:$q):$b);
  dag orig_set_dag_val = !setdagarg(orig, "b", (bar foo:$p, qux:$q));
  // CHECK: dag orig_clr_val = (foo 1, ?:$a, ?:$b);
  dag orig_clr_val = !setdagarg(orig, "a", ?);
  // CHECK: dag orig_set_name = (foo 1:$c, 2:$a, ?:$b);
  dag orig_set_name = !setdagname(orig, 0, "c");
  // CHECK: dag orig_clr_name = (foo 1, 2, ?:$b);
  dag orig_clr_name = !setdagname(orig, 1, ?);
  // CHECK: dag orig_rename = (foo 1, 2:$x, ?:$y);
  dag orig_rename = !setdagname(!setdagname(orig, "a", "x"), "b", "y");

#ifdef ERROR7
  // ERROR7: error: !setdagarg index -1 is negative
  dag orig_negative = !setdagarg(orig, -1, "val");
#endif

#ifdef ERROR8
  // ERROR8: error: !setdagarg index 3 is out of range (dag has 3 arguments)
  dag orig_out_of_range = !setdagarg(orig, 3, "val");
#endif

#ifdef ERROR9
  // ERROR9: error: expected integer index or string name, got type 'Base'
  dag orig_out_of_range = !setdagarg(orig, foo, (foo qux:$a));
#endif
}

// Copy a list (Predicates) that is a field in a dag operator
// (TestInstruction), which is defined in the same multiclass
// (TestInstructionAndPattern) as the destination of the copy
// (TestPattern::Predicates).
class TestInstruction<list<int> _Predicates> {
  list<int> Predicates = _Predicates;
}
#ifdef ERROR10
class OtherTestInstruction<list<int> _Predicates> {
  list<int> Predicates = _Predicates;
}
// ERROR10: error: Expected type 'OtherTestInstruction', got 'TestInstruction'
class TestPattern<dag D> {
  list<int> Predicates = !getdagop<OtherTestInstruction>(D).Predicates;
}
#else
class TestPattern<dag D> {
  list<int> Predicates = !getdagop<TestInstruction>(D).Predicates;
}
#endif

multiclass TestInstructionAndPattern<list<int> Predicates> {
  def NAME : TestInstruction<Predicates>;
  def : TestPattern<(!cast<TestInstruction>(NAME) foo)>;
}
// CHECK: def testInst0 { // TestInstruction
// CHECK-NEXT: list<int> Predicates = [7];
defm testInst0 : TestInstructionAndPattern<[7]>;
