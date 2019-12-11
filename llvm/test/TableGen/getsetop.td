// RUN: llvm-tblgen %s | FileCheck %s
// RUN: not llvm-tblgen -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// RUN: not llvm-tblgen -DERROR2 %s 2>&1 | FileCheck --check-prefix=ERROR2 %s
// RUN: not llvm-tblgen -DERROR3 %s 2>&1 | FileCheck --check-prefix=ERROR3 %s

class Base;
class OtherBase;

def foo: Base;
def bar: Base;
def qux: OtherBase;

def test {
  dag orig = (foo 1, 2:$a, $b);
  dag another = (qux "hello", $world);

  // CHECK: dag replaceWithBar = (bar 1, 2:$a, ?:$b);
  dag replaceWithBar = !setop(orig, bar);

  // CHECK: dag replaceWithBaz = (qux 1, 2:$a, ?:$b);
  dag replaceWithBaz = !setop(orig, qux);

  // CHECK: Base getopWithCast = foo;
  Base getopWithCast = !getop<Base>(orig);

  // CHECK: dag getopToSetop = (foo "hello", ?:$world);
  dag getopToSetop = !setop(another, !getop(orig));

  // CHECK: dag getopToBangDag = (foo 1:$a, 2:$b, 3:$c);
  dag getopToBangDag = !dag(!getop(orig), [1, 2, 3], ["a", "b", "c"]);

  // CHECK: dag getopToDagInit = (foo "it worked");
  dag getopToDagInit = (!getop(orig) "it worked");

#ifdef ERROR1
  // !getop(...) has a static type of 'any record at all, with no
  // required superclasses'. That's too general to use in an
  // assignment whose LHS demands an instance of Base, so we expect a
  // static (parse-time) type-checking error.

  // ERROR1: error: Value 'noCast' of type 'Base' is incompatible with initializer '!getop(orig)' of type '{}'
  Base noCast = !getop(orig);
#endif

#ifdef ERROR2
  // Here, we expect a _dynamic_ type error, when it turns out at
  // evaluation time that the operator of 'another' is a record that
  // isn't an instance of the specified base class.

  // ERROR2: error: Expected type 'Base', got 'OtherBase' in: !getop((qux "hello", ?:$world))
  Base badCast = !getop<Base>(another);
#endif

#ifdef ERROR3
  // Obviously, you shouldn't be able to give any type to !getop that
  // isn't a class type.

  // ERROR3: error: type for !getop must be a record type
  int ridiculousCast = !getop<int>(orig);
#endif
}
