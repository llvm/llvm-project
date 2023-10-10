// RUN: llvm-tblgen %s -o -  2>&1 >/dev/null | FileCheck %s

// CHECK-LABEL: Debug message
dump "Debug message";

// CHECK-LABEL: The Value of A is:
// CHECK-NEXT: a {     // A
// CHECK-NEXT: string A = "some text";
// CHECK-NEXT: dag X = (op op);
// CHECK-NEXT: }
def op;
class A {
  string A = "some text";
  dag X =(op op);
}
def a : A;
dump "The Value of A is: \n" # !repr(a);


// CHECK-LABEL: b {     // A
// CHECK-NEXT: string A = "some text";
// CHECK-NEXT: dag X = (op op);
// CHECK-NEXT: }
def b : A;
dump !repr(b) ;

// CHECK-LABEL: got a pair of values ["some other text" : 12], and an empty record:
// CHECK-NEXT: X {
// CHECK-NEXT: }
defvar value_A = "some other text";
defvar value_B = 12;
def X;
dump "got a pair of values [" # !repr(value_A) # " : " # !repr(value_B) # "], " # "and an empty record:\n" # !repr(X);

multiclass MC<dag s> {
  dump "s = " # !repr(s);
  dump "args[0] = " # !repr(!getdagarg<A>(s,0));
  def A;
}
// CHECK-LABEL: note: s = (op a)
// CHECK-NEXT: args[0] = a {        // A
// CHECK-NEXT:   string A = "some text";
// CHECK-NEXT: dag X = (op op);
// CHECK-NEXT: }
defm X : MC<(op a)>;

