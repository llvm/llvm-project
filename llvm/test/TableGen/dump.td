// RUN: llvm-tblgen %s -o -  2>&1 >/dev/null | FileCheck %s -DFILE=%s

// CHECK: [[FILE]]:[[@LINE+1]]:1: note: Debug message
dump "Debug message";

def op;
class A {
  string A = "some text";
  dag X =(op op);
}
def a : A;
// CHECK: [[FILE]]:[[@LINE+5]]:1: note: The Value of A is:
// CHECK-NEXT: a {     // A
// CHECK-NEXT: string A = "some text";
// CHECK-NEXT: dag X = (op op);
// CHECK-NEXT: }
dump "The Value of A is: \n" # !repr(a);

def b : A;
// CHECK: [[FILE]]:[[@LINE+4]]:1: note: b {     // A
// CHECK-NEXT: string A = "some text";
// CHECK-NEXT: dag X = (op op);
// CHECK-NEXT: }
dump !repr(b) ;

defvar value_A = "some other text";
defvar value_B = 12;
def X;
// CHECK: [[FILE]]:[[@LINE+3]]:1: note: got a pair of values ["some other text" : 12], and an empty record:
// CHECK-NEXT: X {
// CHECK-NEXT: }
dump "got a pair of values [" # !repr(value_A) # " : " # !repr(value_B) # "], " # "and an empty record:\n" # !repr(X);

multiclass MC<dag s> {
// CHECK: [[FILE]]:[[@LINE+1]]:3: note: s = (op a)
  dump "s = " # !repr(s);
// CHECK: [[FILE]]:[[@LINE+4]]:3: note: args[0] = a {        // A
// CHECK-NEXT:   string A = "some text";
// CHECK-NEXT: dag X = (op op);
// CHECK-NEXT: }
  dump "args[0] = " # !repr(!getdagarg<A>(s,0));
  def A;
}
defm X : MC<(op a)>;

foreach i = [-1, 2] in {
// CHECK: [[FILE]]:[[@LINE+4]]:3: note: i = -1 (negative)
// CHECK: [[FILE]]:[[@LINE+8]]:5: note: i + 1  <= 0
// CHECK: [[FILE]]:[[@LINE+2]]:3: note: i = 2 (positive)
// CHECK: [[FILE]]:[[@LINE+4]]:5: note: i + 1  > 0 (i + 1 = 3)
  dump "i = " # !repr(i) # !if(!ge(i,0), " (positive)", " (negative)");
  defvar ip1 = !add(i, 1);
  if !gt(ip1,0) then {
    dump "i + 1 > 0 (i + 1 = " # !repr(ip1) # ")";
  } else {
    dump "i + 1 <= 0" ;
  }
}

class Code<code val> {
  dump "val = " # !repr(val);
  code Val = val;
  int number = 0;
}
// CHECK: [[FILE]]:[[@LINE-4]]:3: note: val = [{a = a +1;}]
def IncrementA : Code<[{a = a +1;}]>;
class InheritFromCode : Code<[{f(x);}]>{
  let number = 33;
  dump "number = " # !repr(number);
}
// CHECK: [[FILE]]:[[@LINE-10]]:3: note: val = [{f(x);}]
// CHECK: [[FILE]]:[[@LINE-3]]:3: note: number = 33
def ModeCode : InheritFromCode;


class BaseClassForSet;
multiclass DefineSubSet {
  def _One : BaseClassForSet;
  def _Two : BaseClassForSet;
}
defset list<BaseClassForSet> TheSet = {
defm Subset: DefineSubSet;
def Three : BaseClassForSet;
}
// CHECK: [[FILE]]:[[@LINE+1]]:1: note: TheSet = [Subset_One, Subset_Two, Three]
dump "TheSet = " # !repr(TheSet);
