// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

//===----------------------------------------------------------------------===//
// Substitution of an int.
def X1;

class C1<int N> {
  dag d = (X1 N);
}

def VAL1 : C1<13>;

// CHECK: def VAL1 {
// CHECK-NEXT: dag d = (X1 13)


//===----------------------------------------------------------------------===//
// Substitution of a DAG.
def X2;

class yclass;
def Y2 : yclass;

class C2<yclass N> {
  dag d = (X2 N);
  dag e = (N X2);
}

def VAL2 : C2<Y2>;

// CHECK: def VAL2 {
// CHECK-NEXT: dag d = (X2 Y2)
// CHECK-NEXT: dag e = (Y2 X2)


//===----------------------------------------------------------------------===//
// Complex dag operator (F.TheOp).

class operator;
def somedef1 : operator;
def somedef2 : operator;

class foo<operator a> {
 operator TheOp = a;
}

class bar<foo F, operator a> {
  dag Dag1 = (somedef1 1);
  dag Dag2 = (a 2);
  dag Dag3 = (F.TheOp 2);
}

def foo1 : foo<somedef1>;
def foo2 : foo<somedef2>;

def VAL3 : bar<foo1, somedef1>;

// CHECK:     def VAL3 {	// bar
// CHECK-NEXT:  dag Dag1 = (somedef1 1);
// CHECK-NEXT:  dag Dag2 = (somedef1 2);
// CHECK-NEXT:  dag Dag3 = (somedef1 2);
// CHECK-NEXT: }


def VAL4 : bar<foo2, somedef2>;
// CHECK:      def VAL4 {
// CHECK-NEXT:  dag Dag1 = (somedef1 1);
// CHECK-NEXT:  dag Dag2 = (somedef2 2);
// CHECK-NEXT:  dag Dag3 = (somedef2 2);
// CHECK-NEXT: }

def VAL5 : bar<foo2, somedef2> {
  // Named operands.
  let Dag1 = (somedef1 1:$name1);

  // Name, no node.
  let Dag2 = (somedef2 $name2, $name3);
}

// CHECK:      def VAL5 {
// CHECK-NEXT:  dag Dag1 = (somedef1 1:$name1);
// CHECK-NEXT:  dag Dag2 = (somedef2 ?:$name2, ?:$name3);
