// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A0 {
// CHECK:   dag ret = (ops);
// CHECK: }

// CHECK: def A1 {
// CHECK:   dag ret = (ops ?:$a, 1:$b, 2);
// CHECK: }

// CHECK: def A2 {
// CHECK:   dag ret = (ops (ops ?:$name):$a, (ops 1):$b, (ops "foo"):$c);
// CHECK: }

// CHECK: def A3 {
// CHECK:   dag ret = (ops NodeA0:$a, NodeB0:$b);
// CHECK: }

// CHECK: def A4 {
// CHECK:   dag ret = (ops NodeA0, NodeB0);
// CHECK: }

// CHECK: def B0 {
// CHECK:   dag ret = (ops);
// CHECK: }

// CHECK: def B1 {
// CHECK:   dag ret = (ops 1:$a, 2:$b);
// CHECK: }

// CHECK: def C0 {
// CHECK:   dag ret1 = (ops ?:$a, ?:$b);
// CHECK:   dag ret2 = (ops 1, 2);
// CHECK: }

// CHECK: def D {
// CHECK:   dag d1 = (ops 1, ?:$name1, 2, 3);
// CHECK: }

// CHECK: def E0 {
// CHECK:   dag ret = (ops 1, 2);
// CHECK: }

class Ops;

def ops : Ops;

class Node<int val, string name> {
  int Val = val;
  string Name = name;
}

class Aint<list<int> nodes, list<string> names> {
  dag ret = !dag(ops, nodes, names);
}

class Adag<list<dag> nodes, list<string> names> {
  dag ret = !dag(ops, nodes, names);
}

class NodeBase;

class NodeA<int val> : NodeBase {
  int x = val;
}

class NodeB<int val> : NodeBase {
  int y = val;
}

class Anode<list<NodeBase> nodes, list<string> names> {
  dag ret = !dag(ops, nodes, names);
}

class B<list<Node> nodes> {
  dag ret = !foldl((ops), nodes, lhs, rhs, !con(lhs, !dag(ops, [rhs.Val], [rhs.Name])));
}

def A0 : Aint<[], []>;
def A1 : Aint<[?, 1, 2], ["a", "b", ?]>;

def A2 : Adag<[(ops $name), (ops 1), (ops "foo")], ["a", "b", "c"]>;

def NodeA0 : NodeA<0>;
def NodeB0 : NodeB<0>;

def A3 : Anode<[NodeA0, NodeB0], ["a", "b"]>;

def A4 {
  // Like A3, but with a literal list directly in the !dag.
  dag ret = !dag(ops, [NodeA0, NodeB0], ?);
}

def B0 : B<[]>;
def B1 : B<[Node<1, "a">, Node<2, "b">]>;

class C<list<int> nodes, list<string> names> {
  dag ret1 = !dag(ops, ?, names);
  dag ret2 = !dag(ops, nodes, ?);
}

def C0 : C<[1, 2], ["a", "b"]>;

def D {
  dag d1 = !con((ops 1), (ops $name1), (ops), (ops 2, 3));
}

class E<Ops op> {
  // Allow concatenation of DAG nodes with operators from template arguments.
  dag ret = !con((op 1), (op 2));
}

def E0 : E<ops>;
