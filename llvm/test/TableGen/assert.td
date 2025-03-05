// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s -DFILE=%s

// -----------------------------------------------------------------------------
// Test the assert statement at top level.
// -----------------------------------------------------------------------------

// CHECK: [[FILE]]:[[@LINE+4]]:8: error: assertion failed: primary name is too long
// CHECK-NOT: note: primary name is too short
defvar Name = "Grace Brewster Murray Hopper";
assert !ge(!size(Name), 20), "primary name is too short: " # Name;
assert !le(!size(Name), 20), "primary name is too long: " # Name;

// CHECK: assertion failed: first name is incorrect
def Rec01 {
  string name = "Fred Smith";
}

assert !eq(!substr(Rec01.name, 0, 3), "Jane"),
       !strconcat("first name is incorrect: ", Rec01.name);

// CHECK: assertion failed: record Rec02 is broken
def Rec02 {
  bit broken = true;
}
assert !not(Rec02.broken), "record Rec02 is broken";

// CHECK: assertion failed: cube of 9 should be 729
class Cube<int n> {
  int result = !mul(n, n, n);
}
assert !eq(Cube<9>.result, 81), "cube of 9 should be 729";

// Verify that the assert fails only once.
// CHECK: assertion failed: foreach i cannot be 2
// CHECK-NOT: assertion failed: foreach i cannot be 2
foreach i = 1...3 in {
  assert !ne(i, 2), "foreach i cannot be 2";
  def bar_ # i;
}

// -----------------------------------------------------------------------------
// Test the assert statement in a record definition.
// -----------------------------------------------------------------------------

// CHECK: [[FILE]]:[[@LINE+6]]:10: error: assertion failed: primary first name is not "Grack"
// CHECK-NOT: primary first name is not "Grace"
// CHECK: [[FILE]]:[[@LINE+6]]:10: error: assertion failed: foo field should be
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec10 {
  assert !eq(!substr(Name, 0, 5), "Grace"), "primary first name is not \"Grace\"";
  assert !eq(!substr(Name, 0, 5), "Grack"), "primary first name is not \"Grack\"";
  string foo = "Foo";
  assert !eq(foo, "foo"), "foo field should be \"Foo\"";
}

// CHECK: [[FILE]]:[[@LINE+4]]:10: error: assertion failed: magic field is incorrect: 42
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec11 {
  int magic = 13;
  assert !eq(magic, 13), "magic field is incorrect: " # magic;
  let magic = 42;       
}

// CHECK: [[FILE]]:[[@LINE+5]]:10: error: assertion failed: var field has wrong value
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec12 {
  defvar prefix = "foo_";
  string var = prefix # "snork";
  assert !eq(var, "foo_snorx"), "var field has wrong value: " # var;
}

// CHECK: assertion failed: kind field has wrong value
class Kind {
  int kind = 7;
}

// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec13 : Kind {
  let kind = 8;
  assert !eq(kind, 7), "kind field has wrong value: " # kind;
}

// CHECK: assertion failed: double_result should be
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec14 : Cube<3> {
  int double_result = !mul(result, 2);
  assert !eq(double_result, 53), "double_result should be 54";
}

// -----------------------------------------------------------------------------
// Test the assert statement in a class definition.
// -----------------------------------------------------------------------------

class PersonName<string name> {
  assert !le(!size(name), 32), "person name is too long: " # name;
  string Name = name;
}

class Person<string name, int age> : PersonName<name> {
  assert !and(!ge(age, 1), !le(age, 120)),
         "person age is invalid: " # age;
  int Age = age;
}

def Rec20 : Person<"Donald Knuth", 60>;

// CHECK: assertion failed: person name is too long
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec21 : Person<"Donald Uh Oh This Name Is Too Long Knuth", 50>;

// CHECK: assertion failed: person age is invalid
// CHECK: [[FILE]]:[[@LINE+1]]:5: error: assertion failed in this record
def Rec22 : Person<"Donald Knuth", 150>;

// Test the assert statement in an anonymous class invocation.
def Rec30 {
  string Name = Person<"Margaret Heafield Hamilton", 25>.Name;
  int Age = Person<"Margaret Heafield Hamilton", 25>.Age;
}

// CHECK: assertion failed: person name is too long
// CHECK: [[FILE]]:[[@LINE+2]]:17: error: assertion failed in this record
def Rec31 {
  string Name = Person<"Margaret Heafield And More Middle Names Hamilton", 25>.Name;
  int Age = Person<"Margaret Heafield Hamilton", 25>.Age;
}

// CHECK: assertion failed: person age is invalid: 0
// CHECK: [[FILE]]:[[@LINE+3]]:13: error: assertion failed in this record
def Rec32 {
  string Name = Person<"Margaret Heafield Hamilton", 25>.Name;
  int Age = Person<"Margaret Heafield Hamilton", 0>.Age;
}

// -----------------------------------------------------------------------------
// Test the assert statement in a multiclass.
// -----------------------------------------------------------------------------

// CHECK: assertion failed: MC1 id string is too long
// CHECK: assertion failed: MC1 seq is too high
multiclass MC1<string id, int seq> {
  assert !le(!size(id), 5), "MC1 id string is too long";
  assert !le(seq, 999999), "MC1 seq is too high";

  def _mc1 {
    string ID = id;
    int Seq = seq;
  }
}

defm Rec40 : MC1<"ILISP", 999>;
defm Rec41 : MC1<"ILISPX", 999>;
defm Rec42 : MC1<"ILISP", 999999999>;

// CHECK: assertion failed: MC2 phrase must be secret: secrex code
multiclass MC2<string phr> {
  assert !eq(!substr(phr, 0, 6), "secret"), "MC2 phrase must be secret: " # phr;

  def _mc2 {
    string phrase = phr;
  }
}

multiclass MC3<string phr> {
  defm _mc3 : MC2<phr>;
}

defm Rec43 : MC3<"secrex code">;

// CHECK: assertion failed: MC2 phrase must be secret: xecret code
multiclass MC4<string phr> : MC2<phr> {
  def _def;
}

defm Rec44 : MC4<"xecret code">;

// Test a defm in a multiclass that inherits from a class with asserts.

// CHECK: assertion failed: MC5 name must include a space: Ada_Lovelace
// CHECK: assertion failed: person age is invalid: 666
multiclass MC5<string phr, string name, int age> {
  assert !ne(!find(name, " "), -1), "MC5 name must include a space: " # name;

  defm _mc5 : MC2<phr>, Person<name, age>;
}

defm Rec45 : MC5<"secret password", "Ada_Lovelace", 666>;
