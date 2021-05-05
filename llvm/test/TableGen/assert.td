// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

// Test the assert statement at top level.

// CHECK: assertion failed
// CHECK-NOT: note: primary name is too short
// CHECK: note: primary name is too long

defvar Name = "Grace Brewster Murray Hopper";

assert !ge(!size(Name), 20), "primary name is too short: " # Name;
assert !le(!size(Name), 20), "primary name is too long: " # Name;

// CHECK: assertion failed
// CHECK: note: first name is incorrect

def Rec01 {
  string name = "Fred Smith";
}

assert !eq(!substr(Rec01.name, 0, 3), "Jane"),
       !strconcat("first name is incorrect: ", Rec01.name);

// CHECK: assertion failed
// CHECK: note: record Rec02 is broken

def Rec02 {
  bit broken = true;
}

assert !not(Rec02.broken), "record Rec02 is broken";

// CHECK: assertion failed
// CHECK: note: cube of 9

class Cube<int n> {
  int result = !mul(n, n, n);
}

assert !eq(Cube<9>.result, 81), "cube of 9 should be 729";

// CHECK: assertion failed
// CHECK: note: foreach i cannot be 2
// CHECK-NOT: note: foreach i cannot be 2

foreach i = 1...3 in {
  assert !ne(i, 2), "foreach i cannot be 2";
  def bar_ # i;
}

// Test the assert statement in a record definition.

// CHECK: assertion failed
// CHECK-NOT: primary first name is not "Grace"
// CHECK: primary first name is not "Grack"
// CHECK: assertion failed
// CHECK: foo field should be

def Rec10 {
  assert !eq(!substr(Name, 0, 5), "Grace"), "primary first name is not \"Grace\"";
  assert !eq(!substr(Name, 0, 5), "Grack"), "primary first name is not \"Grack\"";
  string foo = "Foo";
  assert !eq(foo, "foo"), "foo field should be \"Foo\"";
}

// CHECK: assertion failed
// CHECK: note: magic field is incorrect: 42

def Rec11 {
  int magic = 13;
  assert !eq(magic, 13), "magic field is incorrect: " # magic;
  let magic = 42;       
}

// CHECK: assertion failed
// CHECK: note: var field has wrong value

def Rec12 {
  defvar prefix = "foo_";
  string var = prefix # "snork";
  assert !eq(var, "foo_snorx"), "var field has wrong value: " # var;
}

// CHECK: assertion failed
// CHECK: note: kind field has wrong value

class Kind {
  int kind = 7;
}

def Rec13 : Kind {
  let kind = 8;
  assert !eq(kind, 7), "kind field has wrong value: " # kind;
}

// CHECK: assertion failed
// CHECK: note: double_result should be

def Rec14 : Cube<3> {
  int double_result = !mul(result, 2);
  assert !eq(double_result, 53), "double_result should be 54";
}

// Test the assert statement in a class definition.

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

// CHECK: assertion failed
// CHECK: note: person name is too long

def Rec21 : Person<"Donald Uh Oh This Name Is Too Long Knuth", 50>;

// CHECK: assertion failed
// CHECK: note: person age is invalid

def Rec22 : Person<"Donald Knuth", 150>;

// Test the assert statement in an anonymous class invocation.

def Rec30 {
  string Name = Person<"Margaret Heafield Hamilton", 25>.Name;
  int Age = Person<"Margaret Heafield Hamilton", 25>.Age;
}

def Rec31 {
  string Name = Person<"Margaret Heafield And More Middle Names Hamilton", 25>.Name;
  int Age = Person<"Margaret Heafield Hamilton", 25>.Age;
}

def Rec32 {
  string Name = Person<"Margaret Heafield Hamilton", 25>.Name;
  int Age = Person<"Margaret Heafield Hamilton", 0>.Age;
}

// Test the assert statement in a multiclass.

// CHECK: assertion failed
// CHECK: note: MC1 id string is too long
// CHECK: assertion failed
// CHECK: note: MC1 seq is too high

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

// CHECK: assertion failed
// CHECK: note: MC2 phrase must be secret: secrex code

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

// CHECK: assertion failed
// CHECK: note: MC2 phrase must be secret: xecret code

multiclass MC4<string phr> : MC2<phr> {
  def _def;
}

defm Rec44 : MC4<"xecret code">;

// Test a defm in a multiclass that inherits from a class with asserts.

// CHECK: assertion failed
// CHECK: note: MC5 name must include a space: Ada_Lovelace
// CHECK: assertion failed
// CHECK: note: person age is invalid: 666

multiclass MC5<string phr, string name, int age> {
  assert !ne(!find(name, " "), -1), "MC5 name must include a space: " # name;

  defm _mc5 : MC2<phr>, Person<name, age>;
}

defm Rec45 : MC5<"secret password", "Ada_Lovelace", 666>;
