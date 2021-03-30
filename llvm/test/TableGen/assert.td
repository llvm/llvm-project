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

