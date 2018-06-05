// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: WorldHelloCC
// CHECK-NOT: WorldHelloCC

class C<string n> {
  string name = n;
}

multiclass Names<string n, string m> {
   def CC : C<n>;
   def World#NAME#CC : C<m>;
}

defm Hello : Names<"hello", "world">;

// Ensure that the same anonymous name is used as the prefix for all defs in an
// anonymous multiclass.

class Outer<C i> {
  C Inner = i;
}

multiclass MC<string name> {
  def hi : C<name>;
  def there : Outer<!cast<C>(!strconcat(NAME, "hi"))>;
}

defm : MC<"foo">;

multiclass MC2<string name> {
  def there : Outer<C<name> >;
}

// Ensure that we've correctly captured the reference to name from the implicit
// anonymous C def in the template parameter list of Outer.
// CHECK-NOT: MC2::name

defm : MC2<"bar">;

multiclass MC3<string s> {
  def ZFizz#s : C<s>;
}

defm "" : MC3<"Buzz">;

// CHECK: def ZFizzBuzz
// CHECK: string name = "Buzz";
// CHECK-NOT: MC3::s

multiclass MC4<string s> {
  def NAME#s : C<s>;
}

defm ZTagazok : MC4<"AToi">;

// CHECK: def ZTagazokAToi
// CHECK: string name = "AToi";
// CHECK-NOT: MC4::s

multiclass MC5<C c> {
  def NAME#c.name : C<c.name>;
}

def CTiger : C<"Tiger">;
defm Zebra : MC5<CTiger>;

// CHECK: def ZebraTiger
// CHECK: string name = "Tiger";
// CHECK-NOT: MC5::c

multiclass MC6<C c> {
  def NAME#Tiger#c.name : C<c.name>;
}

def CAligator : C<"Aligator">;
defm Zebra : MC6<CAligator>;

// CHECK: def ZebraTigerAligator
// CHECK: string name = "Aligator";
// CHECK-NOT: MC6::c

