// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s

class A;
class B;

def A0 : A;

class C<string name> {
  B b = !cast<B>(name);
}

// CHECK: error: Expected type 'B', got 'A' in: !cast<B>("A0")
def Test : C<"A0">;
