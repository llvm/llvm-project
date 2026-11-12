// RUN: llvm-tblgen %s | FileCheck %s

class AClass<bit C> {
  bit Cond = C;
}

def A0: AClass<0>;
def A1: AClass<1>;

class BoolToList<bit Value> {
  list<int> ret = !if(Value, [1]<int>, []<int>);
}

multiclass P<string AStr> {
  foreach i = BoolToList<!cast<AClass>(AStr).Cond>.ret in
    def SubDef;
}

// CHECK-NOT: def XSubDef
defm X : P<"A0">;

// CHECK: def YSubDef
defm Y : P<"A1">;
