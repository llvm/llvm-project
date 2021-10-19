// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

// Checks the cost values for the register tuple.
include "llvm/Target/Target.td"

class MyClass<int size, list<ValueType> types, dag registers>
  : RegisterClass<"MyTarget", types, size, registers> {
  let Size = size;
}

class Indexes<int N> {
  list<int> all = [0,  1,  2,  3];
  list<int> slice =
    !foldl([]<int>, all, acc, cur,
           !listconcat(acc, !if(!lt(cur, N), [cur], [])));
}

foreach Index = 0-3 in {
  def sub#Index : SubRegIndex<32, !shl(Index, 5)>;
}

foreach Size = {2,4} in {
  foreach Index = Indexes<!add(5, !mul(Size, -1))>.slice in {
    def !foldl("", Indexes<Size>.slice, acc, cur,
               !strconcat(acc#!if(!eq(acc,""),"","_"), "sub"#!add(cur, Index))) :
      SubRegIndex<!mul(Size, 32), !shl(Index, 5)> {
      let CoveringSubRegIndices =
        !foldl([]<SubRegIndex>, Indexes<Size>.slice, acc, cur,
               !listconcat(acc, [!cast<SubRegIndex>(sub#!add(cur, Index))]));
    }
  }
}

let Namespace = "MyTarget" in {
  foreach Index = 0-15 in {
    // Adding two cost values per register.
    let CostPerUse = [Index, !shl(Index, 1)] in {
      def S#Index : Register <"s"#Index>;
    }
  }
} // Namespace = "MyTarget"

def GPR32 : MyClass<32,  [i32], (sequence "S%u", 0, 15)>;

def GPR64 : RegisterTuples<[sub0, sub1],
                           [(decimate (shl GPR32, 0), 1),
                            (decimate (shl GPR32, 1), 1)
                           ]>;

def GPR128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                            [
                             (decimate (shl GPR32, 0), 1),
                             (decimate (shl GPR32, 1), 1),
                             (decimate (shl GPR32, 2), 1),
                             (decimate (shl GPR32, 3), 1)
                            ]>;


def GPR_64 : MyClass<64, [v2i32], (add GPR64)>;
def GPR_128 : MyClass<128, [v4i32], (add GPR128)>;


def MyTarget : Target;

// CHECK: static const uint8_t CostPerUseTable[] = {
// CHECK-NEXT:  0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, };

// CHECK:  static const TargetRegisterInfoDesc MyTargetRegInfoDesc = { // Extra Descriptors
// CHECK-NEXT:  CostPerUseTable, 2, InAllocatableClassTable};

// CHECK:  TargetRegisterInfo(&MyTargetRegInfoDesc, RegisterClasses, RegisterClasses+3,
