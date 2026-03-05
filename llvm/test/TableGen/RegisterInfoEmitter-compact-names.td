// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

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

def GPR96 : RegisterTuples<[sub0, sub1, sub2],
                           [(decimate (shl GPR32, 0), 1),
                            (decimate (shl GPR32, 1), 1),
                            (decimate (shl GPR32, 2), 1)
                           ]>;

def GPR96_Alt : RegisterTuples<[sub0, sub1, sub2],
                           [(decimate (shl GPR32, 0), 1),
                            (decimate (shl GPR32, 1), 1),
                            (decimate (shl GPR32, 3), 1)
                           ]>;

def GPR128 : RegisterTuples<[sub0, sub1, sub2, sub3],
                            [
                             (decimate (shl GPR32, 0), 1),
                             (decimate (shl GPR32, 1), 1),
                             (decimate (shl GPR32, 2), 1),
                             (decimate (shl GPR32, 3), 1)
                            ]>;

def GPR128_Evens : RegisterTuples<[sub0, sub1, sub2, sub3],
                            [
                             (decimate (shl GPR32, 0), 1),
                             (decimate (shl GPR32, 2), 1),
                             (decimate (shl GPR32, 4), 1),
                             (decimate (shl GPR32, 6), 1)
                            ]>;


def GPR_64 : MyClass<64, [v2i32], (add GPR64)>;
def GPR_96: MyClass<64, [v3i32], (add GPR96)>;
def GPR_96_Alt: MyClass<64, [v3i32], (add GPR96_Alt)>;
def GPR_128 : MyClass<128, [v4i32], (add GPR128)>;
def GPR_128_Evens : MyClass<128, [v4i32], (add GPR128_Evens)>;


def MyTarget : Target {
  let CompactRegisterNames = true;
}

// CHECK-LABEL: NoRegister
// CHECK: S0_to_1
// CHECK: S1_to_2
// CHECK: S0_S1_S3
// CHECK: S1_S2_S4
// CHECK: S0_to_3
// CHECK: S1_to_4
// CHECK: S0_to_6_by_2
// CHECK: S1_to_7_by_2

// CHECK-LABEL: NoSubRegister
// CHECK: sub0_to_1
// CHECK: sub0_sub1_sub3
// CHECK: sub1_to_2
// CHECK: sub2_to_3
