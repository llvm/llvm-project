// RUN: llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s | FileCheck %s

// This file tests that a synthesized sub-regclass can inherit properties, e.g.,
// tsflags in this case, from the correct super-regclass.

include "llvm/Target/Target.td"

class MyReg<string n, list<Register> subregs = []>
  : Register<n> {
  let Namespace = "Test";
  let SubRegs = subregs;
  let CoveredBySubRegs = 1;
}

class MyClass<int align, list<ValueType> types, dag registers>
  : RegisterClass<"Test", types, align, registers> {
  field bit isA = 0;
  field bit isB = 0;
  let TSFlags{0} = isA;
  let TSFlags{1} = isB;
}

def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;

foreach Num=0-7 in {
  def A#Num : MyReg<"a"#Num>;
}

foreach Num=0-3 in {
  def B#Num : MyReg<"b"#Num>;
}

class AClass<int align, list<ValueType> types, dag registers>
  : MyClass<align, types, registers> {
  let isA = 1;
}

class BClass<int align, list<ValueType> types, dag registers>
  : MyClass<align, types, registers> {
  let isB = 1;
}

def APair : RegisterTuples<[sub0, sub1],
                           [(add A0, A2, A4, A6), (add A1, A3, A5, A7)]>;
def BPair : RegisterTuples<[sub0, sub1],
                           [(add B0, B2), (add B1, B3)]>;
def ARC2 : AClass<32, [untyped], (add APair)>;
def BRC2 : BClass<32, [untyped], (add BPair)>;
def ABRC2 : MyClass<32, [untyped], (add ARC2, BRC2)> {
  let isAllocatable = 0;
}

def ATuple : RegisterTuples<[sub0, sub1, sub2],
                             [(add A0, A1, A2, A3, A4, A5),
                              (add A1, A2, A3, A4, A5, A6),
                              (add A2, A3, A4, A5, A6, A7)]>;

def BTuple : RegisterTuples<[sub0, sub1, sub2],
                             [(add B0, B1), (add B1, B2), (add B2, B3)]>;

def ARC3 : AClass<32, [untyped], (add ATuple)>;
def BRC3 : BClass<32, [untyped], (add BTuple)>;
def ABRC3 : MyClass<32, [untyped], (add ARC3, BRC3)> {
  let isAllocatable = 0;
}

def TestTarget : Target;

// CHECK:      static unsigned const ARC3_with_sub0_sub1Superclasses[] = {
// CHECK-NEXT:   Test::ABRC3RegClassID,
// CHECK-NEXT:   Test::ARC3RegClassID,
// CHECK-NEXT:   Test::ABRC3_with_sub0_sub1RegClassID,
// CHECK-NEXT: };

// CHECK:      static unsigned const ARC3_with_sub1_sub2Superclasses[] = {
// CHECK-NEXT:   Test::ABRC3RegClassID,
// CHECK-NEXT:   Test::ARC3RegClassID,
// CHECK-NEXT:   Test::ABRC3_with_sub1_sub2RegClassID,
// CHECK-NEXT: };

// CHECK: extern const TargetRegisterClass ARC3_with_sub0_sub1RegClass = {
// CHECK:   0x01, /* TSFlags */

// CHECK: extern const TargetRegisterClass ARC3_with_sub1_sub2RegClass = {
// CHECK:   0x01, /* TSFlags */
