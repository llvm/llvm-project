// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s -o %t.inc
// RUN: FileCheck %s < %tTargetDesc.inc

include "llvm/Target/Target.td"

def TestTarget : Target;

def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;
def sub2 : SubRegIndex<32, 64>;
def sub3 : SubRegIndex<32, 96>;
def sub4 : SubRegIndex<32, 128>;
def sub5 : SubRegIndex<32, 160>;
def sub6 : SubRegIndex<32, 192>;
def sub7 : SubRegIndex<32, 224>;

def R0 : Register<"">;
def R1 : Register<"">;
def R2 : Register<"">;
def R3 : Register<"">;
def R4 : Register<"">;
def R5 : Register<"">;
def R6 : Register<"">;
def R7 : Register<"">;

def Super : Register<""> {
  let SubRegs = [R0, R1, R2, R3, R4, R5, R6, R7];
  let SubRegIndices = [sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7];
}

def LeafRC
    : RegisterClass<"Test", [i32], 32, (add R0, R1, R2, R3, R4, R5, R6, R7)>;
foreach I = 0 -31 in def ExtraLeafRC#I
    : RegisterClass<"Test", [i32], 32, (add R0, R1, R2, R3, R4, R5, R6, R7)>;
def SuperRC : RegisterClass<"Test", [v8i32], 256, (add Super)>;

// CHECK-LABEL: getSubClassWithSubReg(
// CHECK: static constexpr uint8_t BlockMap[17][1] = {
// CHECK: static_assert(sizeof(BlockMap) == 17);
// CHECK: static constexpr uint8_t Blocks[2][16] = {
// CHECK-NEXT: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
// CHECK-NEXT: { 0, 0, 0, 0, 0, 0, 0, 0, 34, 34, 34, 34, 34, 34, 34, 34 },
// CHECK: static_assert(sizeof(Blocks) == 32);
// CHECK: unsigned BlockID = BlockMap[RCID / 2][Idx / 8];
// CHECK: unsigned TV = Blocks[BlockID][(RCID % 2) * 8 + Idx % 8];

// CHECK-LABEL: getSubRegisterClass(
// CHECK: static constexpr uint8_t BlockMap[17][1] = {
// CHECK: static_assert(sizeof(BlockMap) == 17);
// CHECK: static constexpr uint8_t Blocks[2][16] = {
// CHECK-NEXT: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
// CHECK-NEXT: { 0, 0, 0, 0, 0, 0, 0, 0, 33, 33, 33, 33, 33, 33, 33, 33 },
// CHECK: static_assert(sizeof(Blocks) == 32);
// CHECK: unsigned BlockID = BlockMap[RCID / 2][Idx / 8];
// CHECK: unsigned TV = Blocks[BlockID][(RCID % 2) * 8 + Idx % 8];
