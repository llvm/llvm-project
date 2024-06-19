// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s
// XFAIL: vg_leak

include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }
def Arch : Target { let InstructionSet = ArchInstrInfo; }

// CHECK-LABEL: GET_InstrTable_IMPL
// CHECK: constexpr MyInstr InstrTable[] = {
// CHECK:   { B, 0xA },
// CHECK:   { C, 0x0 },
// CHECK:   { A, 0x5 },
// CHECK:   { D, 0x8 },
// CHECK: };

// A contiguous primary (Instruction) key should get a direct lookup instead of
// binary search.
// CHECK: const MyInstr *getCustomEncodingHelper(unsigned Opcode) {
// CHECK:   if ((Opcode < B) ||
// CHECK:       (Opcode > D))
// CHECK:     return nullptr;
// CHECK:   auto Table = ArrayRef(InstrTable);
// CHECK:   size_t Idx = Opcode - B;
// CHECK:   return &Table[Idx];


class MyInstr<int op> : Instruction {
  let OutOperandList = (outs);
  let InOperandList = (ins);
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<16> CustomEncoding = op;
}

def A : MyInstr<5>;
def D : MyInstr<8>;
let isPseudo = 1 in {
  def C : MyInstr<0>;
  def B : MyInstr<10>;
}

def InstrTable : GenericTable {
  let FilterClass = "MyInstr";
  let Fields = ["Opcode", "CustomEncoding"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getCustomEncodingHelper";
}


// Non-contiguous instructions should get a binary search instead of direct
// lookup.
// CHECK: const MyInfoEntry *getTable2ByOpcode(unsigned Opcode) {
// CHECK:   auto Idx = std::lower_bound(Table.begin(), Table.end(), Key,
//
// Verify contiguous check for SearchIndex.
// const MyInfoEntry *getTable2ByValue(uint8_t Value) {
// CHECK:   if ((Value < 0xB) ||
// CHECK:      (Value > 0xD))
// CHECK:    return nullptr;
// CHECK:  auto Table = ArrayRef(Index);
// CHECK:  size_t Idx = Value - 0xB;
// CHECK:  return &InstrTable2[Table[Idx]._index];


class MyInfoEntry<int V, string S> {
  Instruction Opcode = !cast<Instruction>(NAME);
  bits<4> Value = V;
  string Name = S;
}

let OutOperandList = (outs), InOperandList = (ins) in {
def W : Instruction, MyInfoEntry<12, "IW">;
def X : Instruction;
def Y : Instruction, MyInfoEntry<13, "IY">;
def Z : Instruction, MyInfoEntry<11, "IZ">;
}

def InstrTable2 : GenericTable {
  let FilterClass = "MyInfoEntry";
  let Fields = ["Opcode", "Value", "Name"];

  let PrimaryKey = ["Opcode"];
  let PrimaryKeyName = "getTable2ByOpcode";
}

def getTable2ByValue : SearchIndex {
  let Table = InstrTable2;
  let Key = ["Value"];
}
