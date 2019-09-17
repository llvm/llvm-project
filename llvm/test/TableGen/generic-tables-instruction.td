// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s
// XFAIL: vg_leak

include "llvm/TableGen/SearchableTable.td"

// CHECK-LABEL: GET_InstrTable_IMPL
// CHECK: constexpr MyInstr InstrTable[] = {
// CHECK:   { B, 0xA },
// CHECK:   { C, 0x0 },
// CHECK:   { A, 0x5 },
// CHECK:   { D, 0x8 },
// CHECK: };

class Instruction {
  bit isPseudo = 0;
}

class MyInstr<int op> : Instruction {
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
