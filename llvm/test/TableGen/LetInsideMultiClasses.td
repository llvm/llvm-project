// RUN: llvm-tblgen %s | grep "bit IsDouble = 1;" | count 3
// XFAIL: vg_leak

class Instruction<bits<4> opc, string Name> {
  bits<4> opcode = opc;
  string name = Name;
  bit IsDouble = 0;
}

multiclass basic_r<bits<4> opc> {
  let name = "newname" in {
    def rr : Instruction<opc, "rr">;
    def rm : Instruction<opc, "rm">;
  }

  let name = "othername" in
    def rx : Instruction<opc, "rx">;
}

multiclass basic_ss<bits<4> opc> {
  let IsDouble = 0 in
    defm SS : basic_r<opc>;

  let IsDouble = 1 in
    defm SD : basic_r<opc>;
}

defm ADD : basic_ss<0xf>;

