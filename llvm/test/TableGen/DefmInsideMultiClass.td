// RUN: llvm-tblgen %s | grep ADDPSrr | count 1
// XFAIL: vg_leak

class Instruction<bits<4> opc, string Name> {
  bits<4> opcode = opc;
  string name = Name;
}

multiclass basic_r<bits<4> opc> {
  def rr : Instruction<opc, "rr">;
  def rm : Instruction<opc, "rm">;
}

multiclass basic_s<bits<4> opc> {
  defm SS : basic_r<opc>;
  defm SD : basic_r<opc>;
}

multiclass basic_p<bits<4> opc> {
  defm PS : basic_r<opc>;
  defm PD : basic_r<opc>;
}

defm ADD : basic_s<0xf>, basic_p<0xf>;
defm SUB : basic_s<0xe>, basic_p<0xe>;
