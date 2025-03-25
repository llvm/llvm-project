class VSetT<bits<5> func5, dag outs, dag ins, string opcodestr, string argstr>
  : RVInst<outs, ins, opcodestr, argstr, [], InstFormatOther> {
  bits<5> rs1;
  bits<5> rd;

  let Inst{31}    = 1;
  let Inst{30-25} = 0b000010;
  let Inst{24-20} = func5;
  let Inst{19-15} = rs1;
  let Inst{14-12} = 0b111;
  let Inst{11-7}  = rd;
  let Inst{6-0}   = 0x57;
}

let Predicates = [HasStdExtZvma] in {
let hasSideEffects = 1, mayLoad = 0, mayStore = 0 in {
  def VSETTN : VSetT<0b00000, (outs GPR:$rd), (ins GPR:$rs1), "vsettn", "$rd, %rs1">;
  def VSETTM : VSetT<0b00001, (outs GPR:$rd), (ins GPR:$rs1), "vsettm", "$rd, %rs1">;
  def VSETTK : VSetT<0b00010, (outs GPR:$rd), (ins GPR:$rs1), "vsettk", "$rd, %rs1">;
} // hasSideEffects = 1, mayLoad = 0, mayStore = 0
} // Predicates = [HasStdExtZvma]
