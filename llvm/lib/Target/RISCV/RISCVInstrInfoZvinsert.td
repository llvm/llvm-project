//===- RISCVInstrInfoZvinsert.td - RISC-V 'Zvinsert' instructions ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the RISC-V V instructions from the standard 'Zvinsert',
// Vector Extension for Moves between Scalars and Vector Elements, version 0.94.
//
//===----------------------------------------------------------------------===//

class RVStdExtZvinsertInst<
  dag outs, dag ins, string opcodestr, string argstr> :
  RVInst<outs, ins, opcodestr, argstr, [], InstFormatR> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let DecoderNamespace = "RVZvinsert";
  let Predicates = [HasStdExtZvinsert];
}

//===----------------------------------------------------------------------===//
// INSERTI.S.X
//===----------------------------------------------------------------------===//

class RVStdExtZvinsertInstINSERTI<
  dag outs, dag ins, string opcodestr, string argstr> :
  RVStdExtZvinsertInst<outs, ins, opcodestr, argstr> {

  bits<5> rs2;
  bits<5> imm;
  bits<5> vd;

  let Inst{31-26} = 0b010100;
  let Inst{25} = 0;
  let Inst{24-20} = rs2;
  let Inst{19-15} = imm;
  let Inst{14-12} = OPIVI.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;
}

class VINSERTI<string opcodestr>
    : RVStdExtZvinsertInstINSERTI<(outs VR:$vd), (ins GPR:$rs2, uimm5:$imm),
              opcodestr, "${vd}, ${rs2}, ${imm}">;

def VINSERTI : VINSERTI<"vinserti.s.x">;

//===----------------------------------------------------------------------===//
// INSERT.S.X
//===----------------------------------------------------------------------===//

class RVStdExtZvinsertInstINSERT<
  dag outs, dag ins, string opcodestr, string argstr> :
  RVStdExtZvinsertInst<outs, ins, opcodestr, argstr> {

  bits<5> rs2;
  bits<5> rs1;
  bits<5> vd;

  let Inst{31-26} = 0b010100;
  let Inst{25} = 0;
  let Inst{24-20} = rs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPIVX.Value;
  let Inst{11-7} = vd;
  let Inst{6-0} = OPC_OP_V.Value;
}

class VINSERT<string opcodestr>
    : RVStdExtZvinsertInstINSERT<(outs VR:$vd), (ins GPR:$rs2, GPR:$rs1),
              opcodestr, "${vd}, ${rs2}, (${rs1})">;

def VINSERT : VINSERT<"vinsert.s.x">;

//===----------------------------------------------------------------------===//
// EXTRACTI.S.X
//===----------------------------------------------------------------------===//

class RVStdExtZvinsertInstEXTRACTI<
  dag outs, dag ins, string opcodestr, string argstr> :
  RVStdExtZvinsertInst<outs, ins, opcodestr, argstr> {

  bits<5> vs2;
  bits<5> imm;
  bits<5> rd;

  let Inst{31-26} = 0b010101;
  let Inst{25} = 0;
  let Inst{24-20} = vs2;
  let Inst{19-15} = imm;
  let Inst{14-12} = OPIVI.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;
}

class VEXTRACTI<string opcodestr>
    : RVStdExtZvinsertInstEXTRACTI<(outs GPR:$rd), (ins VR:$vs2, uimm5:$imm),
              opcodestr, "${rd}, ${vs2}, ${imm}">;

def VEXTRACTI : VEXTRACTI<"vextracti.x.s">;

//===----------------------------------------------------------------------===//
// EXTRACT.S.X
//===----------------------------------------------------------------------===//

class RVStdExtZvinsertInstEXTRACT<
  dag outs, dag ins, string opcodestr, string argstr> :
  RVStdExtZvinsertInst<outs, ins, opcodestr, argstr> {

  bits<5> vs2;
  bits<5> rs1;
  bits<5> rd;

  let Inst{31-26} = 0b010101;
  let Inst{25} = 0;
  let Inst{24-20} = vs2;
  let Inst{19-15} = rs1;
  let Inst{14-12} = OPIVX.Value;
  let Inst{11-7} = rd;
  let Inst{6-0} = OPC_OP_V.Value;
}

class VEXTRACT<string opcodestr>
    : RVStdExtZvinsertInstEXTRACT<(outs GPR:$rd), (ins VR:$vs2, GPR:$rs1),
              opcodestr, "${rd}, ${vs2}, (${rs1})">;

def VEXTRACT : VEXTRACT<"vextract.x.s">;