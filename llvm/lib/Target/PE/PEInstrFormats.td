//通用指令模版，后序修改

class PEOpcode<string name, bits<7> val,bits<3>funct3> {
  string Name = name;
  bits<7> Value = val;
  bits<3> FUNCT3 = funct3;
}

//实例化操作码
def OPC_ADDI : PEOpcode<"ADDI",0b0010011,0b000>;
def OPC_JALR : PEOpcode<"JALR",0b1101111,0b000>;


class PEInst<dag outs, dag ins, string asmstr,list<dag> pattern>
: Instruction
{
    field bits<32> Inst;//指令的编码，每个bit意义
    field bits<32> SoftFail = 0;//软件错误，传0，用于反汇编程序

    let Size = 4;
    let Namespace = "PE";
    dag OutOperandList = outs;
    dag InOperandList = ins;
    let AsmString = asmstr;
    let Pattern = pattern;
}
//伪指令
class PseudoInst<dag outs, dag ins,list<dag> pattern,string asmstr = "">
    : PEInst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;//表示为伪指令
  let isCodeGenOnly = 1;//存在于代码生成
  let Size = 4;
}
class I<PEOpcode opcode, dag outs, dag ins,
              string asmstr, list<dag> pattern>
    : PEInst< outs, ins, asmstr, pattern> {
//RISCV32的I-type指令格式：imm[11:0] rs1 funct3 rd opcode
  bits<7> Opcode = opcode.Value;
  bits<5> rd;
  bits<3> funct3 = opcode.FUNCT3;
  bits<5> rs1;
  bits<12> imm12;

  let Inst{6-0} = Opcode;
  let Inst{11-7} = rd;
  let Inst{14-12} = funct3;
  let Inst{19-15} = rs1;
  let Inst{31-20} = imm12;
}