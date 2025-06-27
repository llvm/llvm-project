//////////////////////7004指令分类
// 1.立即数MOV指令
// 2.单操作数指令
// 3.双/三操作数指令
// 4.长立即数分支指令
// 5.普通分支指令
// 6.立即数LOAD/STORE指令
// 7.LOAD/STORE指令
// 8.16位指控指令
// 9.16位LOAD/STORE指令
// 10.16位运算指令

class InstFormat<bits<5> val> {
  bits<5> Value = val;
}
def InstFormatPseudo : InstFormat<0>;
def InstFormatMOV      : InstFormat<1>;
def InstFormatSO       : InstFormat<2>; // Single Operand
def InstFormatDTO      : InstFormat<3>; // Dual/Triple Operand
def InstFormatLongBranch   : InstFormat<4>; // Long Immediate Branch
def InstFormatBranch       : InstFormat<5>; // Normal Branch
def InstFormatImmLoadStore : InstFormat<6>; // Immediate Load/Store
def InstFormatLoadStore    : InstFormat<7>; // Load/Store
def InstFormatCtrl16       : InstFormat<8>; // 16-bit Control
def InstFormatLoadStore16  : InstFormat<9>; // 16-bit Load/Store
def InstFormatArith16      : InstFormat<10>; // 16-bit Arithmetic

//当前向量指令元素依赖,向量运算结果是否取决于VPE状态
class EltDeps<bit vpe> {
  bit VPE = vpe;
}

def EltDepsNone      : EltDeps<vpe=0>;
def EltDepsVPE       : EltDeps<vpe=1>;

//当前向量指令元素宽度
class EEW <bit val> {
  bits Value = val;
}
def EEW16   : EEW<0>;
def EEW32   : EEW<1>;

//指令基类
class FTInstCommon<dag outs, dag ins, string opcodestr, string argstr,
                   list<dag> pattern, InstFormat format> : Instruction {
  let Namespace = "FT7004";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString = opcodestr # !if(!empty(argstr), "", "\t" # argstr);
  let Pattern = pattern;
}

//标准32位指令
class FTInst<dag outs, dag ins, string opcodestr, string argstr,
             list<dag> pattern, InstFormat format>
    : FTInstCommon<outs, ins, opcodestr, argstr, pattern, format> {
  field bits<32> Inst;
  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<32> SoftFail = 0;
  let Size = 4;
}

//16位指令
class FTInst16<dag outs, dag ins, string opcodestr, string argstr,
             list<dag> pattern, InstFormat format>
    : FTInstCommon<outs, ins, opcodestr, argstr, pattern, format> {
  field bits<16> Inst;
  field bits<16> SoftFail = 0;
  let Size = 2;
}

//伪指令
class Pseudo<dag outs, dag ins, list<dag> pattern, string opcodestr = "", string argstr = "">
    : FTInstCommon<outs, ins, opcodestr, argstr, pattern, InstFormatPseudo> {
  let isPseudo = 1;         // 标记为伪指令
  let isCodeGenOnly = 1;    // 仅用于代码生成，不参与解码
  let Size = 4;             // 默认长度为4字节（方便统一处理）
}

// Pseudo load instructions.  bare_symbol代表一个符号地址（如标签、全局变量名）
class PseudoLoad<string opcodestr, DAGOperand rdty = GPR>
    : Pseudo<(outs rdty:$rd), (ins bare_symbol:$addr), [], opcodestr, "$rd, $addr"> {
  let hasSideEffects = 0;
  let mayLoad = 1;
  let mayStore = 0;
  let isCodeGenOnly = 0;
  let isAsmParserOnly = 1;
}

// Pseudo store instructions.
class PseudoStore<string opcodestr, DAGOperand rsty = GPR>
    : Pseudo<(outs GPR:$tmp), (ins rsty:$rs, bare_symbol:$addr), [], opcodestr, "$rs, $addr, $tmp"> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 1;
  let isCodeGenOnly = 0;
  let isAsmParserOnly = 1;
}

