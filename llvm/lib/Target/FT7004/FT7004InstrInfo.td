include "FT7004InstrFormats.td"
include "FT7004RegisterInfo.td"

// 这里定义具体的指令信息，每一条指令对应一个def
// 示例：返回指令
// let isReturn = 1, isBarrier = 1 in
def RET : I<
  (outs),
  (ins),
  "RET",
  []
>;

// 操作数类型定义

def imm12 : Operand<i32>, ImmLeaf<i32, [{return isInt<12>(Imm);}]>;
def imm16 : Operand<i32>, ImmLeaf<i32, [{return isInt<16>(Imm);}]>;
def imm20 : Operand<i32>, ImmLeaf<i32, [{return isInt<20>(Imm);}]>;
def Constant : Operand<i32>, ImmLeaf<i32, [{return isInt<32>(Imm);}]>;

def mem : Operand<iPTR> {
  let MIOperandInfo = (ops GPR, imm12);
  let PrintMethod = "PrintMemOperand";
}

def AddrFI : ComplexPattern<iPTR,2,"SelectAddrFI",[frameindex],[]>;

// 内存访问指令
multiclass STORE<PatFrag op, string asm> {
  def FI : S<(outs), (ins GPR:$ra, mem:$addr), !strconcat(asm, "\t$ra, $addr"), [(op GPR:$ra, AddrFI:$addr)]>;
  def R  : S<(outs), (ins GPR:$ra, GPR:$rb), !strconcat(asm, "\t$ra, 0($rb)"), [(op GPR:$ra, GPR:$rb)]>;
}
defm STOREW:STORE<store,"sw">;
defm STOREH:STORE<truncstorei16,"sh">;
defm STOREB:STORE<truncstorei8,"sb">;

multiclass LOAD<PatFrag op, string asm> {
  def FI : I<(outs GPR:$ra), (ins mem:$addr), !strconcat(asm, "\t$ra, $addr"), [(set GPR:$ra, (op AddrFI:$addr))]>;
  def R  : I<(outs GPR:$ra), (ins GPR:$rb), !strconcat(asm, "\t$ra, 0($rb)"), [(set GPR:$ra, (op GPR:$rb))]>;
}
defm LOADW:LOAD<load,"lw">;
defm LOADH:LOAD<extloadi16,"lh">;
defm LOADSH:LOAD<sextloadi16,"lh">;
defm LOADZH:LOAD<zextloadi16,"lh">;
defm LOADB:LOAD<extloadi8,"lb">;
defm LOADSB:LOAD<sextloadi8,"lb">;
defm LOADZB:LOAD<zextloadi8,"lb">;

// 算术/逻辑指令（立即数型）
class ArithLogicI<string inst, SDNode node> : I<
  (outs GPR:$rd),
  (ins GPR:$rs1, imm16:$imm16),
  !strconcat(inst, "\t$rd, $rs1, $imm16"),
  [(set GPR:$rd, (node GPR:$rs1, imm16:$imm16))]
>;
def ADDI : ArithLogicI<"ADDI", add>;
def ANDI : ArithLogicI<"ANDI", and>;
def ORI  : ArithLogicI<"ORI", or>;
def XORI : ArithLogicI<"XORI", xor>;
def SLLI : ArithLogicI<"SLLI", shl>;
def SRLI : ArithLogicI<"SRLI", srl>;

// 算术/逻辑指令（寄存器型）
class ArithLogicR<string inst, SDNode node> : I<
  (outs GPR:$rd),
  (ins GPR:$rs1, GPR:$rs2),
  !strconcat(inst, "\t$rd, $rs1, $rs2"),
  [(set GPR:$rd, (node GPR:$rs1, GPR:$rs2))]
>;
def ADD : ArithLogicR<"ADD", add>;
def SUB : ArithLogicR<"SUB", sub>;
def AND : ArithLogicR<"AND", and>;
def OR  : ArithLogicR<"OR", or>;
def XOR : ArithLogicR<"XOR", xor>;
def SLL : ArithLogicR<"SLL", shl>;
def SRL : ArithLogicR<"SRL", srl>;
def MUL : ArithLogicR<"MUL", mul>;
def DIV : ArithLogicR<"DIV", sdiv>;
def REM : ArithLogicR<"REM", srem>;

def : Pat<(sdiv GPR:$rs1, GPR:$rs2), (DIV GPR:$rs1, GPR:$rs2)>;
def : Pat<(srem GPR:$rs1, GPR:$rs2), (REM GPR:$rs1, GPR:$rs2)>;

// 比较指令
class CmpR<string inst, SDNode node> : I<
  (outs GPR:$rd),
  (ins GPR:$rs1, GPR:$rs2),
  !strconcat(inst, "\t$rd, $rs1, $rs2"),
  [(set GPR:$rd, (node GPR:$rs1, GPR:$rs2))]
>;
def SLT : CmpR<"SLT", setlt>;
def SGT : CmpR<"SGT", setgt>;
def SLE : CmpR<"SLE", setle>;
def SGE : CmpR<"SGE", setge>;
def SEQ : CmpR<"SEQ", seteq>;
def SNE : CmpR<"SNE", setne>;

// 跳转/分支指令
let isTerminator = 1 in {
  def J   : J<(outs), (ins brtarget:$dst), "J $dst",[]>;
  def BEQ : B<(outs), (ins GPR:$rs1, GPR:$rs2, brtarget:$dst), "BEQ $rs1, $rs2, $dst",[]>;
  def BNE : B<(outs), (ins GPR:$rs1, GPR:$rs2, brtarget:$dst), "BNE $rs1, $rs2, $dst",[]>;
  def BLT : B<(outs), (ins GPR:$rs1, GPR:$rs2, brtarget:$dst), "BLT $rs1, $rs2, $dst",[]>;
  def BGE : B<(outs), (ins GPR:$rs1, GPR:$rs2, brtarget:$dst), "BGE $rs1, $rs2, $dst",[]>;
}

def brtarget : Operand<OtherVT> {
  let PrintMethod = "printBranchOperand";
}

// 调用指令
// SDNode/Pattern
// def FT7004SDT_Call : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;
def ft7004_call : SDNode<"FT7004ISD::Call", SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>, [SDNPHasChain, SDNPOutGlue, SDNPOptInGlue, SDNPVariadic]>;
def call_symbol : Operand<iPTR>;
def CALL : I<(outs), (ins call_symbol:$symbol), "call\t$symbol", [(ft7004_call tglobaladdr:$symbol)]>;

// 向量基础运算指令
class ArithLogicVR<string inst, SDNode node> : I<
  (outs VR:$rd),
  (ins VR:$rs1, VR:$rs2),
  !strconcat(inst, "\t$rd, $rs1, $rs2"),
  [(set VR:$rd, (node VR:$rs1, VR:$rs2))]
>;
def VADD : ArithLogicVR<"VADD", add>;
def VSUB : ArithLogicVR<"VSUB", sub>;
def VMUL : ArithLogicVR<"VMUL", mul>;
def VDIV : ArithLogicVR<"VDIV", sdiv>;

// 向量规约
// 例：向量加法规约到标量
def VREDUCE_ADD : I<
  (outs GPR:$rd),
  (ins VR:$rs1),
  "VREDUCE_ADD\t$rd, $rs1",
  [(set GPR:$rd, (vecreduce_add VR:$rs1))]
>;
// 向量点积
def VDOT : I<
  (outs GPR:$rd),
  (ins VR:$rs1, VR:$rs2),
  "VDOT\t$rd, $rs1, $rs2",
  [(set GPR:$rd, (vecreduce_add (mul VR:$rs1, VR:$rs2)))]
>;

// 全局变量加载相关
def HI : SDNode<"FT7004ISD::HI", SDTIntUnaryOp>;
def LO : SDNode<"FT7004ISD::LO", SDTIntUnaryOp>;
def LUI : U<(outs GPR:$rd), (ins imm20:$imm20), "lui\t$rd, $imm20",[]>;
def : Pat<(HI tglobaladdr:$in), (LUI tglobaladdr:$in)>;
def : Pat<(add GPR:$hi, (LO tglobaladdr:$lo)), (ADDI GPR:$hi, tglobaladdr:$lo)>;

// 伪指令/别名
def : InstAlias<"nop", (ADDI GPR:$rd, GPR:$rd, 0)>;
def : InstAlias<"move $rd, $rs", (ADDI GPR:$rd, GPR:$rs, 0)>;

// 常用Pattern
// 立即数加载优化
def : Pat<(i32 (imm16:$imm16)), (ADDI R0, imm16:$imm16)>;
def : Pat<(i32 (Constant:$imm)), (ADDI R0, Constant:$imm)>;
