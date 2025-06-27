include "PEInstrFormats.td"
include "PERegisterInfo.td"
/*
    定义具体的指令信息，每一条指令对应一个def

    BasicBlock(IR) -> SelectionDAG(SDNode)
    数据结构从IR到DAG图
    ret语句需要调用特定后端的lowerReturn函数进行处理

*/
//SDNode代表指令约束，PEISD::RET_GLUE代表节点的枚举值，SDTNode：节点的类型约束
//[SDNPHasChain,SDNPOptInGlue,SDNPVariadic] ：指令属性
def retglue : SDNode <"PEISD::RET_GLUE",SDTNone,
                            [SDNPHasChain,SDNPOptInGlue,SDNPVariadic]>;
let isReturn = 1, isBarrier = 1 in
def RET : I<
  (outs),
  (ins),
  "RET",
  [(retglue)]
>;

def imm12 : Operand<i32>,ImmLeaf<i32,[{return isInt<12>(Imm);}]>;
def imm16 : Operand<i32>,ImmLeaf<i32,[{return isInt<16>(Imm);}]>;
def imm20 : Operand<i32>,ImmLeaf<i32,[{return isInt<20>(Imm);}]>;
def Constant : Operand<i32>,ImmLeaf<i32,[{return isInt<32>(Imm);}]>;
def NonZeroImm32 : ImmLeaf<i32, [{ return Imm != 0; }]>;
def ZeroImm : ImmLeaf<i32, [{ return Imm = 0; }]>;

//自定义内存操作数
def mem : Operand<iPTR>{
    ///指明具体操作数
    let MIOperandInfo = (ops GPR,imm12);
    ///指明自定义打印函数
    let PrintMethod = "PrintMemOperand";
}

def vmem : Operand<iPTR>{
    ///指明具体操作数
    let MIOperandInfo = (ops GPR,imm12);
    ///指明自定义打印函数
    let PrintMethod = "PrintVMemOperand";
}

def ptr : Operand<iPTR>{
    ///指明具体操作数
    let MIOperandInfo = (ops GPR,imm12);

    ///指明自定义打印函数
    let PrintMethod = "PrintPtrOperand";
}

//地址约束规则，判断是不是地址 12(sp)
def AddrFI : ComplexPattern<iPTR,2,"SelectAddrFI",[frameindex],[]>;

/// 每一种store都有frameindex和register两种情况，可以使用multiclass抽取公共的部分 
multiclass STORE<PatFrag op, string asm> {
	def FI : S<(outs), (ins GPR:$ra, mem:$addr), !strconcat(asm, "\t$ra, $addr"), [(op GPR:$ra, AddrFI:$addr)]>;

    def VFI : S<(outs), (ins VR:$ra, vmem:$addr), !strconcat(asm, "\t$ra, $addr"), [(op VR:$ra, AddrFI:$addr)]>;

	def R : S<(outs), (ins GPR:$ra, GPR:$rb), !strconcat(asm, "\t$ra, 0($rb)"), [(op GPR:$ra, GPR:$rb)]>;	
}
defm STOREW:STORE<store,"sw">;
defm STOREH:STORE<truncstorei16,"sh">;
defm STOREB:STORE<truncstorei8,"sb">;


/// 同理每一个load也有针对frameindex和register的两种情况
multiclass LOAD<PatFrag op, string asm> {
	def FI : I<(outs GPR:$ra), (ins mem:$addr), !strconcat(asm, "\t$ra, $addr"), [(set GPR:$ra, (op AddrFI:$addr))]>;

    def VFI : I<(outs VR:$ra), (ins vmem:$addr), !strconcat(asm, "\t$ra, $addr"), [(set VR:$ra, (op AddrFI:$addr))]>;

	def R : I<(outs GPR:$ra), (ins GPR:$rb), !strconcat(asm, "\t$ra, 0($rb)"), [(set GPR:$ra, (op GPR:$rb))]>;
}
defm LOADW:LOAD<load,"lw">;

defm LOADH:LOAD<extloadi16,"lh">;
defm LOADSH:LOAD<sextloadi16,"lh">;
defm LOADZH:LOAD<zextloadi16,"lh">;

defm LOADB:LOAD<extloadi8,"lb">;
defm LOADSB:LOAD<sextloadi8,"lb">;
defm LOADZB:LOAD<zextloadi8,"lb">;


def : I<(outs GPR:$ra),(ins ptr:$p),"LD \t$ra, $p",[(set GPR:$ra,AddrFI:$p)]>;//将frameindex->GPR
//运算指令
class ArithLogicI<string inst, SDNode node> : I<
    (outs GPR: $rd),
    (ins GPR: $rs1, imm16: $imm16),
    !strconcat(inst, "\t$rd, $rs1, $imm16"),
    [(set GPR:$rd, (node GPR:$rs1, imm16:$imm16))]
>;
def ADDI : ArithLogicI<"ADDI", add>;//使用LLVM内置add node匹配
def ANDI : ArithLogicI<"ANDI", and>;
def ORI : ArithLogicI<"ORI", or>;
def XORI : ArithLogicI<"XORI", xor>;
def SLLI : ArithLogicI<"SLLI", shl>;
def SRLI : ArithLogicI<"SRLI", srl>;

class ArithLogicR<string inst, SDNode node> : I<
    (outs GPR: $rd),
    (ins GPR: $rs1, GPR: $rs2),
    !strconcat(inst, "\t$rd, $rs1, $rs2"),
    [(set GPR:$rd, (node GPR: $rs1, GPR: $rs2))]
>;
def ADD : ArithLogicR<"ADD", add>;
def AND : ArithLogicR<"AND", and>;
def OR : ArithLogicR<"OR", or>;
def XOR : ArithLogicR<"XOR", xor>;
def SLL : ArithLogicR<"SLL", shl>;
def SRL : ArithLogicR<"SRL", srl>;
def SUB : ArithLogicR<"SUB", sub>;
def MUL : ArithLogicR<"MUL", mul>;
def DIV : ArithLogicR<"DIV", sdiv>;
def DIVR : ArithLogicR<"DIVR", srem>;
// 匹配其它类型到DIV
def : Pat<(sdiv GPR:$rs1, GPR:$rs2), (DIV GPR:$rs1, GPR:$rs2)>;
def : Pat<(udiv GPR:$rs1, GPR:$rs2), (DIV GPR:$rs1, GPR:$rs2)>;

//向量基础运算指令
class ArithLogicVR<string inst, SDNode node> : I<
    (outs VR: $rd),
    (ins VR: $rs1, VR: $rs2),
    !strconcat(inst, "\t$rd, $rs1, $rs2"),
    [(set VR:$rd, (node VR: $rs1, VR: $rs2))]
>;
def ADDV : ArithLogicVR<"ADDV", add>;
def SUBV : ArithLogicVR<"SUBV", sub>;
def HADP : ArithLogicVR<"HADP", mul>;
def DIVS : ArithLogicVR<"DIVS", sdiv>;

//LLVM没有向量*标量，需要现将标量变为向量再去匹配
// class ArithLogicR2VR<string inst, SDNode node> : I<
//     (outs VR: $rd),
//     (ins VR: $rs1, GPR: $rs2),
//     !strconcat(inst, "\t$rd, $rs1, $rs2"),
//     [(set VR:$rd, (node VR: $rs1, GPR: $rs2))]
// >;
// def MULS : ArithLogicR2VR<"MULS", mul>;

//向量规约
def SOD : I<
    (outs GPR:$rd),
    (ins VR:$rs1),
    "SOD\t$rd, $rs1",
    [(set GPR:$rd, (vecreduce_add VR:$rs1))]
>;

//向量点积
def VIP : I<
    (outs GPR:$rd),
    (ins VR:$rs1, VR:$rs2),
    "VIP\t$rd, $rs1, $rs2",
    [(set GPR:$rd, (vecreduce_add (mul VR:$rs1, VR:$rs2)))]
>;


//比较指令
def SLT : I<
    (outs GPR:$rd), 
    (ins GPR:$rs1, GPR:$rs2), 
	"SLT\t$rd, $rs1, $rs2", 
    []
>;

def : Pat<(setgt GPR:$lhs, GPR:$rhs), (SLT GPR:$rhs, GPR:$lhs)>;//大于，交换输入寄存器
def : Pat<(setlt GPR:$lhs, GPR:$rhs), (SLT GPR:$lhs, GPR:$rhs)>;//小于
def : Pat<(seteq GPR:$lhs, GPR:$rhs), (SLT (XOR GPR:$lhs, GPR:$rhs), 1)>;//等于
def : Pat<(setne GPR:$lhs, GPR:$rhs), (SLT RS0, (XOR GPR:$lhs, GPR:$rhs))>;//不等于
def : Pat<(setge GPR:$lhs, GPR:$rhs), (XORI (SLT GPR:$lhs, GPR:$rhs), 1)>;
def : Pat<(setle GPR:$lhs, GPR:$rhs), (XORI (SLT GPR:$rhs, GPR:$lhs), 1)>;

//取余指令 %
// def DIVR : PEInst<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2), "DIVR\t$rd,$rs1,$rs2", []>;
// def : Pat<(strm GPR:$rs1, GPR:$rs2), (DIVR GPR:$rs1, GPR:$rs2)>

// def DIVR : PEInst<(outs GPR:$rd), (ins), "DIVR\t$rd", []>;
// def DIV_SDN : SDNode<"PEISD::DIV", SDTIntBinOp, [SDNPHasChain, SDNPOutGlue]>;
// def SDTIntZeroOp : SDTypeProfile<1, 0, [SDTCisInt<0>]>;
// def DIVR_SDN : SDNode<"PEISD::DIVR", SDTIntZeroOp, [SDNPHasChain, SDNPInGlue]>;
// def : Pat<(DIV_SDN GPR:$rs1, GPR:$rs2), (DIV GPR:$rs1, GPR:$rs2)>;
// def : Pat<(DIVR_SDN), (DIVR)>;


//跳转指令
def brtarget :Operand<OtherVT>{
  let PrintMethod = "printBranchOperand";
}

let isTerminator = 1 in{
    def J : J<(outs),(ins brtarget : $imm12),"J ${imm12}",[]>; 
    def BL : B<(outs),(ins GPR : $rs1,GPR : $rs2,brtarget : $imm),"BL \t$rs1,$rs2,$imm",[]>;//大于
    def BGE : B<(outs),(ins GPR : $rs1,GPR : $rs2,brtarget : $imm),"BGE \t$rs1,$rs2,$imm",[]>;//小于等于
    def BEQ : B<(outs),(ins GPR : $rs1,GPR : $rs2,brtarget : $imm),"BEQ \t$rs1,$rs2,$imm",[]>;//等于
    def BNE : B<(outs),(ins GPR : $rs1,GPR : $rs2,brtarget : $imm),"BNE \t$rs1,$rs2,$imm",[]>;//不等于
}
def : Pat<(brcond (i32(setgt GPR:$rs1, GPR:$rs2)),bb:$dst),(BL GPR:$rs1, GPR:$rs2, brtarget:$dst)>;//大于
def : Pat<(brcond (i32(setlt GPR:$rs1, GPR:$rs2)),bb:$dst),(BL GPR:$rs2, GPR:$rs1, brtarget:$dst)>;//小于
def : Pat<(brcond (i32(setge GPR:$rs1, GPR:$rs2)),bb:$dst),(BGE GPR:$rs2, GPR:$rs1, brtarget:$dst)>;//大于等于
def : Pat<(brcond (i32(setle GPR:$rs1, GPR:$rs2)),bb:$dst),(BGE GPR:$rs1, GPR:$rs2, brtarget:$dst)>;//小于等于
def : Pat<(brcond (i32(seteq GPR:$rs1, GPR:$rs2)),bb:$dst),(BEQ GPR:$rs1, GPR:$rs2, brtarget:$dst)>;//等于
def : Pat<(brcond (i32(setne GPR:$rs1, GPR:$rs2)),bb:$dst),(BNE GPR:$rs1, GPR:$rs2, brtarget:$dst)>;//不等于
// def : Pat<(brcond (i32 (sub GPR:$rs1, Constant:$imm)), bb:$dst),
//           (BNE GPR:$rs1, Constant:$imm, brtarget:$dst)>;

def : Pat<(br bb:$dst),(J brtarget:$dst)>;

def : Pat<(brcond (i32 (xor (setge GPR:$rs1, GPR:$rs2), 1)), bb:$dst),
          (BL GPR:$rs2, GPR:$rs1, brtarget:$dst)>; // 小于
def : Pat<(brcond (i32 (xor (setle GPR:$rs1, GPR:$rs2), 1)), bb:$dst),
          (BL GPR:$rs1, GPR:$rs2, brtarget:$dst)>; // 大于
def : Pat<(brcond (i32 (xor (setgt GPR:$rs1, GPR:$rs2), 1)), bb:$dst),
          (BGE GPR:$rs1, GPR:$rs2, brtarget:$dst)>; // 小于等于
def : Pat<(brcond (i32 (xor (setlt GPR:$rs1, GPR:$rs2), 1)), bb:$dst),
          (BGE GPR:$rs2, GPR:$rs1, brtarget:$dst)>; // 大于等于
def : Pat<(brcond (i32 (xor (setne GPR:$rs1, GPR:$rs2), 1)), bb:$dst),
          (BEQ GPR:$rs1, GPR:$rs2, brtarget:$dst)>; // 等于
def : Pat<(brcond (i32 (xor (seteq GPR:$rs1, GPR:$rs2), 1)), bb:$dst),
          (BNE GPR:$rs1, GPR:$rs2, brtarget:$dst)>; // 不等于

//当IR中出现形如i32 (imm:$imm16)的立即数操作时（例如加载一个32位整数到寄存器），TableGen会将其匹配为ADDI X0, imm:$imm16。X0为RISCV中的0寄存器，恒为0
def : Pat<(i32 (imm16:$imm16)), (ADDI RS0, imm16:$imm16)>;
def : Pat<(i32 (Constant:$imm)), (ADDI RS0, Constant:$imm)>;

// def : InstAlias<"movil $rd, $imm16", (ADDI GPR:$rd, RS0, imm16:$imm16)>;

// let isCall = 1 in {
//   def JALR : I<(outs GPR:$rd),
//               (ins GPR:$rs1, imm12:$imm12),
//               "jalr $rd, $rs1, $imm12",
//               []>;
// }
// def : InstAlias<"ret", (JALR RS0, RS2, 0)>;
def : InstAlias<"ADDI\t$rd, RS0, $imm16", (ADDI GPR:$rd, RS0, imm16:$imm16)>;


//Call指令,函数调用

/// 1、Call 的SDNode的定义
/// 2、SDNode pattern Instruction 
def PESDT_Call : SDTypeProfile<0, -1, [SDTCisVT<0, iPTR>]>;//对于SDNode约束，0个输入参数，可变长个输出参数，第0个参数为iPTR类型（指针）
def pe_call  : SDNode<"PEISD::Call", PESDT_Call,
                            [SDNPHasChain, SDNPOutGlue,
                             SDNPOptInGlue, SDNPVariadic]>;
def call_symbol : Operand<iPTR>;

//CALL指令定义
def CALL : I <(outs),(ins call_symbol : $symbol), "call\t$symbol",[(pe_call tglobaladdr:$symbol)]>;


//全局变量（暂未实现有bug）
def HI : SDNode<"PEISD::HI", SDTIntUnaryOp>;
def LO : SDNode<"PEISD::LO", SDTIntUnaryOp>;

def LUI : U<(outs GPR:$rd),(ins imm20:$imm20),"lui \t$rd,$imm20",[]>;
def : Pat<(HI tglobaladdr:$in),(LUI tglobaladdr:$in)>;
def : Pat<(add GPR:$hi, (LO tglobaladdr:$lo)), (ADDI GPR:$hi, tglobaladdr:$lo)>;

