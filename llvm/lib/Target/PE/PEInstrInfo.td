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

def imm12 : Operand<i32>,ImmLeaf<i32,[{return isInt<12>(Imm);}]>;
// def imm12Pat : ImmLeaf<i32,[{return isInt<12>(Imm);}]>;

//自定义内存操作数
def mem : Operand<iPTR>{
    ///指明具体操作数
    let MIOperandInfo = (ops GPR,imm12);

    ///指明自定义打印函数
    let PrintMethod = "PrintMemOperand";
}

//地址约束规则，判断是不是地址
def memPat : ComplexPattern<iPTR,2,"SelectAddrFI",[frameindex],[SDNPWantParent]>;

def LOAD : PEInst<(outs GPR: $ra),(ins mem: $addr),"lw \t$ta,$addr">;
def : Pat<(loat memPat:$addr),(LOAD memPat:$addr)>;

def STORE : PEInst<(outs),(ins GPR: $ra,mem: $addr),"sw \t$ta,$addr">;
def : Pat<(store GPR:$ra memPat:$addr),(STORE GPR:$ra memPat:$addr)>;

//class I 的实例化
//OPC_ADDI：指令码，GPR:$rd：一个通用寄存器$rd，i32imm:$imm12：一个立即数$imm12
//"addi $rd,$rs1,$imm12"：汇编字符串
//[]：模式匹配为空
def ADDI : I<(outs GPR:$rd),(ins GPR:$rs1,imm12:$imm12),
                             "addi $rd,$rs1,$imm12",[]>;

//当IR中出现形如i32 (imm:$imm12)的立即数操作时（例如加载一个32位整数到寄存器），TableGen会将其匹配为ADDI X0, imm:$imm12。X0为RISCV中的0寄存器，恒为0
def : Pat<(i32 (imm12:$imm12)),(ADDI X0,imm12:$imm12)>;

// JALR x1, x5, 0    # 跳转到x5的值，返回地址保存到x1
// JALR x0, x10, 8   # 跳转到x10+8，不保存返回地址（用于纯跳转）
// isCall = 1标识该指令为函数调用指令
let isCall = 1 in{
    def JALR : I<(outs GPR:$rd),(ins GPR:$rs1,imm12:$imm12),
    "jalr $rd,$rs1,$imm12",[]>;
}

//​isReturn = 1**：标识为函数返回指令。
// ​isTerminator = 1**：标记为基本块终止指令（如分支、返回等），编译器不会在其后插入其他指令。
// ​isBarrier = 1**：防止指令重排序优化，确保返回操作原子性。
//pseudoRET伪指令定义,无显示输入和输出。[(retglue)]：在LLVM的SelectionDAG中表示指令与返回操作的粘附（glue）依赖，确保代码生成顺序正确
//PseudoInstExpansion<JALR X0, X1, 0>**：将伪指令pseudoRET映射到实际指令JALR X0, X1, 0

//实际是做指令替换，将IR中的ret伪指令翻译成JALR X0, X1, 0
let isBarrier = 1,isReturn = 1,isTerminator = 1 in{
    def pseudoRET : PseudoInst<(outs),(ins),[(retglue)]>,PseudoInstExpansion<(JALR X0, X1, 0)>;
}

def : InstAlias<"ret",(JALR X0, X1, 0)>;
def : InstAlias<"li $rd,&imm",(ADDI GPR:$rd,X0,imm12:&imm)>;