/* --- PEISelLowering.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/3/2025
------------------------------------------ */

#include "PEISelLowering.h"
#include "PESubtarget.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "llvm/Support/GraphWriter.h" // For ViewGraph

using namespace llvm;

#include "PEGenCallingConv.inc"

PETargetLowering::PETargetLowering(const TargetMachine &TM,
                                   const PESubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
        ///注册RegisterClass
        addRegisterClass(MVT::i32, &PE::GPRRegClass);

        // 根据SubTargetInfo中的寄存器信息计算和更新寄存器属性
        computeRegisterProperties(STI.getRegisterInfo());
    }

SDValue PETargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  return Chain;
}
void GenerateDOTFile(const SelectionDAG &DAG) {
    std::error_code EC;
    raw_fd_ostream File("dag.dot", EC, sys::fs::OF_Text);
    if (!EC) {
        WriteGraph(File, &DAG);
        errs() << "DAG written to dag.dot\n";
    } else {
        errs() << "Error writing DAG to file: " << EC.message() << "\n";
    }
}
SDValue PETargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                      bool IsVarArg,
                                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                                     const SmallVectorImpl<SDValue> &OutVals,
                                      const SDLoc &DL, SelectionDAG &DAG) const {
  // errs() << "Current DAG:\n";
  // DAG.dump();
  //  // 将 DAG 转换为 DOT 文件
  //  std::error_code EC;
  //  raw_fd_ostream File("dag.dot", EC, sys::fs::OF_Text);
  //  if (!EC) {
  //    WriteGraph(File, &DAG);
  //    errs() << "DAG written to dag.dot\n";
  //  } else {
  //    errs() << "Error writing DAG to file: " << EC.message() << "\n";
  //  }

  // 1. 返回物理寄存器
  DAG.viewGraph();
  SmallVector<CCValAssign, 16> RVLocs;//存储多个返回值
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),RVLocs,*DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_PE);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  for (unsigned i = 0, e = RVLocs.size(), OutIdx = 0; i < e; ++i, ++OutIdx) {//遍历返回值
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!"); //确保返回值只能通过寄存器返回
    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), OutVals[i], Glue);//生成若干CopyToReg节点，每个 CopyToReg 节点输出两个值：新的 Chain：用于串接后续操作。Glue：用于保证执行顺序。
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;
  if (Glue.getNode())
  {
    RetOps.push_back(Glue); 
  }
    return DAG.getNode(PEISD::RET_GLUE, DL, MVT::Other, RetOps);
}

const char *llvm::PETargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case PEISD::RET_GLUE:
    return "PEISD::RET_GLUE"; 
  default:
    return nullptr;
  }
  return nullptr;
}
