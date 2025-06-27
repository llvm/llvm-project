/* --- PEISelLowering.cpp --- */

/* ------------------------------------------
author: 高宇翔
date: 4/3/2025
------------------------------------------ */

#include "PEISelLowering.h"
#include "MCTargetDesc/PEMCExpr.h"
#include "MCTargetDesc/PEMCTargetDesc.h"
#include "PESubtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/CallingConvLower.h"

using namespace llvm;

#include "PEGenCallingConv.inc"

PETargetLowering::PETargetLowering(const TargetMachine &TM,
                                   const PESubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {

  // enum LegalizeAction : uint8_t {
  //   Legal,   // The target natively supports this operation.
  //   Promote, // This operation should be executed in a larger type.
  //   Expand,  // Try to expand this to other ops, otherwise use a libcall.
  //   LibCall, // Don't try to expand this to other ops, always use a libcall.
  //   Custom   // Use the LowerOperation hook to implement custom lowering.
  // };
  /// 注册RegisterClass
  addRegisterClass(MVT::i32, &PE::GPRRegClass);

  // 根据SubTargetInfo中的寄存器信息计算和更新寄存器属性
  computeRegisterProperties(STI.getRegisterInfo());

  //注册向量合法类型
  addRegisterClass(MVT::v8i32, &PE::VRRegClass);

  setOperationAction(ISD::ADD, MVT::v8i32, Legal);
  setOperationAction(ISD::SUB, MVT::v8i32, Legal);
  setOperationAction(ISD::MUL, MVT::v8i32, Legal);
  setOperationAction(ISD::SDIV, MVT::v8i32, Legal);
  setOperationAction(ISD::LOAD, MVT::v8i32, Legal);
  setOperationAction(ISD::STORE, MVT::v8i32, Legal);

  // 针对 i32 sub 的合法化
  // setOperationAction(ISD::SUB, MVT::i32, Custom);
  // setOperationAction(ISD::SREM, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
}

// 针对于Operation的合法化
SDValue PETargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::GlobalAddress: {
    return LowerGlobalAddress(Op, DAG);
  }
    // case ISD::SREM: {
    //   SDLoc DL(Op);
    //   SDValue LHS = Op.getOperand(0);
    //   SDValue RHS = Op.getOperand(1);
    //   // 1. 生成 DIV 节点，返回 glue
    //   SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
    //   SDValue DivNode = DAG.getNode(PEISD::DIV, DL, VTs, LHS, RHS);

    //   // 2. 生成 DIVR 节点，依赖于 glue
    //   SDValue RemNode =
    //       DAG.getNode(PEISD::DIVR, DL, Op.getValueType(),
    //       DivNode.getValue(1));
    //   return RemNode;
    // }
  }
  return SDValue(); // 其它情况默认处理
}
SDValue PETargetLowering::LowerGlobalAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  GlobalAddressSDNode *N = dyn_cast<GlobalAddressSDNode>(Op.getNode());
  SDLoc DL(N);
  // 1. 获取全局变量的地址
  SDValue Hi =
      DAG.getTargetGlobalAddress(N->getGlobal(), DL, VT, 0, PEMCExpr::HI);
  SDValue Lo =
      DAG.getTargetGlobalAddress(N->getGlobal(), DL, VT, 0, PEMCExpr::LO);

  SDValue HiNode = DAG.getNode(PEISD::HI, DL, VT, Hi);
  SDValue LoNode = DAG.getNode(PEISD::LO, DL, VT, Lo);

  return DAG.getNode(ISD::ADD, DL, VT, HiNode, LoNode);
}
SDValue PETargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  for (unsigned i = 0; i < Ins.size(); ++i) {
    // 这里用undef占位，后续可替换为实际参数
    InVals.push_back(DAG.getUNDEF(Ins[i].VT));
  }
  return Chain;
}
SDValue PETargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool IsVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  // 1. 返回物理寄存器
  SmallVector<CCValAssign, 16> RVLocs; // 存储多个返回值
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeReturn(Outs, RetCC_PE);

  SDValue Glue;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  for (unsigned i = 0, e = RVLocs.size(), OutIdx = 0; i < e; ++i, ++OutIdx) {
    // 遍历返回值
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    // 确保返回值只能通过寄存器返回
    Chain = DAG.getCopyToReg(
        Chain, DL, VA.getLocReg(), OutVals[i],
        Glue); // 生成若干CopyToReg节点，每个 CopyToReg节点输出两个值：新的
               // Chain：用于串接后续操作。Glue：用于保证执行顺序。
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;
  if (Glue.getNode()) {
    RetOps.push_back(Glue);
  }
  return DAG.getNode(PEISD::RET_GLUE, DL, MVT::Other, RetOps);
}

SDValue PETargetLowering::LowerCall(CallLoweringInfo &CLI,
                                    SmallVectorImpl<SDValue> &InVals) const {
  // 1. 解构CLI
  SelectionDAG &DAG = CLI.DAG;
  SDLoc DL = CLI.DL;
  // 2. 获取调用约定
  CallingConv::ID CallConv = CLI.CallConv;
  // 3. 获取返回类型
  Type *RetTy = CLI.RetTy;
  // 4. 获取返回值属性

  // 5. 获取调用链
  SDValue Chain = CLI.Chain;
  // 6. 获取调用目标
  SDValue Callee = CLI.Callee;
  // 7. 获取调用参数
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs; // 函数实参描述信息
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;  // 函数实参
  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;    // 函数形参描述信息

  bool IsVarArg = CLI.IsVarArg;

  /// 1. 处理实参，根据调用约定，通过寄存器或者栈传递参数
  /// 2. 根据参数的寄存器个数，来生成copyFromReg节点
  /// 3. 生成call节点
  /// 4. 处理call的返回值，根据Ins填充InVals

  //(call节点的四个Value：Chain, Callee, RegMask, Glue)
  GlobalAddressSDNode *N = dyn_cast<GlobalAddressSDNode>(Callee);
  Callee = DAG.getTargetGlobalAddress(N->getGlobal(), DL,
                                      getPointerTy(DAG.getDataLayout()));

  SmallVector<SDValue, 8> Ops(1, Chain);
  Ops.push_back(Callee);
  SDValue Glue;

  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask =
      TRI->getCallPreservedMask(DAG.getMachineFunction(), CallConv);
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (Glue.getNode()) {
    Ops.push_back(Glue);
  }

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(PEISD::Call, DL, NodeTys, Ops);

  SmallVector<CCValAssign, 2> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC_PE);

  return Chain;
}
EVT PETargetLowering::getSetCCResultType(const DataLayout &, LLVMContext &,
                                           EVT VT) const {
  if (!VT.isVector())
    return MVT::i32;
  return VT.changeVectorElementTypeToInteger();
}

const char *llvm::PETargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case PEISD::DIV:
    return "PEISD::DIV";
  case PEISD::DIVR:
    return "PEISD::DIVR";
  case PEISD::HI:
    return "PEISD::HI";
  case PEISD::LO:
    return "PEISD::LO";
  case PEISD::RET_GLUE:
    return "PEISD::RET_GLUE";
  case PEISD::Call:
    return "PEISD::Call";
  default:
    return nullptr;
  }
  return nullptr;
}
