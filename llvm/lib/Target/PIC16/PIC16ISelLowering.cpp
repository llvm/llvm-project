//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PIC16 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-lower"

#include "PIC16ISelLowering.h"
#include "PIC16TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Function.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <cstdio>


using namespace llvm;


// PIC16TargetLowering Constructor.
PIC16TargetLowering::PIC16TargetLowering(PIC16TargetMachine &TM)
  : TargetLowering(TM) {
  
  Subtarget = &TM.getSubtarget<PIC16Subtarget>();

  addRegisterClass(MVT::i8, PIC16::GPRRegisterClass);

  setShiftAmountType(MVT::i8);
  setShiftAmountFlavor(Extend);

  // SRA library call names
  setPIC16LibcallName(PIC16ISD::SRA_I8, "__intrinsics.sra.i8");
  setLibcallName(RTLIB::SRA_I16, "__intrinsics.sra.i16");
  setLibcallName(RTLIB::SRA_I32, "__intrinsics.sra.i32");

  // SHL library call names
  setPIC16LibcallName(PIC16ISD::SLL_I8, "__intrinsics.sll.i8");
  setLibcallName(RTLIB::SHL_I16, "__intrinsics.sll.i16");
  setLibcallName(RTLIB::SHL_I32, "__intrinsics.sll.i32");

  // SRL library call names
  setPIC16LibcallName(PIC16ISD::SRL_I8, "__intrinsics.srl.i8");
  setLibcallName(RTLIB::SRL_I16, "__intrinsics.srl.i16");
  setLibcallName(RTLIB::SRL_I32, "__intrinsics.srl.i32");

  // MUL Library call names
  setPIC16LibcallName(PIC16ISD::MUL_I8, "__intrinsics.mul.i8");
  setLibcallName(RTLIB::MUL_I16, "__intrinsics.mul.i16");
  setLibcallName(RTLIB::MUL_I32, "__intrinsics.mul.i32");

  setOperationAction(ISD::GlobalAddress, MVT::i16, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i16, Custom);

  setOperationAction(ISD::LOAD,   MVT::i8,  Legal);
  setOperationAction(ISD::LOAD,   MVT::i16, Custom);
  setOperationAction(ISD::LOAD,   MVT::i32, Custom);

  setOperationAction(ISD::STORE,  MVT::i8,  Legal);
  setOperationAction(ISD::STORE,  MVT::i16, Custom);
  setOperationAction(ISD::STORE,  MVT::i32, Custom);

  setOperationAction(ISD::ADDE,    MVT::i8,  Custom);
  setOperationAction(ISD::ADDC,    MVT::i8,  Custom);
  setOperationAction(ISD::SUBE,    MVT::i8,  Custom);
  setOperationAction(ISD::SUBC,    MVT::i8,  Custom);
  setOperationAction(ISD::ADD,    MVT::i8,  Custom);
  setOperationAction(ISD::ADD,    MVT::i16, Custom);

  setOperationAction(ISD::OR,     MVT::i8,  Custom);
  setOperationAction(ISD::AND,    MVT::i8,  Custom);
  setOperationAction(ISD::XOR,    MVT::i8,  Custom);

  setOperationAction(ISD::FrameIndex, MVT::i16, Custom);
  setOperationAction(ISD::CALL,   MVT::i16, Custom);
  setOperationAction(ISD::RET,    MVT::Other, Custom);

  setOperationAction(ISD::MUL,    MVT::i8,  Custom); 
  setOperationAction(ISD::MUL,    MVT::i16, Expand);
  setOperationAction(ISD::MUL,    MVT::i32, Expand);

  setOperationAction(ISD::SMUL_LOHI,    MVT::i8,  Expand);
  setOperationAction(ISD::SMUL_LOHI,    MVT::i16, Expand);
  setOperationAction(ISD::SMUL_LOHI,    MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI,    MVT::i8,  Expand);
  setOperationAction(ISD::UMUL_LOHI,    MVT::i16, Expand);
  setOperationAction(ISD::UMUL_LOHI,    MVT::i32, Expand);
  setOperationAction(ISD::MULHU,        MVT::i8, Expand);
  setOperationAction(ISD::MULHU,        MVT::i16, Expand);
  setOperationAction(ISD::MULHU,        MVT::i32, Expand);
  setOperationAction(ISD::MULHS,        MVT::i8, Expand);
  setOperationAction(ISD::MULHS,        MVT::i16, Expand);
  setOperationAction(ISD::MULHS,        MVT::i32, Expand);

  setOperationAction(ISD::SRA,    MVT::i8,  Custom);
  setOperationAction(ISD::SRA,    MVT::i16, Expand);
  setOperationAction(ISD::SRA,    MVT::i32, Expand);
  setOperationAction(ISD::SHL,    MVT::i8,  Custom);
  setOperationAction(ISD::SHL,    MVT::i16, Expand);
  setOperationAction(ISD::SHL,    MVT::i32, Expand);
  setOperationAction(ISD::SRL,    MVT::i8,  Custom);
  setOperationAction(ISD::SRL,    MVT::i16, Expand);
  setOperationAction(ISD::SRL,    MVT::i32, Expand);

  // PIC16 does not support shift parts
  setOperationAction(ISD::SRA_PARTS,    MVT::i8,  Expand);
  setOperationAction(ISD::SRA_PARTS,    MVT::i16, Expand);
  setOperationAction(ISD::SRA_PARTS,    MVT::i32, Expand);
  setOperationAction(ISD::SHL_PARTS,    MVT::i8, Expand);
  setOperationAction(ISD::SHL_PARTS,    MVT::i16, Expand);
  setOperationAction(ISD::SHL_PARTS,    MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS,    MVT::i8, Expand);
  setOperationAction(ISD::SRL_PARTS,    MVT::i16, Expand);
  setOperationAction(ISD::SRL_PARTS,    MVT::i32, Expand);


  // PIC16 does not have a SETCC, expand it to SELECT_CC.
  setOperationAction(ISD::SETCC,  MVT::i8, Expand);
  setOperationAction(ISD::SELECT,  MVT::i8, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  setOperationAction(ISD::SELECT_CC,  MVT::i8, Custom);
  setOperationAction(ISD::BR_CC,  MVT::i8, Custom);

  //setOperationAction(ISD::TRUNCATE, MVT::i16, Custom);
  setTruncStoreAction(MVT::i16,   MVT::i8,  Custom);

  // Now deduce the information based on the above mentioned 
  // actions
  computeRegisterProperties();
}

static void PopulateResults(SDValue N, SmallVectorImpl<SDValue>&Results) {
  if (N.getOpcode() == ISD::MERGE_VALUES) {
    int NumResults = N.getNumOperands();
    for( int i = 0; i < NumResults; i++)
      Results.push_back(N.getOperand(i));
  }
  else
    Results.push_back(N);
}

MVT PIC16TargetLowering::getSetCCResultType(MVT ValType) const {
  return MVT::i8;
}


void 
PIC16TargetLowering::setPIC16LibcallName(PIC16ISD::PIC16Libcall Call,
                                         const char *Name) {
 PIC16LibcallNames[Call] = Name; 
}

const char *
PIC16TargetLowering::getPIC16LibcallName(PIC16ISD::PIC16Libcall Call) {
 return PIC16LibcallNames[Call];
}

SDValue
PIC16TargetLowering::MakePIC16Libcall(PIC16ISD::PIC16Libcall Call,
                                      MVT RetVT, const SDValue *Ops,
                                      unsigned NumOps, bool isSigned,
                                      SelectionDAG &DAG, DebugLoc dl) {

 TargetLowering::ArgListTy Args;
 Args.reserve(NumOps);

 TargetLowering::ArgListEntry Entry;
 for (unsigned i = 0; i != NumOps; ++i) {
   Entry.Node = Ops[i];
   Entry.Ty = Entry.Node.getValueType().getTypeForMVT();
   Entry.isSExt = isSigned;
   Entry.isZExt = !isSigned;
   Args.push_back(Entry);
 }
 SDValue Callee = DAG.getExternalSymbol(getPIC16LibcallName(Call), MVT::i8);

  const Type *RetTy = RetVT.getTypeForMVT();
  std::pair<SDValue,SDValue> CallInfo = 
     LowerCallTo(DAG.getEntryNode(), RetTy, isSigned, !isSigned, false,
                     false, CallingConv::C, false, Callee, Args, DAG, dl);

  return CallInfo.first;
}

SDValue
PIC16TargetLowering::getCurrentFrame(SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string FuncName = Func->getName();

  // this is causing memory waste
  // because for every call new memory will be allocated
  char *tmpName = new char [strlen(FuncName.c_str()) +  6];
  sprintf(tmpName, "%s.tmp", FuncName.c_str());

  // if the external symbol of the same name already exists then
  // it will not create the new one.
  return DAG.getTargetExternalSymbol(tmpName, MVT::i8);
}

void 
PIC16TargetLowering::getCurrentFrameIndex(SelectionDAG &DAG, SDValue &ES, 
                                        unsigned SlotSize, int &FromFI) {
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string FuncName = Func->getName();

  // this is causing memory waste
  // because for every call new memory will be allocated
  char *tmpName = new char [strlen(FuncName.c_str()) +  6];
  sprintf(tmpName, "%s.tmp", FuncName.c_str());

  // if the external symbol of the same name already exists then
  // it will not create the new one.
  ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);

  // Alignment is always 1
  //FromFI = MF.getFrameInfo()->CreateStackObject(SlotSize, 1);
  FromFI = MF.getFrameInfo()->CreateStackObject(1, 1);
  int FI;
  for(unsigned i=1;i<SlotSize; ++i) {
    FI = MF.getFrameInfo()->CreateStackObject(1, 1);
  }
}

const char *PIC16TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default:                         return NULL;
  case PIC16ISD::Lo:               return "PIC16ISD::Lo";
  case PIC16ISD::Hi:               return "PIC16ISD::Hi";
  case PIC16ISD::MTLO:             return "PIC16ISD::MTLO";
  case PIC16ISD::MTHI:             return "PIC16ISD::MTHI";
  case PIC16ISD::Banksel:          return "PIC16ISD::Banksel";
  case PIC16ISD::PIC16Load:        return "PIC16ISD::PIC16Load";
  case PIC16ISD::PIC16LdWF:        return "PIC16ISD::PIC16LdWF";
  case PIC16ISD::PIC16Store:       return "PIC16ISD::PIC16Store";
  case PIC16ISD::PIC16StWF:        return "PIC16ISD::PIC16StWF";
  case PIC16ISD::BCF:              return "PIC16ISD::BCF";
  case PIC16ISD::LSLF:             return "PIC16ISD::LSLF";
  case PIC16ISD::LRLF:             return "PIC16ISD::LRLF";
  case PIC16ISD::RLF:              return "PIC16ISD::RLF";
  case PIC16ISD::RRF:              return "PIC16ISD::RRF";
  case PIC16ISD::CALL:             return "PIC16ISD::CALL";
  case PIC16ISD::SUBCC:            return "PIC16ISD::SUBCC";
  case PIC16ISD::SELECT_ICC:       return "PIC16ISD::SELECT_ICC";
  case PIC16ISD::BRCOND:           return "PIC16ISD::BRCOND";
  case PIC16ISD::Dummy:            return "PIC16ISD::Dummy";
  }
}

void PIC16TargetLowering::ReplaceNodeResults(SDNode *N,
                                             SmallVectorImpl<SDValue>&Results,
                                             SelectionDAG &DAG) {

  switch (N->getOpcode()) {
    case ISD::GlobalAddress:
      Results.push_back(ExpandGlobalAddress(N, DAG));
      return;
    case ISD::ExternalSymbol:
      Results.push_back(ExpandExternalSymbol(N, DAG));
      return;
    case ISD::STORE:
      Results.push_back(ExpandStore(N, DAG));
      return;
    case ISD::LOAD:
      PopulateResults(ExpandLoad(N, DAG), Results);
      return;
    case ISD::ADD:
      // Results.push_back(ExpandAdd(N, DAG));
      return;
    case ISD::FrameIndex:
      Results.push_back(ExpandFrameIndex(N, DAG));
      return;
    default:
      assert (0 && "not implemented");
      return;
  }
}

SDValue PIC16TargetLowering::ExpandFrameIndex(SDNode *N, SelectionDAG &DAG) {

  // Currently handling FrameIndex of size MVT::i16 only
  // One example of this scenario is when return value is written on
  // FrameIndex#0

  if (N->getValueType(0) != MVT::i16)
    return SDValue();

  // Expand the FrameIndex into ExternalSymbol and a Constant node
  // The constant will represent the frame index number
  // Get the current function frame
  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string Name = Func->getName();

  FrameIndexSDNode *FR = dyn_cast<FrameIndexSDNode>(SDValue(N,0));
  int Index = FR->getIndex();

  SDValue FI[2];
  FI[0] = DAG.getTargetFrameIndex(Index, MVT::i8);
  FI[1] = DAG.getTargetFrameIndex(Index + 1, MVT::i8);
  return DAG.getNode(ISD::BUILD_PAIR, N->getValueType(0), FI[0], FI[1]);
}


SDValue PIC16TargetLowering::ExpandStore(SDNode *N, SelectionDAG &DAG) { 
  StoreSDNode *St = cast<StoreSDNode>(N);
  SDValue Chain = St->getChain();
  SDValue Src = St->getValue();
  SDValue Ptr = St->getBasePtr();
  MVT ValueType = Src.getValueType();
  unsigned StoreOffset = 0;

  SDValue PtrLo, PtrHi;
  LegalizeAddress(Ptr, DAG, PtrLo, PtrHi, StoreOffset);
 
  if (ValueType == MVT::i8) {
    return DAG.getNode (PIC16ISD::PIC16Store, MVT::Other, Chain, Src,
                        PtrLo, PtrHi, 
                        DAG.getConstant (0 + StoreOffset, MVT::i8));
  }
  else if (ValueType == MVT::i16) {
    // Get the Lo and Hi parts from MERGE_VALUE or BUILD_PAIR.
    SDValue SrcLo, SrcHi;
    GetExpandedParts(Src, DAG, SrcLo, SrcHi);
    SDValue ChainLo = Chain, ChainHi = Chain;
    if (Chain.getOpcode() == ISD::TokenFactor) {
      ChainLo = Chain.getOperand(0);
      ChainHi = Chain.getOperand(1);
    }
    SDValue Store1 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other,
                                 ChainLo,
                                 SrcLo, PtrLo, PtrHi,
                                 DAG.getConstant (0 + StoreOffset, MVT::i8));

    SDValue Store2 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainHi, 
                                 SrcHi, PtrLo, PtrHi,
                                 DAG.getConstant (1 + StoreOffset, MVT::i8));

    return DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(Store1),
                       getChain(Store2));
  }
  else if (ValueType == MVT::i32) {
    // Get the Lo and Hi parts from MERGE_VALUE or BUILD_PAIR.
    SDValue SrcLo, SrcHi;
    GetExpandedParts(Src, DAG, SrcLo, SrcHi);

    // Get the expanded parts of each of SrcLo and SrcHi.
    SDValue SrcLo1, SrcLo2, SrcHi1, SrcHi2;
    GetExpandedParts(SrcLo, DAG, SrcLo1, SrcLo2);
    GetExpandedParts(SrcHi, DAG, SrcHi1, SrcHi2);

    SDValue ChainLo = Chain, ChainHi = Chain;
    if (Chain.getOpcode() == ISD::TokenFactor) {  
      ChainLo = Chain.getOperand(0);
      ChainHi = Chain.getOperand(1);
    }
    SDValue ChainLo1 = ChainLo, ChainLo2 = ChainLo, ChainHi1 = ChainHi,
            ChainHi2 = ChainHi;
    if (ChainLo.getOpcode() == ISD::TokenFactor) {
      ChainLo1 = ChainLo.getOperand(0);
      ChainLo2 = ChainLo.getOperand(1);
    }
    if (ChainHi.getOpcode() == ISD::TokenFactor) {
      ChainHi1 = ChainHi.getOperand(0);
      ChainHi2 = ChainHi.getOperand(1);
    }
    SDValue Store1 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other,
                                 ChainLo1,
                                 SrcLo1, PtrLo, PtrHi,
                                 DAG.getConstant (0 + StoreOffset, MVT::i8));

    SDValue Store2 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainLo2,
                                 SrcLo2, PtrLo, PtrHi,
                                 DAG.getConstant (1 + StoreOffset, MVT::i8));

    SDValue Store3 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainHi1,
                                 SrcHi1, PtrLo, PtrHi,
                                 DAG.getConstant (2 + StoreOffset, MVT::i8));

    SDValue Store4 = DAG.getNode(PIC16ISD::PIC16Store, MVT::Other, ChainHi2,
                                 SrcHi2, PtrLo, PtrHi,
                                 DAG.getConstant (3 + StoreOffset, MVT::i8));

    SDValue RetLo =  DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(Store1),
                                 getChain(Store2));
    SDValue RetHi =  DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(Store3),
                                getChain(Store4));
    return  DAG.getNode(ISD::TokenFactor, MVT::Other, RetLo, RetHi);

  }
  else {
    assert (0 && "value type not supported");
    return SDValue();
  }
}

SDValue PIC16TargetLowering::ExpandExternalSymbol(SDNode *N, SelectionDAG &DAG)
{
  ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(SDValue(N, 0));

  SDValue TES = DAG.getTargetExternalSymbol(ES->getSymbol(), MVT::i8);

  SDValue Lo = DAG.getNode(PIC16ISD::Lo, MVT::i8, TES);
  SDValue Hi = DAG.getNode(PIC16ISD::Hi, MVT::i8, TES);

  return DAG.getNode(ISD::BUILD_PAIR, MVT::i16, Lo, Hi);
}

// ExpandGlobalAddress - 
SDValue PIC16TargetLowering::ExpandGlobalAddress(SDNode *N, SelectionDAG &DAG) {
  GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(SDValue(N, 0));
  
  SDValue TGA = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i8,
                                           G->getOffset());

  SDValue Lo = DAG.getNode(PIC16ISD::Lo, MVT::i8, TGA);
  SDValue Hi = DAG.getNode(PIC16ISD::Hi, MVT::i8, TGA);

  return DAG.getNode(ISD::BUILD_PAIR, MVT::i16, Lo, Hi);
}

bool PIC16TargetLowering::isDirectAddress(const SDValue &Op) {
  assert (Op.getNode() != NULL && "Can't operate on NULL SDNode!!");

  if (Op.getOpcode() == ISD::BUILD_PAIR) {
   if (Op.getOperand(0).getOpcode() == PIC16ISD::Lo) 
     return true;
  }
  return false;
}

// Return true if DirectAddress is in ROM_SPACE
bool PIC16TargetLowering::isRomAddress(const SDValue &Op) {

  // RomAddress is a GlobalAddress in ROM_SPACE_
  // If the Op is not a GlobalAddress return NULL without checking
  // anything further.
  if (!isDirectAddress(Op))
    return false; 

  // Its a GlobalAddress.
  // It is BUILD_PAIR((PIC16Lo TGA), (PIC16Hi TGA)) and Op is BUILD_PAIR
  SDValue TGA = Op.getOperand(0).getOperand(0);
  GlobalAddressSDNode *GSDN = dyn_cast<GlobalAddressSDNode>(TGA);
  const Type *ValueType = GSDN->getGlobal()->getType();

  if (!isa<PointerType>(ValueType)) {
    assert(0 && "TGA must be of a PointerType");
  }

  int AddrSpace = dyn_cast<PointerType>(ValueType)->getAddressSpace();
  if (AddrSpace == PIC16ISD::ROM_SPACE)
    return true;

  // Any other address space return it false
  return false;
}

// Extract the out flag 
SDValue PIC16TargetLowering::getOutFlag(SDValue &Op) {
  SDValue Flag = Op.getValue(Op.getNode()->getNumValues() - 1);

  assert (Flag.getValueType() == MVT::Flag && "Node does not have an out Flag");

  return Flag;
}

// To extract chain value from the SDValue Nodes
// This function will help to maintain the chain extracting
// code at one place. In case of any change in future it will
// help maintain the code.
SDValue PIC16TargetLowering::getChain(SDValue &Op) { 
  SDValue Chain = Op.getValue(Op.getNode()->getNumValues() - 1);

  // If the last value returned in Flag then the chain is
  // second last value returned.
  if (Chain.getValueType() == MVT::Flag)
    Chain = Op.getValue(Op.getNode()->getNumValues() - 2);
  
  // All nodes may not produce a chain. Therefore following assert
  // verifies that the node is returning a chain only.
  assert (Chain.getValueType() == MVT::Other && "Node does not have a chain");

  return Chain;
}

void PIC16TargetLowering::GetExpandedParts(SDValue Op, SelectionDAG &DAG,

                                           SDValue &Lo, SDValue &Hi) {  
  SDNode *N = Op.getNode();
  MVT NewVT;
  std::vector<SDValue> Opers;
  NewVT = getTypeToTransformTo(N->getValueType(0));

  // extract the lo component
  Opers.push_back(Op);
  Opers.push_back(DAG.getConstant(0,MVT::i8));
  Lo = DAG.getNode(ISD::EXTRACT_ELEMENT,NewVT,&Opers[0],Opers.size());
  // extract the hi component
  Opers.clear();
  Opers.push_back(Op);
  Opers.push_back(DAG.getConstant(1,MVT::i8));
  Hi = DAG.getNode(ISD::EXTRACT_ELEMENT,NewVT,&Opers[0],Opers.size());
}

// Legalize FrameIndex into ExternalSymbol and offset.
void 
PIC16TargetLowering::LegalizeFrameIndex(SDValue Op, SelectionDAG &DAG,
                                        SDValue &ES, int &Offset) {

  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string Name = Func->getName();

  char *tmpName = new char [strlen(Name.c_str()) +  8];
  sprintf(tmpName, "%s.args", Name.c_str());
  ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);
  FrameIndexSDNode *FR = dyn_cast<FrameIndexSDNode>(Op);
  Offset = FR->getIndex();

  return;
}

// This function legalizes the PIC16 Addresses. If the Pointer is  
//  -- Direct address variable residing 
//     --> then a Banksel for that variable will be created.
//  -- Rom variable            
//     --> then it will be treated as an indirect address.
//  -- Indirect address 
//     --> then the address will be loaded into FSR
//  -- ADD with constant operand
//     --> then constant operand of ADD will be returned as Offset
//         and non-constant operand of ADD will be treated as pointer.
// Returns the high and lo part of the address, and the offset(in case of ADD).

void PIC16TargetLowering:: LegalizeAddress(SDValue Ptr, SelectionDAG &DAG, 
                                           SDValue &Lo, SDValue &Hi,
                                           unsigned &Offset) {

  // Offset, by default, should be 0
  Offset = 0;

  // If the pointer is ADD with constant,
  // return the constant value as the offset  
  if (Ptr.getOpcode() == ISD::ADD) {
    SDValue OperLeft = Ptr.getOperand(0);
    SDValue OperRight = Ptr.getOperand(1);
    if (OperLeft.getOpcode() == ISD::Constant) {
      Offset = dyn_cast<ConstantSDNode>(OperLeft)->getZExtValue();
      Ptr = OperRight;
    } else if (OperRight.getOpcode() == ISD::Constant) {
      Offset = dyn_cast<ConstantSDNode>(OperRight)->getZExtValue();
      Ptr = OperLeft;
    }
  }

  // If the pointer is Type i8 and an external symbol
  // then treat it as direct address.
  // One example for such case is storing and loading
  // from function frame during a call
  if (Ptr.getValueType() == MVT::i8) {
    switch (Ptr.getOpcode()) {
    case ISD::TargetExternalSymbol:
      Lo = Ptr;
      Hi = DAG.getConstant(1, MVT::i8);
      return;
    }
  }

  if (Ptr.getOpcode() == ISD::BUILD_PAIR && 
      Ptr.getOperand(0).getOpcode() == ISD::TargetFrameIndex) {

    int FrameOffset;
    LegalizeFrameIndex(Ptr.getOperand(0), DAG, Lo, FrameOffset);
    Hi = DAG.getConstant(1, MVT::i8);
    Offset += FrameOffset; 
    return;
  }

  if (isDirectAddress(Ptr) && !isRomAddress(Ptr)) {
    // Direct addressing case for RAM variables. The Hi part is constant
    // and the Lo part is the TGA itself.
    Lo = Ptr.getOperand(0).getOperand(0);

    // For direct addresses Hi is a constant. Value 1 for the constant
    // signifies that banksel needs to generated for it. Value 0 for
    // the constant signifies that banksel does not need to be generated 
    // for it. Mark it as 1 now and optimize later. 
    Hi = DAG.getConstant(1, MVT::i8);
    return; 
  }

  // Indirect addresses. Get the hi and lo parts of ptr. 
  GetExpandedParts(Ptr, DAG, Lo, Hi);

  // Put the hi and lo parts into FSR.
  Lo = DAG.getNode(PIC16ISD::MTLO, MVT::i8, Lo);
  Hi = DAG.getNode(PIC16ISD::MTHI, MVT::i8, Hi);

  return;
}

//SDValue PIC16TargetLowering::ExpandAdd(SDNode *N, SelectionDAG &DAG) {
  //SDValue OperLeft = N->getOperand(0);
  //SDValue OperRight = N->getOperand(1);

  //if((OperLeft.getOpcode() == ISD::Constant) ||
     //(OperRight.getOpcode() == ISD::Constant)) {
    //return SDValue();
  //}

  // These case are yet to be handled
  //return SDValue();
//}

SDValue PIC16TargetLowering::ExpandLoad(SDNode *N, SelectionDAG &DAG) {
  LoadSDNode *LD = dyn_cast<LoadSDNode>(SDValue(N, 0));
  SDValue Chain = LD->getChain();
  SDValue Ptr = LD->getBasePtr();

  SDValue Load, Offset;
  SDVTList Tys; 
  MVT VT, NewVT;
  SDValue PtrLo, PtrHi;
  unsigned LoadOffset;

  // Legalize direct/indirect addresses. This will give the lo and hi parts
  // of the address and the offset.
  LegalizeAddress(Ptr, DAG, PtrLo, PtrHi, LoadOffset);

  // Load from the pointer (direct address or FSR) 
  VT = N->getValueType(0);
  unsigned NumLoads = VT.getSizeInBits() / 8; 
  std::vector<SDValue> PICLoads;
  unsigned iter;
  MVT MemVT = LD->getMemoryVT();
  if(ISD::isNON_EXTLoad(N)) {
    for (iter=0; iter<NumLoads ; ++iter) {
      // Add the pointer offset if any
      Offset = DAG.getConstant(iter + LoadOffset, MVT::i8);
      Tys = DAG.getVTList(MVT::i8, MVT::Other); 
      Load = DAG.getNode(PIC16ISD::PIC16Load, Tys, Chain, PtrLo, PtrHi,
                         Offset); 
      PICLoads.push_back(Load);
    }
  } else {
    // If it is extended load then use PIC16Load for Memory Bytes
    // and for all extended bytes perform action based on type of
    // extention - i.e. SignExtendedLoad or ZeroExtendedLoad

    
    // For extended loads this is the memory value type
    // i.e. without any extension
    MVT MemVT = LD->getMemoryVT();
    unsigned MemBytes = MemVT.getSizeInBits() / 8;
    unsigned ExtdBytes = VT.getSizeInBits() / 8;
    Offset = DAG.getConstant(LoadOffset, MVT::i8);

    Tys = DAG.getVTList(MVT::i8, MVT::Other); 
    // For MemBytes generate PIC16Load with proper offset
    for (iter=0; iter<MemBytes; ++iter) {
      // Add the pointer offset if any
      Offset = DAG.getConstant(iter + LoadOffset, MVT::i8);
      Load = DAG.getNode(PIC16ISD::PIC16Load, Tys, Chain, PtrLo, PtrHi,
                         Offset); 
      PICLoads.push_back(Load);
    }

    // For SignExtendedLoad
    if (ISD::isSEXTLoad(N)) {
      // For all ExtdBytes use the Right Shifted(Arithmetic) Value of the 
      // highest MemByte
      SDValue SRA = DAG.getNode(ISD::SRA, MVT::i8, Load, 
                                DAG.getConstant(7, MVT::i8));
      for (iter=MemBytes; iter<ExtdBytes; ++iter) { 
        PICLoads.push_back(SRA);
      }
    } else if (ISD::isZEXTLoad(N)) {
      // ZeroExtendedLoad -- For all ExtdBytes use constant 0
      SDValue ConstZero = DAG.getConstant(0, MVT::i8);
      for (iter=MemBytes; iter<ExtdBytes; ++iter) { 
        PICLoads.push_back(ConstZero);
      }
    }
  }
  SDValue BP;

  if (VT == MVT::i8) {
    // Operand of Load is illegal -- Load itself is legal
    return PICLoads[0];
  }
  else if (VT == MVT::i16) {
    BP = DAG.getNode(ISD::BUILD_PAIR, VT, PICLoads[0], PICLoads[1]);
    if (MemVT == MVT::i8)
      Chain = getChain(PICLoads[0]);
    else
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(PICLoads[0]),
                          getChain(PICLoads[1]));
  } else if (VT == MVT::i32) {
    SDValue BPs[2];
    BPs[0] = DAG.getNode(ISD::BUILD_PAIR, MVT::i16, PICLoads[0], PICLoads[1]);
    BPs[1] = DAG.getNode(ISD::BUILD_PAIR, MVT::i16, PICLoads[2], PICLoads[3]);
    BP = DAG.getNode(ISD::BUILD_PAIR, VT, BPs[0], BPs[1]);
    if (MemVT == MVT::i8)
      Chain = getChain(PICLoads[0]);
    else if (MemVT == MVT::i16)
      Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, getChain(PICLoads[0]),
                          getChain(PICLoads[1]));
    else {
      SDValue Chains[2];
      Chains[0] = DAG.getNode(ISD::TokenFactor, MVT::Other,
                              getChain(PICLoads[0]), getChain(PICLoads[1]));
      Chains[1] = DAG.getNode(ISD::TokenFactor, MVT::Other,
                              getChain(PICLoads[2]), getChain(PICLoads[3]));
      Chain =  DAG.getNode(ISD::TokenFactor, MVT::Other, Chains[0], Chains[1]);
    }
  }
  Tys = DAG.getVTList(VT, MVT::Other); 
  return DAG.getNode(ISD::MERGE_VALUES, Tys, BP, Chain);
}

SDValue PIC16TargetLowering::LowerShift(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal shift to lower");
 
  SDNode *N = Op.getNode();
  SDValue Value = N->getOperand(0);
  SDValue Amt = N->getOperand(1);
  PIC16ISD::PIC16Libcall CallCode;
  switch (N->getOpcode()) {
  case ISD::SRA:
    CallCode = PIC16ISD::SRA_I8;
    break;
  case ISD::SHL:
    CallCode = PIC16ISD::SLL_I8;
    break;
  case ISD::SRL:
    CallCode = PIC16ISD::SRL_I8;
    break;
  default:
    assert ( 0 && "This shift is not implemented yet.");
    return SDValue();
  }
  SmallVector<SDValue, 2> Ops(2);
  Ops[0] = Value;
  Ops[1] = Amt;
  SDValue Call = MakePIC16Libcall(CallCode, N->getValueType(0), &Ops[0], 2, 
                                  true, DAG, N->getDebugLoc());
  return Call;
}

void
PIC16TargetLowering::LowerOperationWrapper(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG) {
  SDValue Op = SDValue(N, 0);
  SDValue Res;
  unsigned i;
  switch (Op.getOpcode()) {
    case ISD::FORMAL_ARGUMENTS:
      Res = LowerFORMAL_ARGUMENTS(Op, DAG); break;
    case ISD::LOAD:
      Res = ExpandLoad(Op.getNode(), DAG); break;
    case ISD::CALL:
      Res = LowerCALL(Op, DAG); break;
    default: {
      // All other operations are handled in LowerOperation.
      Res = LowerOperation(Op, DAG);
      if (Res.getNode())
        Results.push_back(Res);
        
      return; 
    }
  }

  N = Res.getNode();
  unsigned NumValues = N->getNumValues(); 
  for (i = 0; i < NumValues ; i++) {
    Results.push_back(SDValue(N, i)); 
  }
}

SDValue PIC16TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
    case ISD::FORMAL_ARGUMENTS:
      return LowerFORMAL_ARGUMENTS(Op, DAG);
    case ISD::ADD:
    case ISD::ADDC:
    case ISD::ADDE:
      return LowerADD(Op, DAG);
    case ISD::SUB:
    case ISD::SUBC:
    case ISD::SUBE:
      return LowerSUB(Op, DAG);
    case ISD::LOAD:
      return ExpandLoad(Op.getNode(), DAG);
    case ISD::STORE:
      return ExpandStore(Op.getNode(), DAG);
    case ISD::SHL:
    case ISD::SRA:
    case ISD::SRL:
      return LowerShift(Op, DAG);
    case ISD::OR:
    case ISD::AND:
    case ISD::XOR:
      return LowerBinOp(Op, DAG);
    case ISD::CALL:
      return LowerCALL(Op, DAG);
    case ISD::RET:
      return LowerRET(Op, DAG);
    case ISD::BR_CC:
      return LowerBR_CC(Op, DAG);
    case ISD::SELECT_CC:
      return LowerSELECT_CC(Op, DAG);
  }
  return SDValue();
}

SDValue PIC16TargetLowering::ConvertToMemOperand(SDValue Op,
                                                 SelectionDAG &DAG) {

  assert (Op.getValueType() == MVT::i8 
          && "illegal value type to store on stack.");

  MachineFunction &MF = DAG.getMachineFunction();
  const Function *Func = MF.getFunction();
  const std::string FuncName = Func->getName();

  char *tmpName = new char [strlen(FuncName.c_str()) +  6];

  // Put the value on stack.
  // Get a stack slot index and convert to es.
  int FI = MF.getFrameInfo()->CreateStackObject(1, 1);
  sprintf(tmpName, "%s.tmp", FuncName.c_str());
  SDValue ES = DAG.getTargetExternalSymbol(tmpName, MVT::i8);

  // Store the value to ES.
  SDValue Store = DAG.getNode (PIC16ISD::PIC16Store, MVT::Other,
                               DAG.getEntryNode(),
                               Op, ES, 
                               DAG.getConstant (1, MVT::i8), // Banksel.
                               DAG.getConstant (FI, MVT::i8));

  // Load the value from ES.
  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other);
  SDValue Load = DAG.getNode(PIC16ISD::PIC16Load, Tys, Store,
                             ES, DAG.getConstant (1, MVT::i8),
                             DAG.getConstant (FI, MVT::i8));
    
  return Load.getValue(0);
}
          
SDValue
PIC16TargetLowering::LowerCallArguments(SDValue Op, SDValue Chain,
                                        SDValue FrameAddress, 
                                        SDValue InFlag,
                                        SelectionDAG &DAG) {
  CallSDNode *TheCall = dyn_cast<CallSDNode>(Op);
  unsigned NumOps = TheCall->getNumArgs();
  std::string Name;
  SDValue Arg, StoreAt;
  MVT ArgVT;
  unsigned Size=0;
  unsigned ArgCount=0;


  // FIXME: This portion of code currently assumes only
  // primitive types being passed as arguments.

  // Legalize the address before use
  SDValue PtrLo, PtrHi;
  unsigned AddressOffset;
  int StoreOffset = 0;
  LegalizeAddress(FrameAddress, DAG, PtrLo, PtrHi, AddressOffset);
  SDValue StoreRet;

  std::vector<SDValue> Ops;
  SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
  for (unsigned i=ArgCount, Offset = 0; i<NumOps; i++) {
    // Get the argument
    Arg = TheCall->getArg(i);

    StoreOffset = (Offset + AddressOffset);
   
    // Store the argument on frame

    Ops.clear();
    Ops.push_back(Chain);
    Ops.push_back(Arg.getValue(0));
    Ops.push_back(PtrLo);
    Ops.push_back(PtrHi);
    Ops.push_back(DAG.getConstant(StoreOffset, MVT::i8));
    Ops.push_back(InFlag);

    StoreRet = DAG.getNode (PIC16ISD::PIC16StWF, Tys, &Ops[0], Ops.size());

    Chain = getChain(StoreRet);
    InFlag = getOutFlag(StoreRet);

    // Update the frame offset to be used for next argument
    ArgVT = Arg.getValueType();
    Size = ArgVT.getSizeInBits();
    Size = Size/8;    // Calculate size in bytes
    Offset += Size;   // Increase the frame offset
  }
  return Chain;
}

SDValue
PIC16TargetLowering::LowerCallReturn(SDValue Op, SDValue Chain,
                                     SDValue FrameAddress,
                                     SDValue InFlag,
                                     SelectionDAG &DAG) {
  CallSDNode *TheCall = dyn_cast<CallSDNode>(Op);
  // Currently handling primitive types only. They will come in
  // i8 parts
  unsigned RetVals = TheCall->getNumRetVals();
  
  std::vector<SDValue> ResultVals;

  // Return immediately if the return type is void
  if (RetVals == 0)
    return Chain;

  // Call has something to return
  
  // Legalize the address before use
  SDValue LdLo, LdHi;
  unsigned LdOffset;
  LegalizeAddress(FrameAddress, DAG, LdLo, LdHi, LdOffset);

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Other, MVT::Flag);
  SDValue LoadRet;
 
  for(unsigned i=0, Offset=0;i<RetVals;i++) {

    LoadRet = DAG.getNode(PIC16ISD::PIC16LdWF, Tys, Chain, LdLo, LdHi,
                          DAG.getConstant(LdOffset + Offset, MVT::i8),
                          InFlag);

    InFlag = getOutFlag(LoadRet);

    Chain = getChain(LoadRet);
    Offset++;
    ResultVals.push_back(LoadRet);
  }

  // To return use MERGE_VALUES
  ResultVals.push_back(Chain);
  SDValue Res = DAG.getMergeValues(&ResultVals[0], ResultVals.size());
  return Res;
}

SDValue PIC16TargetLowering::LowerRET(SDValue Op, SelectionDAG &DAG) {
 //int NumOps = Op.getNode()->getNumOperands();

 // For default cases LLVM returns the value on the function frame 
 // So let LLVM do this for all the cases other than character
 return Op; 
}

SDValue PIC16TargetLowering::LowerCALL(SDValue Op, SelectionDAG &DAG) {
    CallSDNode *TheCall = dyn_cast<CallSDNode>(Op);
    SDValue Chain = TheCall->getChain();
    SDValue Callee = TheCall->getCallee();
    unsigned i =0;
    if (Callee.getValueType() == MVT::i16 &&
      Callee.getOpcode() == ISD::BUILD_PAIR) {
      // It has come from TypeLegalizer for lowering

      Callee = Callee.getOperand(0).getOperand(0);

      std::vector<SDValue> Ops;
      Ops.push_back(Chain);
      Ops.push_back(Callee);

      // Add the call arguments and their flags
      unsigned NumArgs = TheCall->getNumArgs();
      for(i=0;i<NumArgs;i++) { 
        Ops.push_back(TheCall->getArg(i));
        Ops.push_back(TheCall->getArgFlagsVal(i));
      }

      std::vector<MVT> NodeTys;
      unsigned NumRets = TheCall->getNumRetVals();
      for(i=0;i<NumRets;i++)
        NodeTys.push_back(TheCall->getRetValType(i));

      // Return a Chain as well
      NodeTys.push_back(MVT::Other);

      SDVTList VTs = DAG.getVTList(&NodeTys[0], NodeTys.size());
      SDValue NewCall = 
              DAG.getCall(TheCall->getCallingConv(), TheCall->isVarArg(), 
                          TheCall->isTailCall(), TheCall->isInreg(), VTs, 
                          &Ops[0], Ops.size());

      return NewCall;
    }
    
    SDValue ZeroOperand = DAG.getConstant(0, MVT::i8);

    // Start the call sequence.
    // Carring the Constant 0 along the CALLSEQSTART
    // because there is nothing else to carry.
    SDValue SeqStart  = DAG.getCALLSEQ_START(Chain, ZeroOperand);
    Chain = getChain(SeqStart);

    // For any direct call - callee will be GlobalAddressNode or
    // ExternalSymbol

    // Considering the GlobalAddressNode case here.
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
      GlobalValue *GV = G->getGlobal();
      Callee = DAG.getTargetGlobalAddress(GV, MVT::i8);
    }

    // Considering the ExternalSymbol case here
    if (ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Callee)) {
      Callee = DAG.getTargetExternalSymbol(ES->getSymbol(), MVT::i8); 
    }

    SDValue OperFlag = getOutFlag(Chain); // To manage the data dependency

    std::string Name;

    // Considering GlobalAddress here
    if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
       Name = G->getGlobal()->getName();

    // Considering ExternalSymbol here
    if (ExternalSymbolSDNode *ES = dyn_cast<ExternalSymbolSDNode>(Callee))
       Name = ES->getSymbol();

    char *argFrame = new char [strlen(Name.c_str()) +  8];
    sprintf(argFrame, "%s.args", Name.c_str());
    SDValue ArgLabel = DAG.getTargetExternalSymbol(argFrame, MVT::i8);

    char *retName = new char [strlen(Name.c_str()) +  8];
    sprintf(retName, "%s.retval", Name.c_str());
    SDValue RetLabel = DAG.getTargetExternalSymbol(retName, MVT::i8);

    // Pass the argument to function before making the call.
    SDValue CallArgs = LowerCallArguments(Op, Chain, ArgLabel, OperFlag, DAG);
    Chain = getChain(CallArgs);
    OperFlag = getOutFlag(CallArgs);

    SDVTList Tys = DAG.getVTList(MVT::Other, MVT::Flag);
    SDValue PICCall = DAG.getNode(PIC16ISD::CALL, Tys, Chain, Callee,
                                  OperFlag);
    Chain = getChain(PICCall);
    OperFlag = getOutFlag(PICCall);


    // Carrying the Constant 0 along the CALLSEQSTART
    // because there is nothing else to carry.
    SDValue SeqEnd = DAG.getCALLSEQ_END(Chain, ZeroOperand, ZeroOperand,
                                        OperFlag);
    Chain = getChain(SeqEnd);
    OperFlag = getOutFlag(SeqEnd);

    // Lower the return value reading after the call.
    return LowerCallReturn(Op, Chain, RetLabel, OperFlag, DAG);
}

bool PIC16TargetLowering::isDirectLoad(const SDValue Op) {
  if (Op.getOpcode() == PIC16ISD::PIC16Load)
    if (Op.getOperand(1).getOpcode() == ISD::TargetGlobalAddress
     || Op.getOperand(1).getOpcode() == ISD::TargetExternalSymbol)
      return true;
  return false;
}

bool PIC16TargetLowering::NeedToConvertToMemOp(SDValue Op, unsigned &MemOp) {
  // Return false if one of the operands is already a direct
  // load and that operand has only one use.
  if (Op.getOperand(0).getOpcode() == ISD::Constant ||
      Op.getOperand(1).getOpcode() == ISD::Constant)
    return false;    
  if (isDirectLoad(Op.getOperand(0))) {
    if (Op.getOperand(0).hasOneUse())
      return false;
    else 
      MemOp = 0;
  }
  if (isDirectLoad(Op.getOperand(1))) {
    if (Op.getOperand(1).hasOneUse())
      return false;
    else 
      MemOp = 1; 
  }
  return true;
}  

SDValue PIC16TargetLowering:: LowerBinOp(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal Op to lower");
  unsigned MemOp = 1;
  if (NeedToConvertToMemOp(Op, MemOp)) {
    // Put one value on stack.
    SDValue NewVal = ConvertToMemOperand (Op.getOperand(MemOp), DAG);

    return DAG.getNode(Op.getOpcode(), MVT::i8, Op.getOperand(MemOp ^ 1),
    NewVal);
  }
  else {
    return Op;
  }
}

SDValue PIC16TargetLowering:: LowerADD(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal add to lower");
  unsigned MemOp = 1;
  if (NeedToConvertToMemOp(Op, MemOp)) {
    // Put one value on stack.
    SDValue NewVal = ConvertToMemOperand (Op.getOperand(MemOp), DAG);
    
    SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);

    if (Op.getOpcode() == ISD::ADDE)
      return DAG.getNode(Op.getOpcode(), Tys, Op.getOperand(MemOp ^ 1), NewVal, 
                         Op.getOperand(2));
    else
      return DAG.getNode(Op.getOpcode(), Tys, Op.getOperand(MemOp ^ 1), NewVal);
  }
  else if (Op.getOpcode() == ISD::ADD) {
    return Op;
  }
  else {
    return SDValue();
  }
}

SDValue PIC16TargetLowering::LowerSUB(SDValue Op, SelectionDAG &DAG) {
  // We should have handled larger operands in type legalizer itself.
  assert (Op.getValueType() == MVT::i8 && "illegal sub to lower");

  // Nothing to do if the first operand is already a direct load and it has
  // only one use.
  if (isDirectLoad(Op.getOperand(0)) && Op.getOperand(0).hasOneUse())
    return SDValue();

  // Put first operand on stack.
  SDValue NewVal = ConvertToMemOperand (Op.getOperand(0), DAG);

  SDVTList Tys = DAG.getVTList(MVT::i8, MVT::Flag);
  if (Op.getOpcode() == ISD::SUBE)
    return DAG.getNode(Op.getOpcode(), Tys, NewVal, Op.getOperand(1),
                       Op.getOperand(2));
  else
    return DAG.getNode(Op.getOpcode(), Tys, NewVal, Op.getOperand(1));
}

// LowerFORMAL_ARGUMENTS - In Lowering FORMAL ARGUMENTS - MERGE_VALUES nodes
// is returned. MERGE_VALUES nodes number of operands and number of values are
// equal. Therefore to construct MERGE_VALUE node, UNDEF nodes equal to the
// number of arguments of function have been created.

SDValue PIC16TargetLowering:: LowerFORMAL_ARGUMENTS(SDValue Op, 
                                                    SelectionDAG &DAG) {
  SmallVector<SDValue, 8> ArgValues;
  unsigned NumArgs = Op.getNumOperands() - 3;

  // Creating UNDEF nodes to meet the requirement of MERGE_VALUES node.
  for(unsigned i = 0 ; i<NumArgs ; i++) {
    SDValue TempNode = DAG.getNode(ISD::UNDEF, Op.getNode()->getValueType(i));
    ArgValues.push_back(TempNode);
  }

  ArgValues.push_back(Op.getOperand(0));
  return DAG.getNode(ISD::MERGE_VALUES, Op.getNode()->getVTList(), 
                     &ArgValues[0],
                     ArgValues.size()).getValue(Op.getResNo());
}

// Perform DAGCombine of PIC16Load 
SDValue PIC16TargetLowering::
PerformPIC16LoadCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Chain = N->getOperand(0); 
  if (N->hasNUsesOfValue(0, 0)) {
    DAG.ReplaceAllUsesOfValueWith(SDValue(N,1), Chain);
  }
  return SDValue();
}


SDValue PIC16TargetLowering::PerformDAGCombine(SDNode *N, 
                                               DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  case PIC16ISD::PIC16Load:
    return PerformPIC16LoadCombine(N, DCI);
  }
  return SDValue();
}

static PIC16CC::CondCodes IntCCToPIC16CC(ISD::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Unknown condition code!");
  case ISD::SETNE:  return PIC16CC::NE;
  case ISD::SETEQ:  return PIC16CC::EQ;
  case ISD::SETGT:  return PIC16CC::GT;
  case ISD::SETGE:  return PIC16CC::GE;
  case ISD::SETLT:  return PIC16CC::LT;
  case ISD::SETLE:  return PIC16CC::LE;
  case ISD::SETULT: return PIC16CC::ULT;
  case ISD::SETULE: return PIC16CC::LE;
  case ISD::SETUGE: return PIC16CC::GE;
  case ISD::SETUGT: return PIC16CC::UGT;
  }
}

// Look at LHS/RHS/CC and see if they are a lowered setcc instruction.  If so
// set LHS/RHS and SPCC to the LHS/RHS of the setcc and SPCC to the condition.
static void LookThroughSetCC(SDValue &LHS, SDValue &RHS,
                             ISD::CondCode CC, unsigned &SPCC) {
  if (isa<ConstantSDNode>(RHS) &&
      cast<ConstantSDNode>(RHS)->getZExtValue() == 0 &&
      CC == ISD::SETNE &&
      (LHS.getOpcode() == PIC16ISD::SELECT_ICC &&
        LHS.getOperand(3).getOpcode() == PIC16ISD::SUBCC) &&
      isa<ConstantSDNode>(LHS.getOperand(0)) &&
      isa<ConstantSDNode>(LHS.getOperand(1)) &&
      cast<ConstantSDNode>(LHS.getOperand(0))->getZExtValue() == 1 &&
      cast<ConstantSDNode>(LHS.getOperand(1))->getZExtValue() == 0) {
    SDValue CMPCC = LHS.getOperand(3);
    SPCC = cast<ConstantSDNode>(LHS.getOperand(2))->getZExtValue();
    LHS = CMPCC.getOperand(0);
    RHS = CMPCC.getOperand(1);
  }
}

// Returns appropriate CMP insn and corresponding condition code in PIC16CC
SDValue PIC16TargetLowering::getPIC16Cmp(SDValue LHS, SDValue RHS, 
                                         unsigned CC, SDValue &PIC16CC, 
                                         SelectionDAG &DAG) {
  PIC16CC::CondCodes CondCode = (PIC16CC::CondCodes) CC;

  // PIC16 sub is literal - W. So Swap the operands and condition if needed.
  // i.e. a < 12 can be rewritten as 12 > a.
  if (RHS.getOpcode() == ISD::Constant) {

    SDValue Tmp = LHS;
    LHS = RHS;
    RHS = Tmp;

    switch (CondCode) {
    default: break;
    case PIC16CC::LT:
      CondCode = PIC16CC::GT; 
      break;
    case PIC16CC::GT:
      CondCode = PIC16CC::LT; 
      break;
    case PIC16CC::ULT:
      CondCode = PIC16CC::UGT; 
      break;
    case PIC16CC::UGT:
      CondCode = PIC16CC::ULT; 
      break;
    case PIC16CC::GE:
      CondCode = PIC16CC::LE; 
      break;
    case PIC16CC::LE:
      CondCode = PIC16CC::GE;
      break;
    case PIC16CC::ULE:
      CondCode = PIC16CC::UGE;
      break;
    case PIC16CC::UGE:
      CondCode = PIC16CC::ULE;
      break;
    }
  }

  PIC16CC = DAG.getConstant(CondCode, MVT::i8);

  // These are signed comparisons. 
  SDValue Mask = DAG.getConstant(128, MVT::i8);
  if (isSignedComparison(CondCode)) {
    LHS = DAG.getNode (ISD::XOR, MVT::i8, LHS, Mask);
    RHS = DAG.getNode (ISD::XOR, MVT::i8, RHS, Mask); 
  }

  SDVTList VTs = DAG.getVTList (MVT::i8, MVT::Flag);
  // We can use a subtract operation to set the condition codes. But
  // we need to put one operand in memory if required.
  // Nothing to do if the first operand is already a valid type (direct load 
  // for subwf and literal for sublw) and it is used by this operation only. 
  if ((LHS.getOpcode() == ISD::Constant || isDirectLoad(LHS)) 
      && LHS.hasOneUse())
    return DAG.getNode(PIC16ISD::SUBCC, VTs, LHS, RHS);

  // else convert the first operand to mem.
  LHS = ConvertToMemOperand (LHS, DAG);
  return DAG.getNode(PIC16ISD::SUBCC, VTs, LHS, RHS);
}


SDValue PIC16TargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDValue TrueVal = Op.getOperand(2);
  SDValue FalseVal = Op.getOperand(3);
  unsigned ORIGCC = ~0;

  // If this is a select_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  // i.e.
  // A setcc: lhs, rhs, cc is expanded by llvm to 
  // select_cc: result of setcc, 0, 1, 0, setne
  // We can think of it as:
  // select_cc: lhs, rhs, 1, 0, cc
  LookThroughSetCC(LHS, RHS, CC, ORIGCC);
  if (ORIGCC == ~0U) ORIGCC = IntCCToPIC16CC (CC);

  SDValue PIC16CC;
  SDValue Cmp = getPIC16Cmp(LHS, RHS, ORIGCC, PIC16CC, DAG);

  return DAG.getNode (PIC16ISD::SELECT_ICC, TrueVal.getValueType(), TrueVal,
                      FalseVal, PIC16CC, Cmp.getValue(1)); 
}

MachineBasicBlock *
PIC16TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *BB) {
  const TargetInstrInfo &TII = *getTargetMachine().getInstrInfo();
  unsigned CC = (PIC16CC::CondCodes)MI->getOperand(3).getImm();

  // To "insert" a SELECT_CC instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It;

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   [f]bCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *sinkMBB = F->CreateMachineBasicBlock(LLVM_BB);
  BuildMI(BB, TII.get(PIC16::pic16brcond)).addMBB(sinkMBB).addImm(CC);
  F->insert(It, copy0MBB);
  F->insert(It, sinkMBB);

  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  sinkMBB->transferSuccessors(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(sinkMBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to sinkMBB
  BB = copy0MBB;

  // Update machine-CFG edges
  BB->addSuccessor(sinkMBB);

  //  sinkMBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = sinkMBB;
  BuildMI(BB, TII.get(PIC16::PHI), MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);

  F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
  return BB;
}


SDValue PIC16TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);   // LHS of the condition.
  SDValue RHS = Op.getOperand(3);   // RHS of the condition.
  SDValue Dest = Op.getOperand(4);  // BB to jump to
  unsigned ORIGCC = ~0;

  // If this is a br_cc of a "setcc", and if the setcc got lowered into
  // an CMP[IF]CC/SELECT_[IF]CC pair, find the original compared values.
  LookThroughSetCC(LHS, RHS, CC, ORIGCC);
  if (ORIGCC == ~0U) ORIGCC = IntCCToPIC16CC (CC);

  // Get the Compare insn and condition code.
  SDValue PIC16CC;
  SDValue Cmp = getPIC16Cmp(LHS, RHS, ORIGCC, PIC16CC, DAG);

  return DAG.getNode(PIC16ISD::BRCOND, MVT::Other, Chain, Dest, PIC16CC, 
                     Cmp.getValue(1));
}

  
