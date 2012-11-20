//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that NVPTX uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//


#include "NVPTX.h"
#include "NVPTXISelLowering.h"
#include "NVPTXTargetMachine.h"
#include "NVPTXTargetObjectFile.h"
#include "NVPTXUtilities.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Module.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/MCSectionELF.h"
#include <sstream>

#undef DEBUG_TYPE
#define DEBUG_TYPE "nvptx-lower"

using namespace llvm;

static unsigned int uniqueCallSite = 0;

static cl::opt<bool>
RetainVectorOperands("nvptx-codegen-vectors",
     cl::desc("NVPTX Specific: Retain LLVM's vectors and generate PTX vectors"),
                     cl::init(true));

static cl::opt<bool>
sched4reg("nvptx-sched4reg",
          cl::desc("NVPTX Specific: schedule for register pressue"),
          cl::init(false));

// NVPTXTargetLowering Constructor.
NVPTXTargetLowering::NVPTXTargetLowering(NVPTXTargetMachine &TM)
: TargetLowering(TM, new NVPTXTargetObjectFile()),
  nvTM(&TM),
  nvptxSubtarget(TM.getSubtarget<NVPTXSubtarget>()) {

  // always lower memset, memcpy, and memmove intrinsics to load/store
  // instructions, rather
  // then generating calls to memset, mempcy or memmove.
  maxStoresPerMemset = (unsigned)0xFFFFFFFF;
  maxStoresPerMemcpy = (unsigned)0xFFFFFFFF;
  maxStoresPerMemmove = (unsigned)0xFFFFFFFF;

  setBooleanContents(ZeroOrNegativeOneBooleanContent);

  // Jump is Expensive. Don't create extra control flow for 'and', 'or'
  // condition branches.
  setJumpIsExpensive(true);

  // By default, use the Source scheduling
  if (sched4reg)
    setSchedulingPreference(Sched::RegPressure);
  else
    setSchedulingPreference(Sched::Source);

  addRegisterClass(MVT::i1, &NVPTX::Int1RegsRegClass);
  addRegisterClass(MVT::i8, &NVPTX::Int8RegsRegClass);
  addRegisterClass(MVT::i16, &NVPTX::Int16RegsRegClass);
  addRegisterClass(MVT::i32, &NVPTX::Int32RegsRegClass);
  addRegisterClass(MVT::i64, &NVPTX::Int64RegsRegClass);
  addRegisterClass(MVT::f32, &NVPTX::Float32RegsRegClass);
  addRegisterClass(MVT::f64, &NVPTX::Float64RegsRegClass);

  if (RetainVectorOperands) {
    addRegisterClass(MVT::v2f32, &NVPTX::V2F32RegsRegClass);
    addRegisterClass(MVT::v4f32, &NVPTX::V4F32RegsRegClass);
    addRegisterClass(MVT::v2i32, &NVPTX::V2I32RegsRegClass);
    addRegisterClass(MVT::v4i32, &NVPTX::V4I32RegsRegClass);
    addRegisterClass(MVT::v2f64, &NVPTX::V2F64RegsRegClass);
    addRegisterClass(MVT::v2i64, &NVPTX::V2I64RegsRegClass);
    addRegisterClass(MVT::v2i16, &NVPTX::V2I16RegsRegClass);
    addRegisterClass(MVT::v4i16, &NVPTX::V4I16RegsRegClass);
    addRegisterClass(MVT::v2i8, &NVPTX::V2I8RegsRegClass);
    addRegisterClass(MVT::v4i8, &NVPTX::V4I8RegsRegClass);

    setOperationAction(ISD::BUILD_VECTOR, MVT::v4i32  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4f32  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4i16  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v4i8   , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v2i64  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v2f64  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v2i32  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v2f32  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v2i16  , Custom);
    setOperationAction(ISD::BUILD_VECTOR, MVT::v2i8   , Custom);

    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i32  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4f32  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i16  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i8   , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i64  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2f64  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i32  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2f32  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i16  , Custom);
    setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i8   , Custom);
  }

  // Operations not directly supported by NVPTX.
  setOperationAction(ISD::SELECT_CC,         MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,             MVT::Other, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i64, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8 , Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1 , Expand);

  if (nvptxSubtarget.hasROT64()) {
    setOperationAction(ISD::ROTL , MVT::i64, Legal);
    setOperationAction(ISD::ROTR , MVT::i64, Legal);
  }
  else {
    setOperationAction(ISD::ROTL , MVT::i64, Expand);
    setOperationAction(ISD::ROTR , MVT::i64, Expand);
  }
  if (nvptxSubtarget.hasROT32()) {
    setOperationAction(ISD::ROTL , MVT::i32, Legal);
    setOperationAction(ISD::ROTR , MVT::i32, Legal);
  }
  else {
    setOperationAction(ISD::ROTL , MVT::i32, Expand);
    setOperationAction(ISD::ROTR , MVT::i32, Expand);
  }

  setOperationAction(ISD::ROTL , MVT::i16, Expand);
  setOperationAction(ISD::ROTR , MVT::i16, Expand);
  setOperationAction(ISD::ROTL , MVT::i8, Expand);
  setOperationAction(ISD::ROTR , MVT::i8, Expand);
  setOperationAction(ISD::BSWAP , MVT::i16, Expand);
  setOperationAction(ISD::BSWAP , MVT::i32, Expand);
  setOperationAction(ISD::BSWAP , MVT::i64, Expand);

  // Indirect branch is not supported.
  // This also disables Jump Table creation.
  setOperationAction(ISD::BR_JT,             MVT::Other, Expand);
  setOperationAction(ISD::BRIND,             MVT::Other, Expand);

  setOperationAction(ISD::GlobalAddress   , MVT::i32  , Custom);
  setOperationAction(ISD::GlobalAddress   , MVT::i64  , Custom);

  // We want to legalize constant related memmove and memcopy
  // intrinsics.
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

  // Turn FP extload into load/fextend
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
  // Turn FP truncstore into trunc + store.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // PTX does not support load / store predicate registers
  setOperationAction(ISD::LOAD, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);

  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setTruncStoreAction(MVT::i64, MVT::i1, Expand);
  setTruncStoreAction(MVT::i32, MVT::i1, Expand);
  setTruncStoreAction(MVT::i16, MVT::i1, Expand);
  setTruncStoreAction(MVT::i8, MVT::i1, Expand);

  // This is legal in NVPTX
  setOperationAction(ISD::ConstantFP,         MVT::f64, Legal);
  setOperationAction(ISD::ConstantFP,         MVT::f32, Legal);

  // TRAP can be lowered to PTX trap
  setOperationAction(ISD::TRAP,               MVT::Other, Legal);

  // By default, CONCAT_VECTORS is implemented via store/load
  // through stack. It is slow and uses local memory. We need
  // to custom-lowering them.
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4i32  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4f32  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4i16  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4i8   , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v2i64  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v2f64  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v2i32  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v2f32  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v2i16  , Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v2i8   , Custom);

  // Expand vector int to float and float to int conversions
  // - For SINT_TO_FP and UINT_TO_FP, the src type
  //   (Node->getOperand(0).getValueType())
  //   is used to determine the action, while for FP_TO_UINT and FP_TO_SINT,
  //   the dest type (Node->getValueType(0)) is used.
  //
  //   See VectorLegalizer::LegalizeOp() (LegalizeVectorOps.cpp) for the vector
  //   case, and
  //   SelectionDAGLegalize::LegalizeOp() (LegalizeDAG.cpp) for the scalar case.
  //
  //   That is why v4i32 or v2i32 are used here.
  //
  //   The expansion for vectors happens in VectorLegalizer::LegalizeOp()
  //   (LegalizeVectorOps.cpp).
  setOperationAction(ISD::SINT_TO_FP, MVT::v4i32, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::v2i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::v4i32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::v2i32, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::v2i32, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::v4i32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::v2i32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::v4i32, Expand);

  // Now deduce the information based on the above mentioned
  // actions
  computeRegisterProperties();
}


const char *NVPTXTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case NVPTXISD::CALL:            return "NVPTXISD::CALL";
  case NVPTXISD::RET_FLAG:        return "NVPTXISD::RET_FLAG";
  case NVPTXISD::Wrapper:         return "NVPTXISD::Wrapper";
  case NVPTXISD::NVBuiltin:       return "NVPTXISD::NVBuiltin";
  case NVPTXISD::DeclareParam:    return "NVPTXISD::DeclareParam";
  case NVPTXISD::DeclareScalarParam:
    return "NVPTXISD::DeclareScalarParam";
  case NVPTXISD::DeclareRet:      return "NVPTXISD::DeclareRet";
  case NVPTXISD::DeclareRetParam: return "NVPTXISD::DeclareRetParam";
  case NVPTXISD::PrintCall:       return "NVPTXISD::PrintCall";
  case NVPTXISD::LoadParam:       return "NVPTXISD::LoadParam";
  case NVPTXISD::StoreParam:      return "NVPTXISD::StoreParam";
  case NVPTXISD::StoreParamS32:   return "NVPTXISD::StoreParamS32";
  case NVPTXISD::StoreParamU32:   return "NVPTXISD::StoreParamU32";
  case NVPTXISD::MoveToParam:     return "NVPTXISD::MoveToParam";
  case NVPTXISD::CallArgBegin:    return "NVPTXISD::CallArgBegin";
  case NVPTXISD::CallArg:         return "NVPTXISD::CallArg";
  case NVPTXISD::LastCallArg:     return "NVPTXISD::LastCallArg";
  case NVPTXISD::CallArgEnd:      return "NVPTXISD::CallArgEnd";
  case NVPTXISD::CallVoid:        return "NVPTXISD::CallVoid";
  case NVPTXISD::CallVal:         return "NVPTXISD::CallVal";
  case NVPTXISD::CallSymbol:      return "NVPTXISD::CallSymbol";
  case NVPTXISD::Prototype:       return "NVPTXISD::Prototype";
  case NVPTXISD::MoveParam:       return "NVPTXISD::MoveParam";
  case NVPTXISD::MoveRetval:      return "NVPTXISD::MoveRetval";
  case NVPTXISD::MoveToRetval:    return "NVPTXISD::MoveToRetval";
  case NVPTXISD::StoreRetval:     return "NVPTXISD::StoreRetval";
  case NVPTXISD::PseudoUseParam:  return "NVPTXISD::PseudoUseParam";
  case NVPTXISD::RETURN:          return "NVPTXISD::RETURN";
  case NVPTXISD::CallSeqBegin:    return "NVPTXISD::CallSeqBegin";
  case NVPTXISD::CallSeqEnd:      return "NVPTXISD::CallSeqEnd";
  }
}


SDValue
NVPTXTargetLowering::LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  Op = DAG.getTargetGlobalAddress(GV, dl, getPointerTy());
  return DAG.getNode(NVPTXISD::Wrapper, dl, getPointerTy(), Op);
}

std::string NVPTXTargetLowering::getPrototype(Type *retTy,
                                              const ArgListTy &Args,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                              unsigned retAlignment) const {

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);

  std::stringstream O;
  O << "prototype_" << uniqueCallSite << " : .callprototype ";

  if (retTy->getTypeID() == Type::VoidTyID)
    O << "()";
  else {
    O << "(";
    if (isABI) {
      if (retTy->isPrimitiveType() || retTy->isIntegerTy()) {
        unsigned size = 0;
        if (const IntegerType *ITy = dyn_cast<IntegerType>(retTy)) {
          size = ITy->getBitWidth();
          if (size < 32) size = 32;
        }
        else {
          assert(retTy->isFloatingPointTy() &&
                 "Floating point type expected here");
          size = retTy->getPrimitiveSizeInBits();
        }

        O << ".param .b" << size << " _";
      }
      else if (isa<PointerType>(retTy))
        O << ".param .b" << getPointerTy().getSizeInBits()
        << " _";
      else {
        if ((retTy->getTypeID() == Type::StructTyID) ||
            isa<VectorType>(retTy)) {
          SmallVector<EVT, 16> vtparts;
          ComputeValueVTs(*this, retTy, vtparts);
          unsigned totalsz = 0;
          for (unsigned i=0,e=vtparts.size(); i!=e; ++i) {
            unsigned elems = 1;
            EVT elemtype = vtparts[i];
            if (vtparts[i].isVector()) {
              elems = vtparts[i].getVectorNumElements();
              elemtype = vtparts[i].getVectorElementType();
            }
            for (unsigned j=0, je=elems; j!=je; ++j) {
              unsigned sz = elemtype.getSizeInBits();
              if (elemtype.isInteger() && (sz < 8)) sz = 8;
              totalsz += sz/8;
            }
          }
          O << ".param .align "
              << retAlignment
              << " .b8 _["
              << totalsz << "]";
        }
        else {
          assert(false &&
                 "Unknown return type");
        }
      }
    }
    else {
      SmallVector<EVT, 16> vtparts;
      ComputeValueVTs(*this, retTy, vtparts);
      unsigned idx = 0;
      for (unsigned i=0,e=vtparts.size(); i!=e; ++i) {
        unsigned elems = 1;
        EVT elemtype = vtparts[i];
        if (vtparts[i].isVector()) {
          elems = vtparts[i].getVectorNumElements();
          elemtype = vtparts[i].getVectorElementType();
        }

        for (unsigned j=0, je=elems; j!=je; ++j) {
          unsigned sz = elemtype.getSizeInBits();
          if (elemtype.isInteger() && (sz < 32)) sz = 32;
          O << ".reg .b" << sz << " _";
          if (j<je-1) O << ", ";
          ++idx;
        }
        if (i < e-1)
          O << ", ";
      }
    }
    O << ") ";
  }
  O << "_ (";

  bool first = true;
  MVT thePointerTy = getPointerTy();

  for (unsigned i=0,e=Args.size(); i!=e; ++i) {
    const Type *Ty = Args[i].Ty;
    if (!first) {
      O << ", ";
    }
    first = false;

    if (Outs[i].Flags.isByVal() == false) {
      unsigned sz = 0;
      if (isa<IntegerType>(Ty)) {
        sz = cast<IntegerType>(Ty)->getBitWidth();
        if (sz < 32) sz = 32;
      }
      else if (isa<PointerType>(Ty))
        sz = thePointerTy.getSizeInBits();
      else
        sz = Ty->getPrimitiveSizeInBits();
      if (isABI)
        O << ".param .b" << sz << " ";
      else
        O << ".reg .b" << sz << " ";
      O << "_";
      continue;
    }
    const PointerType *PTy = dyn_cast<PointerType>(Ty);
    assert(PTy &&
           "Param with byval attribute should be a pointer type");
    Type *ETy = PTy->getElementType();

    if (isABI) {
      unsigned align = Outs[i].Flags.getByValAlign();
      unsigned sz = getDataLayout()->getTypeAllocSize(ETy);
      O << ".param .align " << align
          << " .b8 ";
      O << "_";
      O << "[" << sz << "]";
      continue;
    }
    else {
      SmallVector<EVT, 16> vtparts;
      ComputeValueVTs(*this, ETy, vtparts);
      for (unsigned i=0,e=vtparts.size(); i!=e; ++i) {
        unsigned elems = 1;
        EVT elemtype = vtparts[i];
        if (vtparts[i].isVector()) {
          elems = vtparts[i].getVectorNumElements();
          elemtype = vtparts[i].getVectorElementType();
        }

        for (unsigned j=0,je=elems; j!=je; ++j) {
          unsigned sz = elemtype.getSizeInBits();
          if (elemtype.isInteger() && (sz < 32)) sz = 32;
          O << ".reg .b" << sz << " ";
          O << "_";
          if (j<je-1) O << ", ";
        }
        if (i<e-1)
          O << ", ";
      }
      continue;
    }
  }
  O << ");";
  return O.str();
}


SDValue
NVPTXTargetLowering::LowerCall(TargetLowering::CallLoweringInfo &CLI,
                               SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG                     = CLI.DAG;
  DebugLoc &dl                          = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals     = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  bool &isTailCall                      = CLI.IsTailCall;
  ArgListTy &Args                       = CLI.Args;
  Type *retTy                           = CLI.RetTy;
  ImmutableCallSite *CS                 = CLI.CS;

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);

  SDValue tempChain = Chain;
  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getIntPtrConstant(uniqueCallSite, true));
  SDValue InFlag = Chain.getValue(1);

  assert((Outs.size() == Args.size()) &&
         "Unexpected number of arguments to function call");
  unsigned paramCount = 0;
  // Declare the .params or .reg need to pass values
  // to the function
  for (unsigned i=0, e=Outs.size(); i!=e; ++i) {
    EVT VT = Outs[i].VT;

    if (Outs[i].Flags.isByVal() == false) {
      // Plain scalar
      // for ABI,    declare .param .b<size> .param<n>;
      // for nonABI, declare .reg .b<size> .param<n>;
      unsigned isReg = 1;
      if (isABI)
        isReg = 0;
      unsigned sz = VT.getSizeInBits();
      if (VT.isInteger() && (sz < 32)) sz = 32;
      SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue DeclareParamOps[] = { Chain,
                                    DAG.getConstant(paramCount, MVT::i32),
                                    DAG.getConstant(sz, MVT::i32),
                                    DAG.getConstant(isReg, MVT::i32),
                                    InFlag };
      Chain = DAG.getNode(NVPTXISD::DeclareScalarParam, dl, DeclareParamVTs,
                          DeclareParamOps, 5);
      InFlag = Chain.getValue(1);
      SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      SDValue CopyParamOps[] = { Chain, DAG.getConstant(paramCount, MVT::i32),
                             DAG.getConstant(0, MVT::i32), OutVals[i], InFlag };

      unsigned opcode = NVPTXISD::StoreParam;
      if (isReg)
        opcode = NVPTXISD::MoveToParam;
      else {
        if (Outs[i].Flags.isZExt())
          opcode = NVPTXISD::StoreParamU32;
        else if (Outs[i].Flags.isSExt())
          opcode = NVPTXISD::StoreParamS32;
      }
      Chain = DAG.getNode(opcode, dl, CopyParamVTs, CopyParamOps, 5);

      InFlag = Chain.getValue(1);
      ++paramCount;
      continue;
    }
    // struct or vector
    SmallVector<EVT, 16> vtparts;
    const PointerType *PTy = dyn_cast<PointerType>(Args[i].Ty);
    assert(PTy &&
           "Type of a byval parameter should be pointer");
    ComputeValueVTs(*this, PTy->getElementType(), vtparts);

    if (isABI) {
      // declare .param .align 16 .b8 .param<n>[<size>];
      unsigned sz = Outs[i].Flags.getByValSize();
      SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
      // The ByValAlign in the Outs[i].Flags is alway set at this point, so we
      // don't need to
      // worry about natural alignment or not. See TargetLowering::LowerCallTo()
      SDValue DeclareParamOps[] = { Chain,
                       DAG.getConstant(Outs[i].Flags.getByValAlign(), MVT::i32),
                                    DAG.getConstant(paramCount, MVT::i32),
                                    DAG.getConstant(sz, MVT::i32),
                                    InFlag };
      Chain = DAG.getNode(NVPTXISD::DeclareParam, dl, DeclareParamVTs,
                          DeclareParamOps, 5);
      InFlag = Chain.getValue(1);
      unsigned curOffset = 0;
      for (unsigned j=0,je=vtparts.size(); j!=je; ++j) {
        unsigned elems = 1;
        EVT elemtype = vtparts[j];
        if (vtparts[j].isVector()) {
          elems = vtparts[j].getVectorNumElements();
          elemtype = vtparts[j].getVectorElementType();
        }
        for (unsigned k=0,ke=elems; k!=ke; ++k) {
          unsigned sz = elemtype.getSizeInBits();
          if (elemtype.isInteger() && (sz < 8)) sz = 8;
          SDValue srcAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                                        OutVals[i],
                                        DAG.getConstant(curOffset,
                                                        getPointerTy()));
          SDValue theVal = DAG.getLoad(elemtype, dl, tempChain, srcAddr,
                                MachinePointerInfo(), false, false, false, 0);
          SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
          SDValue CopyParamOps[] = { Chain, DAG.getConstant(paramCount,
                                                            MVT::i32),
                                           DAG.getConstant(curOffset, MVT::i32),
                                                            theVal, InFlag };
          Chain = DAG.getNode(NVPTXISD::StoreParam, dl, CopyParamVTs,
                              CopyParamOps, 5);
          InFlag = Chain.getValue(1);
          curOffset += sz/8;
        }
      }
      ++paramCount;
      continue;
    }
    // Non-abi, struct or vector
    // Declare a bunch or .reg .b<size> .param<n>
    unsigned curOffset = 0;
    for (unsigned j=0,je=vtparts.size(); j!=je; ++j) {
      unsigned elems = 1;
      EVT elemtype = vtparts[j];
      if (vtparts[j].isVector()) {
        elems = vtparts[j].getVectorNumElements();
        elemtype = vtparts[j].getVectorElementType();
      }
      for (unsigned k=0,ke=elems; k!=ke; ++k) {
        unsigned sz = elemtype.getSizeInBits();
        if (elemtype.isInteger() && (sz < 32)) sz = 32;
        SDVTList DeclareParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue DeclareParamOps[] = { Chain, DAG.getConstant(paramCount,
                                                             MVT::i32),
                                                  DAG.getConstant(sz, MVT::i32),
                                                   DAG.getConstant(1, MVT::i32),
                                                             InFlag };
        Chain = DAG.getNode(NVPTXISD::DeclareScalarParam, dl, DeclareParamVTs,
                            DeclareParamOps, 5);
        InFlag = Chain.getValue(1);
        SDValue srcAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), OutVals[i],
                                      DAG.getConstant(curOffset,
                                                      getPointerTy()));
        SDValue theVal = DAG.getLoad(elemtype, dl, tempChain, srcAddr,
                                  MachinePointerInfo(), false, false, false, 0);
        SDVTList CopyParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue CopyParamOps[] = { Chain, DAG.getConstant(paramCount, MVT::i32),
                                   DAG.getConstant(0, MVT::i32), theVal,
                                   InFlag };
        Chain = DAG.getNode(NVPTXISD::MoveToParam, dl, CopyParamVTs,
                            CopyParamOps, 5);
        InFlag = Chain.getValue(1);
        ++paramCount;
      }
    }
  }

  GlobalAddressSDNode *Func = dyn_cast<GlobalAddressSDNode>(Callee.getNode());
  unsigned retAlignment = 0;

  // Handle Result
  unsigned retCount = 0;
  if (Ins.size() > 0) {
    SmallVector<EVT, 16> resvtparts;
    ComputeValueVTs(*this, retTy, resvtparts);

    // Declare one .param .align 16 .b8 func_retval0[<size>] for ABI or
    // individual .reg .b<size> func_retval<0..> for non ABI
    unsigned resultsz = 0;
    for (unsigned i=0,e=resvtparts.size(); i!=e; ++i) {
      unsigned elems = 1;
      EVT elemtype = resvtparts[i];
      if (resvtparts[i].isVector()) {
        elems = resvtparts[i].getVectorNumElements();
        elemtype = resvtparts[i].getVectorElementType();
      }
      for (unsigned j=0,je=elems; j!=je; ++j) {
        unsigned sz = elemtype.getSizeInBits();
        if (isABI == false) {
          if (elemtype.isInteger() && (sz < 32)) sz = 32;
        }
        else {
          if (elemtype.isInteger() && (sz < 8)) sz = 8;
        }
        if (isABI == false) {
          SDVTList DeclareRetVTs = DAG.getVTList(MVT::Other, MVT::Glue);
          SDValue DeclareRetOps[] = { Chain, DAG.getConstant(2, MVT::i32),
                                      DAG.getConstant(sz, MVT::i32),
                                      DAG.getConstant(retCount, MVT::i32),
                                      InFlag };
          Chain = DAG.getNode(NVPTXISD::DeclareRet, dl, DeclareRetVTs,
                              DeclareRetOps, 5);
          InFlag = Chain.getValue(1);
          ++retCount;
        }
        resultsz += sz;
      }
    }
    if (isABI) {
      if (retTy->isPrimitiveType() || retTy->isIntegerTy() ||
          retTy->isPointerTy() ) {
        // Scalar needs to be at least 32bit wide
        if (resultsz < 32)
          resultsz = 32;
        SDVTList DeclareRetVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue DeclareRetOps[] = { Chain, DAG.getConstant(1, MVT::i32),
                                    DAG.getConstant(resultsz, MVT::i32),
                                    DAG.getConstant(0, MVT::i32), InFlag };
        Chain = DAG.getNode(NVPTXISD::DeclareRet, dl, DeclareRetVTs,
                            DeclareRetOps, 5);
        InFlag = Chain.getValue(1);
      }
      else {
        if (Func) { // direct call
          if (!llvm::getAlign(*(CS->getCalledFunction()), 0, retAlignment))
            retAlignment = getDataLayout()->getABITypeAlignment(retTy);
        } else { // indirect call
          const CallInst *CallI = dyn_cast<CallInst>(CS->getInstruction());
          if (!llvm::getAlign(*CallI, 0, retAlignment))
            retAlignment = getDataLayout()->getABITypeAlignment(retTy);
        }
        SDVTList DeclareRetVTs = DAG.getVTList(MVT::Other, MVT::Glue);
        SDValue DeclareRetOps[] = { Chain, DAG.getConstant(retAlignment,
                                                           MVT::i32),
                                          DAG.getConstant(resultsz/8, MVT::i32),
                                         DAG.getConstant(0, MVT::i32), InFlag };
        Chain = DAG.getNode(NVPTXISD::DeclareRetParam, dl, DeclareRetVTs,
                            DeclareRetOps, 5);
        InFlag = Chain.getValue(1);
      }
    }
  }

  if (!Func) {
    // This is indirect function call case : PTX requires a prototype of the
    // form
    // proto_0 : .callprototype(.param .b32 _) _ (.param .b32 _);
    // to be emitted, and the label has to used as the last arg of call
    // instruction.
    // The prototype is embedded in a string and put as the operand for an
    // INLINEASM SDNode.
    SDVTList InlineAsmVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    std::string proto_string = getPrototype(retTy, Args, Outs, retAlignment);
    const char *asmstr = nvTM->getManagedStrPool()->
        getManagedString(proto_string.c_str())->c_str();
    SDValue InlineAsmOps[] = { Chain,
                               DAG.getTargetExternalSymbol(asmstr,
                                                           getPointerTy()),
                                                           DAG.getMDNode(0),
                                   DAG.getTargetConstant(0, MVT::i32), InFlag };
    Chain = DAG.getNode(ISD::INLINEASM, dl, InlineAsmVTs, InlineAsmOps, 5);
    InFlag = Chain.getValue(1);
  }
  // Op to just print "call"
  SDVTList PrintCallVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue PrintCallOps[] = { Chain,
                             DAG.getConstant(isABI ? ((Ins.size()==0) ? 0 : 1)
                                 : retCount, MVT::i32),
                                   InFlag };
  Chain = DAG.getNode(Func?(NVPTXISD::PrintCallUni):(NVPTXISD::PrintCall), dl,
      PrintCallVTs, PrintCallOps, 3);
  InFlag = Chain.getValue(1);

  // Ops to print out the function name
  SDVTList CallVoidVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue CallVoidOps[] = { Chain, Callee, InFlag };
  Chain = DAG.getNode(NVPTXISD::CallVoid, dl, CallVoidVTs, CallVoidOps, 3);
  InFlag = Chain.getValue(1);

  // Ops to print out the param list
  SDVTList CallArgBeginVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue CallArgBeginOps[] = { Chain, InFlag };
  Chain = DAG.getNode(NVPTXISD::CallArgBegin, dl, CallArgBeginVTs,
                      CallArgBeginOps, 2);
  InFlag = Chain.getValue(1);

  for (unsigned i=0, e=paramCount; i!=e; ++i) {
    unsigned opcode;
    if (i==(e-1))
      opcode = NVPTXISD::LastCallArg;
    else
      opcode = NVPTXISD::CallArg;
    SDVTList CallArgVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue CallArgOps[] = { Chain, DAG.getConstant(1, MVT::i32),
                             DAG.getConstant(i, MVT::i32),
                             InFlag };
    Chain = DAG.getNode(opcode, dl, CallArgVTs, CallArgOps, 4);
    InFlag = Chain.getValue(1);
  }
  SDVTList CallArgEndVTs = DAG.getVTList(MVT::Other, MVT::Glue);
  SDValue CallArgEndOps[] = { Chain,
                              DAG.getConstant(Func ? 1 : 0, MVT::i32),
                              InFlag };
  Chain = DAG.getNode(NVPTXISD::CallArgEnd, dl, CallArgEndVTs, CallArgEndOps,
                      3);
  InFlag = Chain.getValue(1);

  if (!Func) {
    SDVTList PrototypeVTs = DAG.getVTList(MVT::Other, MVT::Glue);
    SDValue PrototypeOps[] = { Chain,
                               DAG.getConstant(uniqueCallSite, MVT::i32),
                               InFlag };
    Chain = DAG.getNode(NVPTXISD::Prototype, dl, PrototypeVTs, PrototypeOps, 3);
    InFlag = Chain.getValue(1);
  }

  // Generate loads from param memory/moves from registers for result
  if (Ins.size() > 0) {
    if (isABI) {
      unsigned resoffset = 0;
      for (unsigned i=0,e=Ins.size(); i!=e; ++i) {
        unsigned sz = Ins[i].VT.getSizeInBits();
        if (Ins[i].VT.isInteger() && (sz < 8)) sz = 8;
        std::vector<EVT> LoadRetVTs;
        LoadRetVTs.push_back(Ins[i].VT);
        LoadRetVTs.push_back(MVT::Other); LoadRetVTs.push_back(MVT::Glue);
        std::vector<SDValue> LoadRetOps;
        LoadRetOps.push_back(Chain);
        LoadRetOps.push_back(DAG.getConstant(1, MVT::i32));
        LoadRetOps.push_back(DAG.getConstant(resoffset, MVT::i32));
        LoadRetOps.push_back(InFlag);
        SDValue retval = DAG.getNode(NVPTXISD::LoadParam, dl, LoadRetVTs,
                                     &LoadRetOps[0], LoadRetOps.size());
        Chain = retval.getValue(1);
        InFlag = retval.getValue(2);
        InVals.push_back(retval);
        resoffset += sz/8;
      }
    }
    else {
      SmallVector<EVT, 16> resvtparts;
      ComputeValueVTs(*this, retTy, resvtparts);

      assert(Ins.size() == resvtparts.size() &&
             "Unexpected number of return values in non-ABI case");
      unsigned paramNum = 0;
      for (unsigned i=0,e=Ins.size(); i!=e; ++i) {
        assert(EVT(Ins[i].VT) == resvtparts[i] &&
               "Unexpected EVT type in non-ABI case");
        unsigned numelems = 1;
        EVT elemtype = Ins[i].VT;
        if (Ins[i].VT.isVector()) {
          numelems = Ins[i].VT.getVectorNumElements();
          elemtype = Ins[i].VT.getVectorElementType();
        }
        std::vector<SDValue> tempRetVals;
        for (unsigned j=0; j<numelems; ++j) {
          std::vector<EVT> MoveRetVTs;
          MoveRetVTs.push_back(elemtype);
          MoveRetVTs.push_back(MVT::Other); MoveRetVTs.push_back(MVT::Glue);
          std::vector<SDValue> MoveRetOps;
          MoveRetOps.push_back(Chain);
          MoveRetOps.push_back(DAG.getConstant(0, MVT::i32));
          MoveRetOps.push_back(DAG.getConstant(paramNum, MVT::i32));
          MoveRetOps.push_back(InFlag);
          SDValue retval = DAG.getNode(NVPTXISD::LoadParam, dl, MoveRetVTs,
                                       &MoveRetOps[0], MoveRetOps.size());
          Chain = retval.getValue(1);
          InFlag = retval.getValue(2);
          tempRetVals.push_back(retval);
          ++paramNum;
        }
        if (Ins[i].VT.isVector())
          InVals.push_back(DAG.getNode(ISD::BUILD_VECTOR, dl, Ins[i].VT,
                                       &tempRetVals[0], tempRetVals.size()));
        else
          InVals.push_back(tempRetVals[0]);
      }
    }
  }
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getIntPtrConstant(uniqueCallSite, true),
                             DAG.getIntPtrConstant(uniqueCallSite+1, true),
                             InFlag);
  uniqueCallSite++;

  // set isTailCall to false for now, until we figure out how to express
  // tail call optimization in PTX
  isTailCall = false;
  return Chain;
}

// By default CONCAT_VECTORS is lowered by ExpandVectorBuildThroughStack()
// (see LegalizeDAG.cpp). This is slow and uses local memory.
// We use extract/insert/build vector just as what LegalizeOp() does in llvm 2.5
SDValue NVPTXTargetLowering::
LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  SmallVector<SDValue, 8> Ops;
  unsigned NumOperands = Node->getNumOperands();
  for (unsigned i=0; i < NumOperands; ++i) {
    SDValue SubOp = Node->getOperand(i);
    EVT VVT = SubOp.getNode()->getValueType(0);
    EVT EltVT = VVT.getVectorElementType();
    unsigned NumSubElem = VVT.getVectorNumElements();
    for (unsigned j=0; j < NumSubElem; ++j) {
      Ops.push_back(DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl, EltVT, SubOp,
                                DAG.getIntPtrConstant(j)));
    }
  }
  return DAG.getNode(ISD::BUILD_VECTOR, dl, Node->getValueType(0),
                     &Ops[0], Ops.size());
}

SDValue NVPTXTargetLowering::
LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::RETURNADDR: return SDValue();
  case ISD::FRAMEADDR:  return SDValue();
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN: return Op;
  case ISD::BUILD_VECTOR:
  case ISD::EXTRACT_SUBVECTOR:
    return Op;
  case ISD::CONCAT_VECTORS: return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::STORE: return LowerSTORE(Op, DAG);
  case ISD::LOAD: return LowerLOAD(Op, DAG);
  default:
    llvm_unreachable("Custom lowering not defined for operation");
  }
}


// v = ld i1* addr
//   =>
// v1 = ld i8* addr
// v = trunc v1 to i1
SDValue NVPTXTargetLowering::
LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  LoadSDNode *LD = cast<LoadSDNode>(Node);
  DebugLoc dl = Node->getDebugLoc();
  assert(LD->getExtensionType() == ISD::NON_EXTLOAD) ;
  assert(Node->getValueType(0) == MVT::i1 &&
         "Custom lowering for i1 load only");
  SDValue newLD = DAG.getLoad(MVT::i8, dl, LD->getChain(), LD->getBasePtr(),
                              LD->getPointerInfo(),
                              LD->isVolatile(), LD->isNonTemporal(),
                              LD->isInvariant(),
                              LD->getAlignment());
  SDValue result = DAG.getNode(ISD::TRUNCATE, dl, MVT::i1, newLD);
  // The legalizer (the caller) is expecting two values from the legalized
  // load, so we build a MergeValues node for it. See ExpandUnalignedLoad()
  // in LegalizeDAG.cpp which also uses MergeValues.
  SDValue Ops[] = {result, LD->getChain()};
  return DAG.getMergeValues(Ops, 2, dl);
}

// st i1 v, addr
//    =>
// v1 = zxt v to i8
// st i8, addr
SDValue NVPTXTargetLowering::
LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  DebugLoc dl = Node->getDebugLoc();
  StoreSDNode *ST = cast<StoreSDNode>(Node);
  SDValue Tmp1 = ST->getChain();
  SDValue Tmp2 = ST->getBasePtr();
  SDValue Tmp3 = ST->getValue();
  assert(Tmp3.getValueType() == MVT::i1 && "Custom lowering for i1 store only");
  unsigned Alignment = ST->getAlignment();
  bool isVolatile = ST->isVolatile();
  bool isNonTemporal = ST->isNonTemporal();
  Tmp3 = DAG.getNode(ISD::ZERO_EXTEND, dl,
                     MVT::i8, Tmp3);
  SDValue Result = DAG.getStore(Tmp1, dl, Tmp3, Tmp2,
                                ST->getPointerInfo(), isVolatile,
                                isNonTemporal, Alignment);
  return Result;
}


SDValue
NVPTXTargetLowering::getExtSymb(SelectionDAG &DAG, const char *inname, int idx,
                                EVT v) const {
  std::string *name = nvTM->getManagedStrPool()->getManagedString(inname);
  std::stringstream suffix;
  suffix << idx;
  *name += suffix.str();
  return DAG.getTargetExternalSymbol(name->c_str(), v);
}

SDValue
NVPTXTargetLowering::getParamSymbol(SelectionDAG &DAG, int idx, EVT v) const {
  return getExtSymb(DAG, ".PARAM", idx, v);
}

SDValue
NVPTXTargetLowering::getParamHelpSymbol(SelectionDAG &DAG, int idx) {
  return getExtSymb(DAG, ".HLPPARAM", idx);
}

// Check to see if the kernel argument is image*_t or sampler_t

bool llvm::isImageOrSamplerVal(const Value *arg, const Module *context) {
  static const char *const specialTypes[] = {
                                             "struct._image2d_t",
                                             "struct._image3d_t",
                                             "struct._sampler_t"
  };

  const Type *Ty = arg->getType();
  const PointerType *PTy = dyn_cast<PointerType>(Ty);

  if (!PTy)
    return false;

  if (!context)
    return false;

  const StructType *STy = dyn_cast<StructType>(PTy->getElementType());
  const std::string TypeName = STy ? STy->getName() : "";

  for (int i = 0, e = array_lengthof(specialTypes); i != e; ++i)
    if (TypeName == specialTypes[i])
      return true;

  return false;
}

SDValue
NVPTXTargetLowering::LowerFormalArguments(SDValue Chain,
                                        CallingConv::ID CallConv, bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                          DebugLoc dl, SelectionDAG &DAG,
                                       SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const DataLayout *TD = getDataLayout();

  const Function *F = MF.getFunction();
  const AttrListPtr &PAL = F->getAttributes();

  SDValue Root = DAG.getRoot();
  std::vector<SDValue> OutChains;

  bool isKernel = llvm::isKernelFunction(*F);
  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);

  std::vector<Type *> argTypes;
  std::vector<const Argument *> theArgs;
  for (Function::const_arg_iterator I = F->arg_begin(), E = F->arg_end();
      I != E; ++I) {
    theArgs.push_back(I);
    argTypes.push_back(I->getType());
  }
  assert(argTypes.size() == Ins.size() &&
         "Ins types and function types did not match");

  int idx = 0;
  for (unsigned i=0, e=Ins.size(); i!=e; ++i, ++idx) {
    Type *Ty = argTypes[i];
    EVT ObjectVT = getValueType(Ty);
    assert(ObjectVT == Ins[i].VT &&
           "Ins type did not match function type");

    // If the kernel argument is image*_t or sampler_t, convert it to
    // a i32 constant holding the parameter position. This can later
    // matched in the AsmPrinter to output the correct mangled name.
    if (isImageOrSamplerVal(theArgs[i],
                           (theArgs[i]->getParent() ?
                               theArgs[i]->getParent()->getParent() : 0))) {
      assert(isKernel && "Only kernels can have image/sampler params");
      InVals.push_back(DAG.getConstant(i+1, MVT::i32));
      continue;
    }

    if (theArgs[i]->use_empty()) {
      // argument is dead
      InVals.push_back(DAG.getNode(ISD::UNDEF, dl, ObjectVT));
      continue;
    }

    // In the following cases, assign a node order of "idx+1"
    // to newly created nodes. The SDNOdes for params have to
    // appear in the same order as their order of appearance
    // in the original function. "idx+1" holds that order.
    if (PAL.getParamAttributes(i+1).hasAttribute(Attributes::ByVal) == false) {
      // A plain scalar.
      if (isABI || isKernel) {
        // If ABI, load from the param symbol
        SDValue Arg = getParamSymbol(DAG, idx);
        Value *srcValue = new Argument(PointerType::get(ObjectVT.getTypeForEVT(
            F->getContext()),
            llvm::ADDRESS_SPACE_PARAM));
        SDValue p = DAG.getLoad(ObjectVT, dl, Root, Arg,
                                MachinePointerInfo(srcValue), false, false,
                                false,
                                TD->getABITypeAlignment(ObjectVT.getTypeForEVT(
                                  F->getContext())));
        if (p.getNode())
          DAG.AssignOrdering(p.getNode(), idx+1);
        InVals.push_back(p);
      }
      else {
        // If no ABI, just move the param symbol
        SDValue Arg = getParamSymbol(DAG, idx, ObjectVT);
        SDValue p = DAG.getNode(NVPTXISD::MoveParam, dl, ObjectVT, Arg);
        if (p.getNode())
          DAG.AssignOrdering(p.getNode(), idx+1);
        InVals.push_back(p);
      }
      continue;
    }

    // Param has ByVal attribute
    if (isABI || isKernel) {
      // Return MoveParam(param symbol).
      // Ideally, the param symbol can be returned directly,
      // but when SDNode builder decides to use it in a CopyToReg(),
      // machine instruction fails because TargetExternalSymbol
      // (not lowered) is target dependent, and CopyToReg assumes
      // the source is lowered.
      SDValue Arg = getParamSymbol(DAG, idx, getPointerTy());
      SDValue p = DAG.getNode(NVPTXISD::MoveParam, dl, ObjectVT, Arg);
      if (p.getNode())
        DAG.AssignOrdering(p.getNode(), idx+1);
      if (isKernel)
        InVals.push_back(p);
      else {
        SDValue p2 = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, dl, ObjectVT,
                    DAG.getConstant(Intrinsic::nvvm_ptr_local_to_gen, MVT::i32),
                                 p);
        InVals.push_back(p2);
      }
    } else {
      // Have to move a set of param symbols to registers and
      // store them locally and return the local pointer in InVals
      const PointerType *elemPtrType = dyn_cast<PointerType>(argTypes[i]);
      assert(elemPtrType &&
             "Byval parameter should be a pointer type");
      Type *elemType = elemPtrType->getElementType();
      // Compute the constituent parts
      SmallVector<EVT, 16> vtparts;
      SmallVector<uint64_t, 16> offsets;
      ComputeValueVTs(*this, elemType, vtparts, &offsets, 0);
      unsigned totalsize = 0;
      for (unsigned j=0, je=vtparts.size(); j!=je; ++j)
        totalsize += vtparts[j].getStoreSizeInBits();
      SDValue localcopy =  DAG.getFrameIndex(MF.getFrameInfo()->
                                      CreateStackObject(totalsize/8, 16, false),
                                             getPointerTy());
      unsigned sizesofar = 0;
      std::vector<SDValue> theChains;
      for (unsigned j=0, je=vtparts.size(); j!=je; ++j) {
        unsigned numElems = 1;
        if (vtparts[j].isVector()) numElems = vtparts[j].getVectorNumElements();
        for (unsigned k=0, ke=numElems; k!=ke; ++k) {
          EVT tmpvt = vtparts[j];
          if (tmpvt.isVector()) tmpvt = tmpvt.getVectorElementType();
          SDValue arg = DAG.getNode(NVPTXISD::MoveParam, dl, tmpvt,
                                    getParamSymbol(DAG, idx, tmpvt));
          SDValue addr = DAG.getNode(ISD::ADD, dl, getPointerTy(), localcopy,
                                    DAG.getConstant(sizesofar, getPointerTy()));
          theChains.push_back(DAG.getStore(Chain, dl, arg, addr,
                                        MachinePointerInfo(), false, false, 0));
          sizesofar += tmpvt.getStoreSizeInBits()/8;
          ++idx;
        }
      }
      --idx;
      Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other, &theChains[0],
                          theChains.size());
      InVals.push_back(localcopy);
    }
  }

  // Clang will check explicit VarArg and issue error if any. However, Clang
  // will let code with
  // implicit var arg like f() pass.
  // We treat this case as if the arg list is empty.
  //if (F.isVarArg()) {
  // assert(0 && "VarArg not supported yet!");
  //}

  if (!OutChains.empty())
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                            &OutChains[0], OutChains.size()));

  return Chain;
}

SDValue
NVPTXTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                 bool isVarArg,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 DebugLoc dl, SelectionDAG &DAG) const {

  bool isABI = (nvptxSubtarget.getSmVersion() >= 20);

  unsigned sizesofar = 0;
  unsigned idx = 0;
  for (unsigned i=0, e=Outs.size(); i!=e; ++i) {
    SDValue theVal = OutVals[i];
    EVT theValType = theVal.getValueType();
    unsigned numElems = 1;
    if (theValType.isVector()) numElems = theValType.getVectorNumElements();
    for (unsigned j=0,je=numElems; j!=je; ++j) {
      SDValue tmpval = theVal;
      if (theValType.isVector())
        tmpval = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, dl,
                             theValType.getVectorElementType(),
                             tmpval, DAG.getIntPtrConstant(j));
      Chain = DAG.getNode(isABI ? NVPTXISD::StoreRetval :NVPTXISD::MoveToRetval,
          dl, MVT::Other,
          Chain,
          DAG.getConstant(isABI ? sizesofar : idx, MVT::i32),
          tmpval);
      if (theValType.isVector())
        sizesofar += theValType.getVectorElementType().getStoreSizeInBits()/8;
      else
        sizesofar += theValType.getStoreSizeInBits()/8;
      ++idx;
    }
  }

  return DAG.getNode(NVPTXISD::RET_FLAG, dl, MVT::Other, Chain);
}

void
NVPTXTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                  std::string &Constraint,
                                                  std::vector<SDValue> &Ops,
                                                  SelectionDAG &DAG) const
{
  if (Constraint.length() > 1)
    return;
  else
    TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

// NVPTX suuport vector of legal types of any length in Intrinsics because the
// NVPTX specific type legalizer
// will legalize them to the PTX supported length.
bool
NVPTXTargetLowering::isTypeSupportedInIntrinsic(MVT VT) const {
  if (isTypeLegal(VT))
    return true;
  if (VT.isVector()) {
    MVT eVT = VT.getVectorElementType();
    if (isTypeLegal(eVT))
      return true;
  }
  return false;
}


// llvm.ptx.memcpy.const and llvm.ptx.memmove.const need to be modeled as
// TgtMemIntrinsic
// because we need the information that is only available in the "Value" type
// of destination
// pointer. In particular, the address space information.
bool
NVPTXTargetLowering::getTgtMemIntrinsic(IntrinsicInfo& Info, const CallInst &I,
                                        unsigned Intrinsic) const {
  switch (Intrinsic) {
  default:
    return false;

  case Intrinsic::nvvm_atomic_load_add_f32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::f32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.vol = 0;
    Info.readMem = true;
    Info.writeMem = true;
    Info.align = 0;
    return true;

  case Intrinsic::nvvm_atomic_load_inc_32:
  case Intrinsic::nvvm_atomic_load_dec_32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.vol = 0;
    Info.readMem = true;
    Info.writeMem = true;
    Info.align = 0;
    return true;

  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_p:

    Info.opc = ISD::INTRINSIC_W_CHAIN;
    if (Intrinsic == Intrinsic::nvvm_ldu_global_i)
      Info.memVT = MVT::i32;
    else if (Intrinsic == Intrinsic::nvvm_ldu_global_p)
      Info.memVT = getPointerTy();
    else
      Info.memVT = MVT::f32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.vol = 0;
    Info.readMem = true;
    Info.writeMem = false;
    Info.align = 0;
    return true;

  }
  return false;
}

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
/// Used to guide target specific optimizations, like loop strength reduction
/// (LoopStrengthReduce.cpp) and memory optimization for address mode
/// (CodeGenPrepare.cpp)
bool
NVPTXTargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                           Type *Ty) const {

  // AddrMode - This represents an addressing mode of:
  //    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
  //
  // The legal address modes are
  // - [avar]
  // - [areg]
  // - [areg+immoff]
  // - [immAddr]

  if (AM.BaseGV) {
    if (AM.BaseOffs || AM.HasBaseReg || AM.Scale)
      return false;
    return true;
  }

  switch (AM.Scale) {
  case 0:  // "r", "r+i" or "i" is allowed
    break;
  case 1:
    if (AM.HasBaseReg)  // "r+r+i" or "r+r" is not allowed.
      return false;
    // Otherwise we have r+i.
    break;
  default:
    // No scale > 1 is allowed
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
//                         NVPTX Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
NVPTXTargetLowering::ConstraintType
NVPTXTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'r':
    case 'h':
    case 'c':
    case 'l':
    case 'f':
    case 'd':
    case '0':
    case 'N':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}


std::pair<unsigned, const TargetRegisterClass*>
NVPTXTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                  EVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'c':
      return std::make_pair(0U, &NVPTX::Int8RegsRegClass);
    case 'h':
      return std::make_pair(0U, &NVPTX::Int16RegsRegClass);
    case 'r':
      return std::make_pair(0U, &NVPTX::Int32RegsRegClass);
    case 'l':
    case 'N':
      return std::make_pair(0U, &NVPTX::Int64RegsRegClass);
    case 'f':
      return std::make_pair(0U, &NVPTX::Float32RegsRegClass);
    case 'd':
      return std::make_pair(0U, &NVPTX::Float64RegsRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}



/// getFunctionAlignment - Return the Log2 alignment of this function.
unsigned NVPTXTargetLowering::getFunctionAlignment(const Function *) const {
  return 4;
}
