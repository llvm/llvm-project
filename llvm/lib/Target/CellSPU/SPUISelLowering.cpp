//===-- SPUISelLowering.cpp - Cell SPU DAG Lowering Implementation --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPUTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "SPURegisterNames.h"
#include "SPUISelLowering.h"
#include "SPUTargetMachine.h"
#include "SPUFrameInfo.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"

#include <map>

using namespace llvm;

// Used in getTargetNodeName() below
namespace {
  std::map<unsigned, const char *> node_names;

  //! MVT mapping to useful data for Cell SPU
  struct valtype_map_s {
    const MVT   valtype;
    const int   prefslot_byte;
  };

  const valtype_map_s valtype_map[] = {
    { MVT::i1,   3 },
    { MVT::i8,   3 },
    { MVT::i16,  2 },
    { MVT::i32,  0 },
    { MVT::f32,  0 },
    { MVT::i64,  0 },
    { MVT::f64,  0 },
    { MVT::i128, 0 }
  };

  const size_t n_valtype_map = sizeof(valtype_map) / sizeof(valtype_map[0]);

  const valtype_map_s *getValueTypeMapEntry(MVT VT) {
    const valtype_map_s *retval = 0;

    for (size_t i = 0; i < n_valtype_map; ++i) {
      if (valtype_map[i].valtype == VT) {
        retval = valtype_map + i;
        break;
      }
    }

#ifndef NDEBUG
    if (retval == 0) {
      cerr << "getValueTypeMapEntry returns NULL for "
           << VT.getMVTString()
           << "\n";
      abort();
    }
#endif

    return retval;
  }

  //! Expand a library call into an actual call DAG node
  /*!
   \note
   This code is taken from SelectionDAGLegalize, since it is not exposed as
   part of the LLVM SelectionDAG API.
   */

  SDValue
  ExpandLibCall(RTLIB::Libcall LC, SDValue Op, SelectionDAG &DAG,
                bool isSigned, SDValue &Hi, SPUTargetLowering &TLI) {
    // The input chain to this libcall is the entry node of the function.
    // Legalizing the call will automatically add the previous call to the
    // dependence.
    SDValue InChain = DAG.getEntryNode();

    TargetLowering::ArgListTy Args;
    TargetLowering::ArgListEntry Entry;
    for (unsigned i = 0, e = Op.getNumOperands(); i != e; ++i) {
      MVT ArgVT = Op.getOperand(i).getValueType();
      const Type *ArgTy = ArgVT.getTypeForMVT();
      Entry.Node = Op.getOperand(i);
      Entry.Ty = ArgTy;
      Entry.isSExt = isSigned;
      Entry.isZExt = !isSigned;
      Args.push_back(Entry);
    }
    SDValue Callee = DAG.getExternalSymbol(TLI.getLibcallName(LC),
                                           TLI.getPointerTy());

    // Splice the libcall in wherever FindInputOutputChains tells us to.
    const Type *RetTy = Op.getNode()->getValueType(0).getTypeForMVT();
    std::pair<SDValue, SDValue> CallInfo =
            TLI.LowerCallTo(InChain, RetTy, isSigned, !isSigned, false, false,
                            CallingConv::C, false, Callee, Args, DAG,
                            Op.getNode()->getDebugLoc());

    return CallInfo.first;
  }
}

SPUTargetLowering::SPUTargetLowering(SPUTargetMachine &TM)
  : TargetLowering(TM),
    SPUTM(TM)
{
  // Fold away setcc operations if possible.
  setPow2DivIsCheap();

  // Use _setjmp/_longjmp instead of setjmp/longjmp.
  setUseUnderscoreSetJmp(true);
  setUseUnderscoreLongJmp(true);

  // Set RTLIB libcall names as used by SPU:
  setLibcallName(RTLIB::DIV_F64, "__fast_divdf3");

  // Set up the SPU's register classes:
  addRegisterClass(MVT::i8,   SPU::R8CRegisterClass);
  addRegisterClass(MVT::i16,  SPU::R16CRegisterClass);
  addRegisterClass(MVT::i32,  SPU::R32CRegisterClass);
  addRegisterClass(MVT::i64,  SPU::R64CRegisterClass);
  addRegisterClass(MVT::f32,  SPU::R32FPRegisterClass);
  addRegisterClass(MVT::f64,  SPU::R64FPRegisterClass);
  addRegisterClass(MVT::i128, SPU::GPRCRegisterClass);

  // SPU has no sign or zero extended loads for i1, i8, i16:
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);

  setLoadExtAction(ISD::EXTLOAD,  MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD,  MVT::f64, Expand);

  // SPU constant load actions are custom lowered:
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Custom);

  // SPU's loads and stores have to be custom lowered:
  for (unsigned sctype = (unsigned) MVT::i8; sctype < (unsigned) MVT::i128;
       ++sctype) {
    MVT VT = (MVT::SimpleValueType)sctype;

    setOperationAction(ISD::LOAD,   VT, Custom);
    setOperationAction(ISD::STORE,  VT, Custom);
    setLoadExtAction(ISD::EXTLOAD,  VT, Custom);
    setLoadExtAction(ISD::ZEXTLOAD, VT, Custom);
    setLoadExtAction(ISD::SEXTLOAD, VT, Custom);

    for (unsigned stype = sctype - 1; stype >= (unsigned) MVT::i8; --stype) {
      MVT StoreVT = (MVT::SimpleValueType) stype;
      setTruncStoreAction(VT, StoreVT, Expand);
    }
  }

  for (unsigned sctype = (unsigned) MVT::f32; sctype < (unsigned) MVT::f64;
       ++sctype) {
    MVT VT = (MVT::SimpleValueType) sctype;

    setOperationAction(ISD::LOAD,   VT, Custom);
    setOperationAction(ISD::STORE,  VT, Custom);

    for (unsigned stype = sctype - 1; stype >= (unsigned) MVT::f32; --stype) {
      MVT StoreVT = (MVT::SimpleValueType) stype;
      setTruncStoreAction(VT, StoreVT, Expand);
    }
  }

  // Expand the jumptable branches
  setOperationAction(ISD::BR_JT,        MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,        MVT::Other, Expand);

  // Custom lower SELECT_CC for most cases, but expand by default
  setOperationAction(ISD::SELECT_CC,    MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,    MVT::i8,    Custom);
  setOperationAction(ISD::SELECT_CC,    MVT::i16,   Custom);
  setOperationAction(ISD::SELECT_CC,    MVT::i32,   Custom);
  setOperationAction(ISD::SELECT_CC,    MVT::i64,   Custom);

  // SPU has no intrinsics for these particular operations:
  setOperationAction(ISD::MEMBARRIER, MVT::Other, Expand);

  // SPU has no SREM/UREM instructions
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);

  // We don't support sin/cos/sqrt/fmod
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FREM , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);
  setOperationAction(ISD::FREM , MVT::f32, Expand);

  // Expand fsqrt to the appropriate libcall (NOTE: should use h/w fsqrt
  // for f32!)
  setOperationAction(ISD::FSQRT, MVT::f64, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);

  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  // SPU can do rotate right and left, so legalize it... but customize for i8
  // because instructions don't exist.

  // FIXME: Change from "expand" to appropriate type once ROTR is supported in
  //        .td files.
  setOperationAction(ISD::ROTR, MVT::i32,    Expand /*Legal*/);
  setOperationAction(ISD::ROTR, MVT::i16,    Expand /*Legal*/);
  setOperationAction(ISD::ROTR, MVT::i8,     Expand /*Custom*/);

  setOperationAction(ISD::ROTL, MVT::i32,    Legal);
  setOperationAction(ISD::ROTL, MVT::i16,    Legal);
  setOperationAction(ISD::ROTL, MVT::i8,     Custom);

  // SPU has no native version of shift left/right for i8
  setOperationAction(ISD::SHL,  MVT::i8,     Custom);
  setOperationAction(ISD::SRL,  MVT::i8,     Custom);
  setOperationAction(ISD::SRA,  MVT::i8,     Custom);

  // Make these operations legal and handle them during instruction selection:
  setOperationAction(ISD::SHL,  MVT::i64,    Legal);
  setOperationAction(ISD::SRL,  MVT::i64,    Legal);
  setOperationAction(ISD::SRA,  MVT::i64,    Legal);

  // Custom lower i8, i32 and i64 multiplications
  setOperationAction(ISD::MUL,  MVT::i8,     Custom);
  setOperationAction(ISD::MUL,  MVT::i32,    Legal);
  setOperationAction(ISD::MUL,  MVT::i64,    Legal);

  // Need to custom handle (some) common i8, i64 math ops
  setOperationAction(ISD::ADD,  MVT::i8,     Custom);
  setOperationAction(ISD::ADD,  MVT::i64,    Legal);
  setOperationAction(ISD::SUB,  MVT::i8,     Custom);
  setOperationAction(ISD::SUB,  MVT::i64,    Legal);

  // SPU does not have BSWAP. It does have i32 support CTLZ.
  // CTPOP has to be custom lowered.
  setOperationAction(ISD::BSWAP, MVT::i32,   Expand);
  setOperationAction(ISD::BSWAP, MVT::i64,   Expand);

  setOperationAction(ISD::CTPOP, MVT::i8,    Custom);
  setOperationAction(ISD::CTPOP, MVT::i16,   Custom);
  setOperationAction(ISD::CTPOP, MVT::i32,   Custom);
  setOperationAction(ISD::CTPOP, MVT::i64,   Custom);

  setOperationAction(ISD::CTTZ , MVT::i32,   Expand);
  setOperationAction(ISD::CTTZ , MVT::i64,   Expand);

  setOperationAction(ISD::CTLZ , MVT::i32,   Legal);

  // SPU has a version of select that implements (a&~c)|(b&c), just like
  // select ought to work:
  setOperationAction(ISD::SELECT, MVT::i8,   Legal);
  setOperationAction(ISD::SELECT, MVT::i16,  Legal);
  setOperationAction(ISD::SELECT, MVT::i32,  Legal);
  setOperationAction(ISD::SELECT, MVT::i64,  Legal);

  setOperationAction(ISD::SETCC, MVT::i8,    Legal);
  setOperationAction(ISD::SETCC, MVT::i16,   Legal);
  setOperationAction(ISD::SETCC, MVT::i32,   Legal);
  setOperationAction(ISD::SETCC, MVT::i64,   Legal);
  setOperationAction(ISD::SETCC, MVT::f64,   Custom);

  // Custom lower i128 -> i64 truncates
  setOperationAction(ISD::TRUNCATE, MVT::i64, Custom);

  // SPU has a legal FP -> signed INT instruction for f32, but for f64, need
  // to expand to a libcall, hence the custom lowering:
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);

  // FDIV on SPU requires custom lowering
  setOperationAction(ISD::FDIV, MVT::f64, Expand);      // to libcall

  // SPU has [U|S]INT_TO_FP for f32->i32, but not for f64->i32, f64->i64:
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i16, Promote);
  setOperationAction(ISD::SINT_TO_FP, MVT::i8,  Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i16, Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i8,  Promote);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);

  setOperationAction(ISD::BIT_CONVERT, MVT::i32, Legal);
  setOperationAction(ISD::BIT_CONVERT, MVT::f32, Legal);
  setOperationAction(ISD::BIT_CONVERT, MVT::i64, Legal);
  setOperationAction(ISD::BIT_CONVERT, MVT::f64, Legal);

  // We cannot sextinreg(i1).  Expand to shifts.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // Support label based line numbers.
  setOperationAction(ISD::DBG_STOPPOINT, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);

  // We want to legalize GlobalAddress and ConstantPool nodes into the
  // appropriate instructions to materialize the address.
  for (unsigned sctype = (unsigned) MVT::i8; sctype < (unsigned) MVT::f128;
       ++sctype) {
    MVT VT = (MVT::SimpleValueType)sctype;

    setOperationAction(ISD::GlobalAddress,  VT, Custom);
    setOperationAction(ISD::ConstantPool,   VT, Custom);
    setOperationAction(ISD::JumpTable,      VT, Custom);
  }

  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET,           MVT::Other, Custom);

  // VASTART needs to be custom lowered to use the VarArgsFrameIndex
  setOperationAction(ISD::VASTART           , MVT::Other, Custom);

  // Use the default implementation.
  setOperationAction(ISD::VAARG             , MVT::Other, Expand);
  setOperationAction(ISD::VACOPY            , MVT::Other, Expand);
  setOperationAction(ISD::VAEND             , MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE         , MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE      , MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i32  , Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64  , Expand);

  // Cell SPU has instructions for converting between i64 and fp.
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);

  // To take advantage of the above i64 FP_TO_SINT, promote i32 FP_TO_UINT
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Promote);

  // BUILD_PAIR can't be handled natively, and should be expanded to shl/or
  setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);

  // First set operation action for all vector types to expand. Then we
  // will selectively turn on ones that can be effectively codegen'd.
  addRegisterClass(MVT::v16i8, SPU::VECREGRegisterClass);
  addRegisterClass(MVT::v8i16, SPU::VECREGRegisterClass);
  addRegisterClass(MVT::v4i32, SPU::VECREGRegisterClass);
  addRegisterClass(MVT::v2i64, SPU::VECREGRegisterClass);
  addRegisterClass(MVT::v4f32, SPU::VECREGRegisterClass);
  addRegisterClass(MVT::v2f64, SPU::VECREGRegisterClass);

  // "Odd size" vector classes that we're willing to support:
  addRegisterClass(MVT::v2i32, SPU::VECREGRegisterClass);

  for (unsigned i = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
       i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
    MVT VT = (MVT::SimpleValueType)i;

    // add/sub are legal for all supported vector VT's.
    setOperationAction(ISD::ADD,     VT, Legal);
    setOperationAction(ISD::SUB,     VT, Legal);
    // mul has to be custom lowered.
    setOperationAction(ISD::MUL,     VT, Legal);

    setOperationAction(ISD::AND,     VT, Legal);
    setOperationAction(ISD::OR,      VT, Legal);
    setOperationAction(ISD::XOR,     VT, Legal);
    setOperationAction(ISD::LOAD,    VT, Legal);
    setOperationAction(ISD::SELECT,  VT, Legal);
    setOperationAction(ISD::STORE,   VT, Legal);

    // These operations need to be expanded:
    setOperationAction(ISD::SDIV,    VT, Expand);
    setOperationAction(ISD::SREM,    VT, Expand);
    setOperationAction(ISD::UDIV,    VT, Expand);
    setOperationAction(ISD::UREM,    VT, Expand);

    // Custom lower build_vector, constant pool spills, insert and
    // extract vector elements:
    setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
    setOperationAction(ISD::ConstantPool, VT, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR, VT, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
  }

  setOperationAction(ISD::AND, MVT::v16i8, Custom);
  setOperationAction(ISD::OR,  MVT::v16i8, Custom);
  setOperationAction(ISD::XOR, MVT::v16i8, Custom);
  setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4f32, Custom);

  setOperationAction(ISD::FDIV, MVT::v4f32, Legal);

  setShiftAmountType(MVT::i32);
  setBooleanContents(ZeroOrNegativeOneBooleanContent);

  setStackPointerRegisterToSaveRestore(SPU::R1);

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::SIGN_EXTEND);
  setTargetDAGCombine(ISD::ANY_EXTEND);

  computeRegisterProperties();

  // Set pre-RA register scheduler default to BURR, which produces slightly
  // better code than the default (could also be TDRR, but TargetLowering.h
  // needs a mod to support that model):
  setSchedulingPreference(SchedulingForRegPressure);
}

const char *
SPUTargetLowering::getTargetNodeName(unsigned Opcode) const
{
  if (node_names.empty()) {
    node_names[(unsigned) SPUISD::RET_FLAG] = "SPUISD::RET_FLAG";
    node_names[(unsigned) SPUISD::Hi] = "SPUISD::Hi";
    node_names[(unsigned) SPUISD::Lo] = "SPUISD::Lo";
    node_names[(unsigned) SPUISD::PCRelAddr] = "SPUISD::PCRelAddr";
    node_names[(unsigned) SPUISD::AFormAddr] = "SPUISD::AFormAddr";
    node_names[(unsigned) SPUISD::IndirectAddr] = "SPUISD::IndirectAddr";
    node_names[(unsigned) SPUISD::LDRESULT] = "SPUISD::LDRESULT";
    node_names[(unsigned) SPUISD::CALL] = "SPUISD::CALL";
    node_names[(unsigned) SPUISD::SHUFB] = "SPUISD::SHUFB";
    node_names[(unsigned) SPUISD::SHUFFLE_MASK] = "SPUISD::SHUFFLE_MASK";
    node_names[(unsigned) SPUISD::CNTB] = "SPUISD::CNTB";
    node_names[(unsigned) SPUISD::PREFSLOT2VEC] = "SPUISD::PREFSLOT2VEC";
    node_names[(unsigned) SPUISD::VEC2PREFSLOT] = "SPUISD::VEC2PREFSLOT";
    node_names[(unsigned) SPUISD::SHLQUAD_L_BITS] = "SPUISD::SHLQUAD_L_BITS";
    node_names[(unsigned) SPUISD::SHLQUAD_L_BYTES] = "SPUISD::SHLQUAD_L_BYTES";
    node_names[(unsigned) SPUISD::VEC_SHL] = "SPUISD::VEC_SHL";
    node_names[(unsigned) SPUISD::VEC_SRL] = "SPUISD::VEC_SRL";
    node_names[(unsigned) SPUISD::VEC_SRA] = "SPUISD::VEC_SRA";
    node_names[(unsigned) SPUISD::VEC_ROTL] = "SPUISD::VEC_ROTL";
    node_names[(unsigned) SPUISD::VEC_ROTR] = "SPUISD::VEC_ROTR";
    node_names[(unsigned) SPUISD::ROTBYTES_LEFT] = "SPUISD::ROTBYTES_LEFT";
    node_names[(unsigned) SPUISD::ROTBYTES_LEFT_BITS] =
            "SPUISD::ROTBYTES_LEFT_BITS";
    node_names[(unsigned) SPUISD::SELECT_MASK] = "SPUISD::SELECT_MASK";
    node_names[(unsigned) SPUISD::SELB] = "SPUISD::SELB";
    node_names[(unsigned) SPUISD::ADD64_MARKER] = "SPUISD::ADD64_MARKER";
    node_names[(unsigned) SPUISD::SUB64_MARKER] = "SPUISD::SUB64_MARKER";
    node_names[(unsigned) SPUISD::MUL64_MARKER] = "SPUISD::MUL64_MARKER";
  }

  std::map<unsigned, const char *>::iterator i = node_names.find(Opcode);

  return ((i != node_names.end()) ? i->second : 0);
}

//===----------------------------------------------------------------------===//
// Return the Cell SPU's SETCC result type
//===----------------------------------------------------------------------===//

MVT SPUTargetLowering::getSetCCResultType(MVT VT) const {
  // i16 and i32 are valid SETCC result types
  return ((VT == MVT::i8 || VT == MVT::i16 || VT == MVT::i32) ? VT : MVT::i32);
}

//===----------------------------------------------------------------------===//
// Calling convention code:
//===----------------------------------------------------------------------===//

#include "SPUGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//  LowerOperation implementation
//===----------------------------------------------------------------------===//

/// Custom lower loads for CellSPU
/*!
 All CellSPU loads and stores are aligned to 16-byte boundaries, so for elements
 within a 16-byte block, we have to rotate to extract the requested element.

 For extending loads, we also want to ensure that the following sequence is
 emitted, e.g. for MVT::f32 extending load to MVT::f64:

\verbatim
%1  v16i8,ch = load
%2  v16i8,ch = rotate %1
%3  v4f8, ch = bitconvert %2
%4  f32      = vec2perfslot %3
%5  f64      = fp_extend %4
\endverbatim
*/
static SDValue
LowerLOAD(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  LoadSDNode *LN = cast<LoadSDNode>(Op);
  SDValue the_chain = LN->getChain();
  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  MVT InVT = LN->getMemoryVT();
  MVT OutVT = Op.getValueType();
  ISD::LoadExtType ExtType = LN->getExtensionType();
  unsigned alignment = LN->getAlignment();
  const valtype_map_s *vtm = getValueTypeMapEntry(InVT);

  switch (LN->getAddressingMode()) {
  case ISD::UNINDEXED: {
    SDValue result;
    SDValue basePtr = LN->getBasePtr();
    SDValue rotate;

    if (alignment == 16) {
      ConstantSDNode *CN;

      // Special cases for a known aligned load to simplify the base pointer
      // and the rotation amount:
      if (basePtr.getOpcode() == ISD::ADD
          && (CN = dyn_cast<ConstantSDNode > (basePtr.getOperand(1))) != 0) {
        // Known offset into basePtr
        int64_t offset = CN->getSExtValue();
        int64_t rotamt = int64_t((offset & 0xf) - vtm->prefslot_byte);

        if (rotamt < 0)
          rotamt += 16;

        rotate = DAG.getConstant(rotamt, MVT::i16);

        // Simplify the base pointer for this case:
        basePtr = basePtr.getOperand(0);
        if ((offset & ~0xf) > 0) {
          basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                                basePtr,
                                DAG.getConstant((offset & ~0xf), PtrVT));
        }
      } else if ((basePtr.getOpcode() == SPUISD::AFormAddr)
                 || (basePtr.getOpcode() == SPUISD::IndirectAddr
                     && basePtr.getOperand(0).getOpcode() == SPUISD::Hi
                     && basePtr.getOperand(1).getOpcode() == SPUISD::Lo)) {
        // Plain aligned a-form address: rotate into preferred slot
        // Same for (SPUindirect (SPUhi ...), (SPUlo ...))
        int64_t rotamt = -vtm->prefslot_byte;
        if (rotamt < 0)
          rotamt += 16;
        rotate = DAG.getConstant(rotamt, MVT::i16);
      } else {
        // Offset the rotate amount by the basePtr and the preferred slot
        // byte offset
        int64_t rotamt = -vtm->prefslot_byte;
        if (rotamt < 0)
          rotamt += 16;
        rotate = DAG.getNode(ISD::ADD, PtrVT,
                             basePtr,
                             DAG.getConstant(rotamt, PtrVT));
      }
    } else {
      // Unaligned load: must be more pessimistic about addressing modes:
      if (basePtr.getOpcode() == ISD::ADD) {
        MachineFunction &MF = DAG.getMachineFunction();
        MachineRegisterInfo &RegInfo = MF.getRegInfo();
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
        SDValue Flag;

        SDValue Op0 = basePtr.getOperand(0);
        SDValue Op1 = basePtr.getOperand(1);

        if (isa<ConstantSDNode>(Op1)) {
          // Convert the (add <ptr>, <const>) to an indirect address contained
          // in a register. Note that this is done because we need to avoid
          // creating a 0(reg) d-form address due to the SPU's block loads.
          basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT, Op0, Op1);
          the_chain = DAG.getCopyToReg(the_chain, VReg, basePtr, Flag);
          basePtr = DAG.getCopyFromReg(the_chain, VReg, PtrVT);
        } else {
          // Convert the (add <arg1>, <arg2>) to an indirect address, which
          // will likely be lowered as a reg(reg) x-form address.
          basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT, Op0, Op1);
        }
      } else {
        basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                              basePtr,
                              DAG.getConstant(0, PtrVT));
      }

      // Offset the rotate amount by the basePtr and the preferred slot
      // byte offset
      rotate = DAG.getNode(ISD::ADD, PtrVT,
                           basePtr,
                           DAG.getConstant(-vtm->prefslot_byte, PtrVT));
    }

    // Re-emit as a v16i8 vector load
    result = DAG.getLoad(MVT::v16i8, the_chain, basePtr,
                         LN->getSrcValue(), LN->getSrcValueOffset(),
                         LN->isVolatile(), 16);

    // Update the chain
    the_chain = result.getValue(1);

    // Rotate into the preferred slot:
    result = DAG.getNode(SPUISD::ROTBYTES_LEFT, MVT::v16i8,
                         result.getValue(0), rotate);

    // Convert the loaded v16i8 vector to the appropriate vector type
    // specified by the operand:
    MVT vecVT = MVT::getVectorVT(InVT, (128 / InVT.getSizeInBits()));
    result = DAG.getNode(SPUISD::VEC2PREFSLOT, InVT,
                         DAG.getNode(ISD::BIT_CONVERT, vecVT, result));

    // Handle extending loads by extending the scalar result:
    if (ExtType == ISD::SEXTLOAD) {
      result = DAG.getNode(ISD::SIGN_EXTEND, OutVT, result);
    } else if (ExtType == ISD::ZEXTLOAD) {
      result = DAG.getNode(ISD::ZERO_EXTEND, OutVT, result);
    } else if (ExtType == ISD::EXTLOAD) {
      unsigned NewOpc = ISD::ANY_EXTEND;

      if (OutVT.isFloatingPoint())
        NewOpc = ISD::FP_EXTEND;

      result = DAG.getNode(NewOpc, OutVT, result);
    }

    SDVTList retvts = DAG.getVTList(OutVT, MVT::Other);
    SDValue retops[2] = {
      result,
      the_chain
    };

    result = DAG.getNode(SPUISD::LDRESULT, retvts,
                         retops, sizeof(retops) / sizeof(retops[0]));
    return result;
  }
  case ISD::PRE_INC:
  case ISD::PRE_DEC:
  case ISD::POST_INC:
  case ISD::POST_DEC:
  case ISD::LAST_INDEXED_MODE:
    cerr << "LowerLOAD: Got a LoadSDNode with an addr mode other than "
            "UNINDEXED\n";
    cerr << (unsigned) LN->getAddressingMode() << "\n";
    abort();
    /*NOTREACHED*/
  }

  return SDValue();
}

/// Custom lower stores for CellSPU
/*!
 All CellSPU stores are aligned to 16-byte boundaries, so for elements
 within a 16-byte block, we have to generate a shuffle to insert the
 requested element into its place, then store the resulting block.
 */
static SDValue
LowerSTORE(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  StoreSDNode *SN = cast<StoreSDNode>(Op);
  SDValue Value = SN->getValue();
  MVT VT = Value.getValueType();
  MVT StVT = (!SN->isTruncatingStore() ? VT : SN->getMemoryVT());
  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  unsigned alignment = SN->getAlignment();

  switch (SN->getAddressingMode()) {
  case ISD::UNINDEXED: {
    // The vector type we really want to load from the 16-byte chunk.
    MVT vecVT = MVT::getVectorVT(VT, (128 / VT.getSizeInBits())),
        stVecVT = MVT::getVectorVT(StVT, (128 / StVT.getSizeInBits()));

    SDValue alignLoadVec;
    SDValue basePtr = SN->getBasePtr();
    SDValue the_chain = SN->getChain();
    SDValue insertEltOffs;

    if (alignment == 16) {
      ConstantSDNode *CN;

      // Special cases for a known aligned load to simplify the base pointer
      // and insertion byte:
      if (basePtr.getOpcode() == ISD::ADD
          && (CN = dyn_cast<ConstantSDNode>(basePtr.getOperand(1))) != 0) {
        // Known offset into basePtr
        int64_t offset = CN->getSExtValue();

        // Simplify the base pointer for this case:
        basePtr = basePtr.getOperand(0);
        insertEltOffs = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                                    basePtr,
                                    DAG.getConstant((offset & 0xf), PtrVT));

        if ((offset & ~0xf) > 0) {
          basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                                basePtr,
                                DAG.getConstant((offset & ~0xf), PtrVT));
        }
      } else {
        // Otherwise, assume it's at byte 0 of basePtr
        insertEltOffs = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                                    basePtr,
                                    DAG.getConstant(0, PtrVT));
      }
    } else {
      // Unaligned load: must be more pessimistic about addressing modes:
      if (basePtr.getOpcode() == ISD::ADD) {
        MachineFunction &MF = DAG.getMachineFunction();
        MachineRegisterInfo &RegInfo = MF.getRegInfo();
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
        SDValue Flag;

        SDValue Op0 = basePtr.getOperand(0);
        SDValue Op1 = basePtr.getOperand(1);

        if (isa<ConstantSDNode>(Op1)) {
          // Convert the (add <ptr>, <const>) to an indirect address contained
          // in a register. Note that this is done because we need to avoid
          // creating a 0(reg) d-form address due to the SPU's block loads.
          basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT, Op0, Op1);
          the_chain = DAG.getCopyToReg(the_chain, VReg, basePtr, Flag);
          basePtr = DAG.getCopyFromReg(the_chain, VReg, PtrVT);
        } else {
          // Convert the (add <arg1>, <arg2>) to an indirect address, which
          // will likely be lowered as a reg(reg) x-form address.
          basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT, Op0, Op1);
        }
      } else {
        basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                              basePtr,
                              DAG.getConstant(0, PtrVT));
      }

      // Insertion point is solely determined by basePtr's contents
      insertEltOffs = DAG.getNode(ISD::ADD, PtrVT,
                                  basePtr,
                                  DAG.getConstant(0, PtrVT));
    }

    // Re-emit as a v16i8 vector load
    alignLoadVec = DAG.getLoad(MVT::v16i8, the_chain, basePtr,
                               SN->getSrcValue(), SN->getSrcValueOffset(),
                               SN->isVolatile(), 16);

    // Update the chain
    the_chain = alignLoadVec.getValue(1);

    LoadSDNode *LN = cast<LoadSDNode>(alignLoadVec);
    SDValue theValue = SN->getValue();
    SDValue result;

    if (StVT != VT
        && (theValue.getOpcode() == ISD::AssertZext
            || theValue.getOpcode() == ISD::AssertSext)) {
      // Drill down and get the value for zero- and sign-extended
      // quantities
      theValue = theValue.getOperand(0);
    }

    // If the base pointer is already a D-form address, then just create
    // a new D-form address with a slot offset and the orignal base pointer.
    // Otherwise generate a D-form address with the slot offset relative
    // to the stack pointer, which is always aligned.
#if !defined(NDEBUG)
      if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
        cerr << "CellSPU LowerSTORE: basePtr = ";
        basePtr.getNode()->dump(&DAG);
        cerr << "\n";
      }
#endif

    SDValue insertEltOp =
            DAG.getNode(SPUISD::SHUFFLE_MASK, vecVT, insertEltOffs);
    SDValue vectorizeOp =
            DAG.getNode(ISD::SCALAR_TO_VECTOR, vecVT, theValue);

    result = DAG.getNode(SPUISD::SHUFB, vecVT,
                         vectorizeOp, alignLoadVec,
                         DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, insertEltOp));

    result = DAG.getStore(the_chain, result, basePtr,
                          LN->getSrcValue(), LN->getSrcValueOffset(),
                          LN->isVolatile(), LN->getAlignment());

#if 0 && !defined(NDEBUG)
    if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
      const SDValue &currentRoot = DAG.getRoot();

      DAG.setRoot(result);
      cerr << "------- CellSPU:LowerStore result:\n";
      DAG.dump();
      cerr << "-------\n";
      DAG.setRoot(currentRoot);
    }
#endif

    return result;
    /*UNREACHED*/
  }
  case ISD::PRE_INC:
  case ISD::PRE_DEC:
  case ISD::POST_INC:
  case ISD::POST_DEC:
  case ISD::LAST_INDEXED_MODE:
    cerr << "LowerLOAD: Got a LoadSDNode with an addr mode other than "
            "UNINDEXED\n";
    cerr << (unsigned) SN->getAddressingMode() << "\n";
    abort();
    /*NOTREACHED*/
  }

  return SDValue();
}

//! Generate the address of a constant pool entry.
SDValue
LowerConstantPool(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  MVT PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  Constant *C = CP->getConstVal();
  SDValue CPI = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment());
  SDValue Zero = DAG.getConstant(0, PtrVT);
  const TargetMachine &TM = DAG.getTarget();

  if (TM.getRelocationModel() == Reloc::Static) {
    if (!ST->usingLargeMem()) {
      // Just return the SDValue with the constant pool address in it.
      return DAG.getNode(SPUISD::AFormAddr, PtrVT, CPI, Zero);
    } else {
      SDValue Hi = DAG.getNode(SPUISD::Hi, PtrVT, CPI, Zero);
      SDValue Lo = DAG.getNode(SPUISD::Lo, PtrVT, CPI, Zero);
      return DAG.getNode(SPUISD::IndirectAddr, PtrVT, Hi, Lo);
    }
  }

  assert(0 &&
         "LowerConstantPool: Relocation model other than static"
         " not supported.");
  return SDValue();
}

//! Alternate entry point for generating the address of a constant pool entry
SDValue
SPU::LowerConstantPool(SDValue Op, SelectionDAG &DAG, const SPUTargetMachine &TM) {
  return ::LowerConstantPool(Op, DAG, TM.getSubtargetImpl());
}

static SDValue
LowerJumpTable(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  MVT PtrVT = Op.getValueType();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDValue JTI = DAG.getTargetJumpTable(JT->getIndex(), PtrVT);
  SDValue Zero = DAG.getConstant(0, PtrVT);
  const TargetMachine &TM = DAG.getTarget();

  if (TM.getRelocationModel() == Reloc::Static) {
    if (!ST->usingLargeMem()) {
      return DAG.getNode(SPUISD::AFormAddr, PtrVT, JTI, Zero);
    } else {
      SDValue Hi = DAG.getNode(SPUISD::Hi, PtrVT, JTI, Zero);
      SDValue Lo = DAG.getNode(SPUISD::Lo, PtrVT, JTI, Zero);
      return DAG.getNode(SPUISD::IndirectAddr, PtrVT, Hi, Lo);
    }
  }

  assert(0 &&
         "LowerJumpTable: Relocation model other than static not supported.");
  return SDValue();
}

static SDValue
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  MVT PtrVT = Op.getValueType();
  GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
  GlobalValue *GV = GSDN->getGlobal();
  SDValue GA = DAG.getTargetGlobalAddress(GV, PtrVT, GSDN->getOffset());
  const TargetMachine &TM = DAG.getTarget();
  SDValue Zero = DAG.getConstant(0, PtrVT);

  if (TM.getRelocationModel() == Reloc::Static) {
    if (!ST->usingLargeMem()) {
      return DAG.getNode(SPUISD::AFormAddr, PtrVT, GA, Zero);
    } else {
      SDValue Hi = DAG.getNode(SPUISD::Hi, PtrVT, GA, Zero);
      SDValue Lo = DAG.getNode(SPUISD::Lo, PtrVT, GA, Zero);
      return DAG.getNode(SPUISD::IndirectAddr, PtrVT, Hi, Lo);
    }
  } else {
    cerr << "LowerGlobalAddress: Relocation model other than static not "
         << "supported.\n";
    abort();
    /*NOTREACHED*/
  }

  return SDValue();
}

//! Custom lower double precision floating point constants
static SDValue
LowerConstantFP(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();

  if (VT == MVT::f64) {
    ConstantFPSDNode *FP = cast<ConstantFPSDNode>(Op.getNode());

    assert((FP != 0) &&
           "LowerConstantFP: Node is not ConstantFPSDNode");

    uint64_t dbits = DoubleToBits(FP->getValueAPF().convertToDouble());
    SDValue T = DAG.getConstant(dbits, MVT::i64);
    SDValue Tvec = DAG.getNode(ISD::BUILD_VECTOR, MVT::v2i64, T, T);
    return DAG.getNode(SPUISD::VEC2PREFSLOT, VT,
                       DAG.getNode(ISD::BIT_CONVERT, MVT::v2f64, Tvec));
  }

  return SDValue();
}

static SDValue
LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG, int &VarArgsFrameIndex)
{
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SmallVector<SDValue, 48> ArgValues;
  SDValue Root = Op.getOperand(0);
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() != 0;

  const unsigned *ArgRegs = SPURegisterInfo::getArgRegs();
  const unsigned NumArgRegs = SPURegisterInfo::getNumArgRegs();

  unsigned ArgOffset = SPUFrameInfo::minStackSize();
  unsigned ArgRegIdx = 0;
  unsigned StackSlotSize = SPUFrameInfo::stackSlotSize();

  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  // Add DAG nodes to load the arguments or copy them out of registers.
  for (unsigned ArgNo = 0, e = Op.getNode()->getNumValues() - 1;
       ArgNo != e; ++ArgNo) {
    MVT ObjectVT = Op.getValue(ArgNo).getValueType();
    unsigned ObjSize = ObjectVT.getSizeInBits()/8;
    SDValue ArgVal;

    if (ArgRegIdx < NumArgRegs) {
      const TargetRegisterClass *ArgRegClass;

      switch (ObjectVT.getSimpleVT()) {
      default: {
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << ObjectVT.getMVTString()
             << "\n";
        abort();
      }
      case MVT::i8:
        ArgRegClass = &SPU::R8CRegClass;
        break;
      case MVT::i16:
        ArgRegClass = &SPU::R16CRegClass;
        break;
      case MVT::i32:
        ArgRegClass = &SPU::R32CRegClass;
        break;
      case MVT::i64:
        ArgRegClass = &SPU::R64CRegClass;
        break;
      case MVT::i128:
        ArgRegClass = &SPU::GPRCRegClass;
        break;
      case MVT::f32:
        ArgRegClass = &SPU::R32FPRegClass;
        break;
      case MVT::f64:
        ArgRegClass = &SPU::R64FPRegClass;
        break;
      case MVT::v2f64:
      case MVT::v4f32:
      case MVT::v2i64:
      case MVT::v4i32:
      case MVT::v8i16:
      case MVT::v16i8:
        ArgRegClass = &SPU::VECREGRegClass;
        break;
      }

      unsigned VReg = RegInfo.createVirtualRegister(ArgRegClass);
      RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
      ArgVal = DAG.getCopyFromReg(Root, VReg, ObjectVT);
      ++ArgRegIdx;
    } else {
      // We need to load the argument to a virtual register if we determined
      // above that we ran out of physical registers of the appropriate type
      // or we're forced to do vararg
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      ArgVal = DAG.getLoad(ObjectVT, Root, FIN, NULL, 0);
      ArgOffset += StackSlotSize;
    }

    ArgValues.push_back(ArgVal);
    // Update the chain
    Root = ArgVal.getOperand(0);
  }

  // vararg handling:
  if (isVarArg) {
    // unsigned int ptr_size = PtrVT.getSizeInBits() / 8;
    // We will spill (79-3)+1 registers to the stack
    SmallVector<SDValue, 79-3+1> MemOps;

    // Create the frame slot

    for (; ArgRegIdx != NumArgRegs; ++ArgRegIdx) {
      VarArgsFrameIndex = MFI->CreateFixedObject(StackSlotSize, ArgOffset);
      SDValue FIN = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
      SDValue ArgVal = DAG.getRegister(ArgRegs[ArgRegIdx], MVT::v16i8);
      SDValue Store = DAG.getStore(Root, ArgVal, FIN, NULL, 0);
      Root = Store.getOperand(0);
      MemOps.push_back(Store);

      // Increment address by stack slot size for the next stored argument
      ArgOffset += StackSlotSize;
    }
    if (!MemOps.empty())
      Root = DAG.getNode(ISD::TokenFactor,MVT::Other,&MemOps[0],MemOps.size());
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getNode(ISD::MERGE_VALUES, Op.getNode()->getVTList(),
                     &ArgValues[0], ArgValues.size());
}

/// isLSAAddress - Return the immediate to use if the specified
/// value is representable as a LSA address.
static SDNode *isLSAAddress(SDValue Op, SelectionDAG &DAG) {
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
  if (!C) return 0;

  int Addr = C->getZExtValue();
  if ((Addr & 3) != 0 ||  // Low 2 bits are implicitly zero.
      (Addr << 14 >> 14) != Addr)
    return 0;  // Top 14 bits have to be sext of immediate.

  return DAG.getConstant((int)C->getZExtValue() >> 2, MVT::i32).getNode();
}

static SDValue
LowerCALL(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  SDValue Chain = TheCall->getChain();
  SDValue Callee    = TheCall->getCallee();
  unsigned NumOps     = TheCall->getNumArgs();
  unsigned StackSlotSize = SPUFrameInfo::stackSlotSize();
  const unsigned *ArgRegs = SPURegisterInfo::getArgRegs();
  const unsigned NumArgRegs = SPURegisterInfo::getNumArgRegs();

  // Handy pointer type
  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();

  // Accumulate how many bytes are to be pushed on the stack, including the
  // linkage area, and parameter passing area.  According to the SPU ABI,
  // we minimally need space for [LR] and [SP]
  unsigned NumStackBytes = SPUFrameInfo::minStackSize();

  // Set up a copy of the stack pointer for use loading and storing any
  // arguments that may not fit in the registers available for argument
  // passing.
  SDValue StackPtr = DAG.getRegister(SPU::R1, MVT::i32);

  // Figure out which arguments are going to go in registers, and which in
  // memory.
  unsigned ArgOffset = SPUFrameInfo::minStackSize(); // Just below [LR]
  unsigned ArgRegIdx = 0;

  // Keep track of registers passing arguments
  std::vector<std::pair<unsigned, SDValue> > RegsToPass;
  // And the arguments passed on the stack
  SmallVector<SDValue, 8> MemOpChains;

  for (unsigned i = 0; i != NumOps; ++i) {
    SDValue Arg = TheCall->getArg(i);

    // PtrOff will be used to store the current argument to the stack if a
    // register cannot be found for it.
    SDValue PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
    PtrOff = DAG.getNode(ISD::ADD, PtrVT, StackPtr, PtrOff);

    switch (Arg.getValueType().getSimpleVT()) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
    case MVT::i128:
      if (ArgRegIdx != NumArgRegs) {
        RegsToPass.push_back(std::make_pair(ArgRegs[ArgRegIdx++], Arg));
      } else {
        MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
        ArgOffset += StackSlotSize;
      }
      break;
    case MVT::f32:
    case MVT::f64:
      if (ArgRegIdx != NumArgRegs) {
        RegsToPass.push_back(std::make_pair(ArgRegs[ArgRegIdx++], Arg));
      } else {
        MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
        ArgOffset += StackSlotSize;
      }
      break;
    case MVT::v2i64:
    case MVT::v2f64:
    case MVT::v4f32:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      if (ArgRegIdx != NumArgRegs) {
        RegsToPass.push_back(std::make_pair(ArgRegs[ArgRegIdx++], Arg));
      } else {
        MemOpChains.push_back(DAG.getStore(Chain, Arg, PtrOff, NULL, 0));
        ArgOffset += StackSlotSize;
      }
      break;
    }
  }

  // Update number of stack bytes actually used, insert a call sequence start
  NumStackBytes = (ArgOffset - SPUFrameInfo::minStackSize());
  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumStackBytes,
                                                            true));

  if (!MemOpChains.empty()) {
    // Adjust the stack pointer for the stack arguments.
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  SmallVector<SDValue, 8> Ops;
  unsigned CallOpc = SPUISD::CALL;

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    GlobalValue *GV = G->getGlobal();
    MVT CalleeVT = Callee.getValueType();
    SDValue Zero = DAG.getConstant(0, PtrVT);
    SDValue GA = DAG.getTargetGlobalAddress(GV, CalleeVT);

    if (!ST->usingLargeMem()) {
      // Turn calls to targets that are defined (i.e., have bodies) into BRSL
      // style calls, otherwise, external symbols are BRASL calls. This assumes
      // that declared/defined symbols are in the same compilation unit and can
      // be reached through PC-relative jumps.
      //
      // NOTE:
      // This may be an unsafe assumption for JIT and really large compilation
      // units.
      if (GV->isDeclaration()) {
        Callee = DAG.getNode(SPUISD::AFormAddr, CalleeVT, GA, Zero);
      } else {
        Callee = DAG.getNode(SPUISD::PCRelAddr, CalleeVT, GA, Zero);
      }
    } else {
      // "Large memory" mode: Turn all calls into indirect calls with a X-form
      // address pairs:
      Callee = DAG.getNode(SPUISD::IndirectAddr, PtrVT, GA, Zero);
    }
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    MVT CalleeVT = Callee.getValueType();
    SDValue Zero = DAG.getConstant(0, PtrVT);
    SDValue ExtSym = DAG.getTargetExternalSymbol(S->getSymbol(),
        Callee.getValueType());

    if (!ST->usingLargeMem()) {
      Callee = DAG.getNode(SPUISD::AFormAddr, CalleeVT, ExtSym, Zero);
    } else {
      Callee = DAG.getNode(SPUISD::IndirectAddr, PtrVT, ExtSym, Zero);
    }
  } else if (SDNode *Dest = isLSAAddress(Callee, DAG)) {
    // If this is an absolute destination address that appears to be a legal
    // local store address, use the munged value.
    Callee = SDValue(Dest, 0);
  }

  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.getNode())
    Ops.push_back(InFlag);
  // Returns a chain and a flag for retval copy to use.
  Chain = DAG.getNode(CallOpc, DAG.getVTList(MVT::Other, MVT::Flag),
                      &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumStackBytes, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  if (TheCall->getValueType(0) != MVT::Other)
    InFlag = Chain.getValue(1);

  SDValue ResultVals[3];
  unsigned NumResults = 0;

  // If the call has results, copy the values out of the ret val registers.
  switch (TheCall->getValueType(0).getSimpleVT()) {
  default: assert(0 && "Unexpected ret value!");
  case MVT::Other: break;
  case MVT::i32:
    if (TheCall->getValueType(1) == MVT::i32) {
      Chain = DAG.getCopyFromReg(Chain, SPU::R4, MVT::i32, InFlag).getValue(1);
      ResultVals[0] = Chain.getValue(0);
      Chain = DAG.getCopyFromReg(Chain, SPU::R3, MVT::i32,
                                 Chain.getValue(2)).getValue(1);
      ResultVals[1] = Chain.getValue(0);
      NumResults = 2;
    } else {
      Chain = DAG.getCopyFromReg(Chain, SPU::R3, MVT::i32, InFlag).getValue(1);
      ResultVals[0] = Chain.getValue(0);
      NumResults = 1;
    }
    break;
  case MVT::i64:
    Chain = DAG.getCopyFromReg(Chain, SPU::R3, MVT::i64, InFlag).getValue(1);
    ResultVals[0] = Chain.getValue(0);
    NumResults = 1;
    break;
  case MVT::i128:
    Chain = DAG.getCopyFromReg(Chain, SPU::R3, MVT::i128, InFlag).getValue(1);
    ResultVals[0] = Chain.getValue(0);
    NumResults = 1;
    break;
  case MVT::f32:
  case MVT::f64:
    Chain = DAG.getCopyFromReg(Chain, SPU::R3, TheCall->getValueType(0),
                               InFlag).getValue(1);
    ResultVals[0] = Chain.getValue(0);
    NumResults = 1;
    break;
  case MVT::v2f64:
  case MVT::v2i64:
  case MVT::v4f32:
  case MVT::v4i32:
  case MVT::v8i16:
  case MVT::v16i8:
    Chain = DAG.getCopyFromReg(Chain, SPU::R3, TheCall->getValueType(0),
                                   InFlag).getValue(1);
    ResultVals[0] = Chain.getValue(0);
    NumResults = 1;
    break;
  }

  // If the function returns void, just return the chain.
  if (NumResults == 0)
    return Chain;

  // Otherwise, merge everything together with a MERGE_VALUES node.
  ResultVals[NumResults++] = Chain;
  SDValue Res = DAG.getMergeValues(ResultVals, NumResults);
  return Res.getValue(Op.getResNo());
}

static SDValue
LowerRET(SDValue Op, SelectionDAG &DAG, TargetMachine &TM) {
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  CCState CCInfo(CC, isVarArg, TM, RVLocs);
  CCInfo.AnalyzeReturn(Op.getNode(), RetCC_SPU);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Chain = Op.getOperand(0);
  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    Chain = DAG.getCopyToReg(Chain, VA.getLocReg(), Op.getOperand(i*2+1), Flag);
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode())
    return DAG.getNode(SPUISD::RET_FLAG, MVT::Other, Chain, Flag);
  else
    return DAG.getNode(SPUISD::RET_FLAG, MVT::Other, Chain);
}


//===----------------------------------------------------------------------===//
// Vector related lowering:
//===----------------------------------------------------------------------===//

static ConstantSDNode *
getVecImm(SDNode *N) {
  SDValue OpVal(0, 0);

  // Check to see if this buildvec has a single non-undef value in its elements.
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (N->getOperand(i).getOpcode() == ISD::UNDEF) continue;
    if (OpVal.getNode() == 0)
      OpVal = N->getOperand(i);
    else if (OpVal != N->getOperand(i))
      return 0;
  }

  if (OpVal.getNode() != 0) {
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
      return CN;
    }
  }

  return 0; // All UNDEF: use implicit def.; not Constant node
}

/// get_vec_i18imm - Test if this vector is a vector filled with the same value
/// and the value fits into an unsigned 18-bit constant, and if so, return the
/// constant
SDValue SPU::get_vec_u18imm(SDNode *N, SelectionDAG &DAG,
                              MVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    uint64_t Value = CN->getZExtValue();
    if (ValueType == MVT::i64) {
      uint64_t UValue = CN->getZExtValue();
      uint32_t upper = uint32_t(UValue >> 32);
      uint32_t lower = uint32_t(UValue);
      if (upper != lower)
        return SDValue();
      Value = Value >> 32;
    }
    if (Value <= 0x3ffff)
      return DAG.getTargetConstant(Value, ValueType);
  }

  return SDValue();
}

/// get_vec_i16imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 16-bit constant, and if so, return the
/// constant
SDValue SPU::get_vec_i16imm(SDNode *N, SelectionDAG &DAG,
                              MVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    int64_t Value = CN->getSExtValue();
    if (ValueType == MVT::i64) {
      uint64_t UValue = CN->getZExtValue();
      uint32_t upper = uint32_t(UValue >> 32);
      uint32_t lower = uint32_t(UValue);
      if (upper != lower)
        return SDValue();
      Value = Value >> 32;
    }
    if (Value >= -(1 << 15) && Value <= ((1 << 15) - 1)) {
      return DAG.getTargetConstant(Value, ValueType);
    }
  }

  return SDValue();
}

/// get_vec_i10imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 10-bit constant, and if so, return the
/// constant
SDValue SPU::get_vec_i10imm(SDNode *N, SelectionDAG &DAG,
                              MVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    int64_t Value = CN->getSExtValue();
    if (ValueType == MVT::i64) {
      uint64_t UValue = CN->getZExtValue();
      uint32_t upper = uint32_t(UValue >> 32);
      uint32_t lower = uint32_t(UValue);
      if (upper != lower)
        return SDValue();
      Value = Value >> 32;
    }
    if (isS10Constant(Value))
      return DAG.getTargetConstant(Value, ValueType);
  }

  return SDValue();
}

/// get_vec_i8imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 8-bit constant, and if so, return the
/// constant.
///
/// @note: The incoming vector is v16i8 because that's the only way we can load
/// constant vectors. Thus, we test to see if the upper and lower bytes are the
/// same value.
SDValue SPU::get_vec_i8imm(SDNode *N, SelectionDAG &DAG,
                             MVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    int Value = (int) CN->getZExtValue();
    if (ValueType == MVT::i16
        && Value <= 0xffff                 /* truncated from uint64_t */
        && ((short) Value >> 8) == ((short) Value & 0xff))
      return DAG.getTargetConstant(Value & 0xff, ValueType);
    else if (ValueType == MVT::i8
             && (Value & 0xff) == Value)
      return DAG.getTargetConstant(Value, ValueType);
  }

  return SDValue();
}

/// get_ILHUvec_imm - Test if this vector is a vector filled with the same value
/// and the value fits into a signed 16-bit constant, and if so, return the
/// constant
SDValue SPU::get_ILHUvec_imm(SDNode *N, SelectionDAG &DAG,
                               MVT ValueType) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    uint64_t Value = CN->getZExtValue();
    if ((ValueType == MVT::i32
          && ((unsigned) Value & 0xffff0000) == (unsigned) Value)
        || (ValueType == MVT::i64 && (Value & 0xffff0000) == Value))
      return DAG.getTargetConstant(Value >> 16, ValueType);
  }

  return SDValue();
}

/// get_v4i32_imm - Catch-all for general 32-bit constant vectors
SDValue SPU::get_v4i32_imm(SDNode *N, SelectionDAG &DAG) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    return DAG.getTargetConstant((unsigned) CN->getZExtValue(), MVT::i32);
  }

  return SDValue();
}

/// get_v4i32_imm - Catch-all for general 64-bit constant vectors
SDValue SPU::get_v2i64_imm(SDNode *N, SelectionDAG &DAG) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    return DAG.getTargetConstant((unsigned) CN->getZExtValue(), MVT::i64);
  }

  return SDValue();
}

// If this is a vector of constants or undefs, get the bits.  A bit in
// UndefBits is set if the corresponding element of the vector is an
// ISD::UNDEF value.  For undefs, the corresponding VectorBits values are
// zero.   Return true if this is not an array of constants, false if it is.
//
static bool GetConstantBuildVectorBits(SDNode *BV, uint64_t VectorBits[2],
                                       uint64_t UndefBits[2]) {
  // Start with zero'd results.
  VectorBits[0] = VectorBits[1] = UndefBits[0] = UndefBits[1] = 0;

  unsigned EltBitSize = BV->getOperand(0).getValueType().getSizeInBits();
  for (unsigned i = 0, e = BV->getNumOperands(); i != e; ++i) {
    SDValue OpVal = BV->getOperand(i);

    unsigned PartNo = i >= e/2;     // In the upper 128 bits?
    unsigned SlotNo = e/2 - (i & (e/2-1))-1;  // Which subpiece of the uint64_t.

    uint64_t EltBits = 0;
    if (OpVal.getOpcode() == ISD::UNDEF) {
      uint64_t EltUndefBits = ~0ULL >> (64-EltBitSize);
      UndefBits[PartNo] |= EltUndefBits << (SlotNo*EltBitSize);
      continue;
    } else if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(OpVal)) {
      EltBits = CN->getZExtValue() & (~0ULL >> (64-EltBitSize));
    } else if (ConstantFPSDNode *CN = dyn_cast<ConstantFPSDNode>(OpVal)) {
      const APFloat &apf = CN->getValueAPF();
      EltBits = (CN->getValueType(0) == MVT::f32
                 ? FloatToBits(apf.convertToFloat())
                 : DoubleToBits(apf.convertToDouble()));
    } else {
      // Nonconstant element.
      return true;
    }

    VectorBits[PartNo] |= EltBits << (SlotNo*EltBitSize);
  }

  //printf("%llx %llx  %llx %llx\n",
  //       VectorBits[0], VectorBits[1], UndefBits[0], UndefBits[1]);
  return false;
}

/// If this is a splat (repetition) of a value across the whole vector, return
/// the smallest size that splats it.  For example, "0x01010101010101..." is a
/// splat of 0x01, 0x0101, and 0x01010101.  We return SplatBits = 0x01 and
/// SplatSize = 1 byte.
static bool isConstantSplat(const uint64_t Bits128[2],
                            const uint64_t Undef128[2],
                            int MinSplatBits,
                            uint64_t &SplatBits, uint64_t &SplatUndef,
                            int &SplatSize) {
  // Don't let undefs prevent splats from matching.  See if the top 64-bits are
  // the same as the lower 64-bits, ignoring undefs.
  uint64_t Bits64  = Bits128[0] | Bits128[1];
  uint64_t Undef64 = Undef128[0] & Undef128[1];
  uint32_t Bits32  = uint32_t(Bits64) | uint32_t(Bits64 >> 32);
  uint32_t Undef32 = uint32_t(Undef64) & uint32_t(Undef64 >> 32);
  uint16_t Bits16  = uint16_t(Bits32)  | uint16_t(Bits32 >> 16);
  uint16_t Undef16 = uint16_t(Undef32) & uint16_t(Undef32 >> 16);

  if ((Bits128[0] & ~Undef128[1]) == (Bits128[1] & ~Undef128[0])) {
    if (MinSplatBits < 64) {

      // Check that the top 32-bits are the same as the lower 32-bits, ignoring
      // undefs.
      if ((Bits64 & (~Undef64 >> 32)) == ((Bits64 >> 32) & ~Undef64)) {
        if (MinSplatBits < 32) {

          // If the top 16-bits are different than the lower 16-bits, ignoring
          // undefs, we have an i32 splat.
          if ((Bits32 & (~Undef32 >> 16)) == ((Bits32 >> 16) & ~Undef32)) {
            if (MinSplatBits < 16) {
              // If the top 8-bits are different than the lower 8-bits, ignoring
              // undefs, we have an i16 splat.
              if ((Bits16 & (uint16_t(~Undef16) >> 8))
                  == ((Bits16 >> 8) & ~Undef16)) {
                // Otherwise, we have an 8-bit splat.
                SplatBits  = uint8_t(Bits16)  | uint8_t(Bits16 >> 8);
                SplatUndef = uint8_t(Undef16) & uint8_t(Undef16 >> 8);
                SplatSize = 1;
                return true;
              }
            } else {
              SplatBits = Bits16;
              SplatUndef = Undef16;
              SplatSize = 2;
              return true;
            }
          }
        } else {
          SplatBits = Bits32;
          SplatUndef = Undef32;
          SplatSize = 4;
          return true;
        }
      }
    } else {
      SplatBits = Bits128[0];
      SplatUndef = Undef128[0];
      SplatSize = 8;
      return true;
    }
  }

  return false;  // Can't be a splat if two pieces don't match.
}

//! Lower a BUILD_VECTOR instruction creatively:
SDValue
LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  // If this is a vector of constants or undefs, get the bits.  A bit in
  // UndefBits is set if the corresponding element of the vector is an
  // ISD::UNDEF value.  For undefs, the corresponding VectorBits values are
  // zero.
  uint64_t VectorBits[2];
  uint64_t UndefBits[2];
  uint64_t SplatBits, SplatUndef;
  int SplatSize;
  if (GetConstantBuildVectorBits(Op.getNode(), VectorBits, UndefBits)
      || !isConstantSplat(VectorBits, UndefBits,
                          VT.getVectorElementType().getSizeInBits(),
                          SplatBits, SplatUndef, SplatSize))
    return SDValue();   // Not a constant vector, not a splat.

  switch (VT.getSimpleVT()) {
  default:
    cerr << "CellSPU: Unhandled VT in LowerBUILD_VECTOR, VT = "
         << VT.getMVTString()
         << "\n";
    abort();
    /*NOTREACHED*/
  case MVT::v4f32: {
    uint32_t Value32 = uint32_t(SplatBits);
    assert(SplatSize == 4
           && "LowerBUILD_VECTOR: Unexpected floating point vector element.");
    // NOTE: pretend the constant is an integer. LLVM won't load FP constants
    SDValue T = DAG.getConstant(Value32, MVT::i32);
    return DAG.getNode(ISD::BIT_CONVERT, MVT::v4f32,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, T, T, T, T));
    break;
  }
  case MVT::v2f64: {
    uint64_t f64val = uint64_t(SplatBits);
    assert(SplatSize == 8
           && "LowerBUILD_VECTOR: 64-bit float vector size > 8 bytes.");
    // NOTE: pretend the constant is an integer. LLVM won't load FP constants
    SDValue T = DAG.getConstant(f64val, MVT::i64);
    return DAG.getNode(ISD::BIT_CONVERT, MVT::v2f64,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v2i64, T, T));
    break;
  }
  case MVT::v16i8: {
   // 8-bit constants have to be expanded to 16-bits
   unsigned short Value16 = SplatBits | (SplatBits << 8);
   SDValue Ops[8];
   for (int i = 0; i < 8; ++i)
     Ops[i] = DAG.getConstant(Value16, MVT::i16);
   return DAG.getNode(ISD::BIT_CONVERT, VT,
                      DAG.getNode(ISD::BUILD_VECTOR, MVT::v8i16, Ops, 8));
  }
  case MVT::v8i16: {
    unsigned short Value16;
    if (SplatSize == 2)
      Value16 = (unsigned short) (SplatBits & 0xffff);
    else
      Value16 = (unsigned short) (SplatBits | (SplatBits << 8));
    SDValue T = DAG.getConstant(Value16, VT.getVectorElementType());
    SDValue Ops[8];
    for (int i = 0; i < 8; ++i) Ops[i] = T;
    return DAG.getNode(ISD::BUILD_VECTOR, VT, Ops, 8);
  }
  case MVT::v4i32: {
    unsigned int Value = SplatBits;
    SDValue T = DAG.getConstant(Value, VT.getVectorElementType());
    return DAG.getNode(ISD::BUILD_VECTOR, VT, T, T, T, T);
  }
  case MVT::v2i32: {
    unsigned int Value = SplatBits;
    SDValue T = DAG.getConstant(Value, VT.getVectorElementType());
    return DAG.getNode(ISD::BUILD_VECTOR, VT, T, T);
  }
  case MVT::v2i64: {
    return SPU::LowerSplat_v2i64(VT, DAG, SplatBits);
  }
  }

  return SDValue();
}

SDValue
SPU::LowerSplat_v2i64(MVT OpVT, SelectionDAG& DAG, uint64_t SplatVal) {
  uint32_t upper = uint32_t(SplatVal >> 32);
  uint32_t lower = uint32_t(SplatVal);

  if (upper == lower) {
    // Magic constant that can be matched by IL, ILA, et. al.
    SDValue Val = DAG.getTargetConstant(upper, MVT::i32);
    return DAG.getNode(ISD::BIT_CONVERT, OpVT,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                   Val, Val, Val, Val));
  } else {
    SDValue LO32;
    SDValue HI32;
    SmallVector<SDValue, 16> ShufBytes;
    SDValue Result;
    bool upper_special, lower_special;

    // NOTE: This code creates common-case shuffle masks that can be easily
    // detected as common expressions. It is not attempting to create highly
    // specialized masks to replace any and all 0's, 0xff's and 0x80's.

    // Detect if the upper or lower half is a special shuffle mask pattern:
    upper_special = (upper == 0 || upper == 0xffffffff || upper == 0x80000000);
    lower_special = (lower == 0 || lower == 0xffffffff || lower == 0x80000000);

    // Create lower vector if not a special pattern
    if (!lower_special) {
      SDValue LO32C = DAG.getConstant(lower, MVT::i32);
      LO32 = DAG.getNode(ISD::BIT_CONVERT, OpVT,
                         DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                     LO32C, LO32C, LO32C, LO32C));
    }

    // Create upper vector if not a special pattern
    if (!upper_special) {
      SDValue HI32C = DAG.getConstant(upper, MVT::i32);
      HI32 = DAG.getNode(ISD::BIT_CONVERT, OpVT,
                         DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                     HI32C, HI32C, HI32C, HI32C));
    }

    // If either upper or lower are special, then the two input operands are
    // the same (basically, one of them is a "don't care")
    if (lower_special)
      LO32 = HI32;
    if (upper_special)
      HI32 = LO32;
    if (lower_special && upper_special) {
      // Unhappy situation... both upper and lower are special, so punt with
      // a target constant:
      SDValue Zero = DAG.getConstant(0, MVT::i32);
      HI32 = LO32 = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, Zero, Zero,
                                Zero, Zero);
    }

    for (int i = 0; i < 4; ++i) {
      uint64_t val = 0;
      for (int j = 0; j < 4; ++j) {
        SDValue V;
        bool process_upper, process_lower;
        val <<= 8;
        process_upper = (upper_special && (i & 1) == 0);
        process_lower = (lower_special && (i & 1) == 1);

        if (process_upper || process_lower) {
          if ((process_upper && upper == 0)
                  || (process_lower && lower == 0))
            val |= 0x80;
          else if ((process_upper && upper == 0xffffffff)
                  || (process_lower && lower == 0xffffffff))
            val |= 0xc0;
          else if ((process_upper && upper == 0x80000000)
                  || (process_lower && lower == 0x80000000))
            val |= (j == 0 ? 0xe0 : 0x80);
        } else
          val |= i * 4 + j + ((i & 1) * 16);
      }

      ShufBytes.push_back(DAG.getConstant(val, MVT::i32));
    }

    return DAG.getNode(SPUISD::SHUFB, OpVT, HI32, LO32,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                   &ShufBytes[0], ShufBytes.size()));
  }
}

/// LowerVECTOR_SHUFFLE - Lower a vector shuffle (V1, V2, V3) to something on
/// which the Cell can operate. The code inspects V3 to ascertain whether the
/// permutation vector, V3, is monotonically increasing with one "exception"
/// element, e.g., (0, 1, _, 3). If this is the case, then generate a
/// SHUFFLE_MASK synthetic instruction. Otherwise, spill V3 to the constant pool.
/// In either case, the net result is going to eventually invoke SHUFB to
/// permute/shuffle the bytes from V1 and V2.
/// \note
/// SHUFFLE_MASK is eventually selected as one of the C*D instructions, generate
/// control word for byte/halfword/word insertion. This takes care of a single
/// element move from V2 into V1.
/// \note
/// SPUISD::SHUFB is eventually selected as Cell's <i>shufb</i> instructions.
static SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) {
  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  SDValue PermMask = Op.getOperand(2);

  if (V2.getOpcode() == ISD::UNDEF) V2 = V1;

  // If we have a single element being moved from V1 to V2, this can be handled
  // using the C*[DX] compute mask instructions, but the vector elements have
  // to be monotonically increasing with one exception element.
  MVT VecVT = V1.getValueType();
  MVT EltVT = VecVT.getVectorElementType();
  unsigned EltsFromV2 = 0;
  unsigned V2Elt = 0;
  unsigned V2EltIdx0 = 0;
  unsigned CurrElt = 0;
  unsigned MaxElts = VecVT.getVectorNumElements();
  unsigned PrevElt = 0;
  unsigned V0Elt = 0;
  bool monotonic = true;
  bool rotate = true;

  if (EltVT == MVT::i8) {
    V2EltIdx0 = 16;
  } else if (EltVT == MVT::i16) {
    V2EltIdx0 = 8;
  } else if (EltVT == MVT::i32 || EltVT == MVT::f32) {
    V2EltIdx0 = 4;
  } else if (EltVT == MVT::i64 || EltVT == MVT::f64) {
    V2EltIdx0 = 2;
  } else
    assert(0 && "Unhandled vector type in LowerVECTOR_SHUFFLE");

  for (unsigned i = 0; i != PermMask.getNumOperands(); ++i) {
    if (PermMask.getOperand(i).getOpcode() != ISD::UNDEF) {
      unsigned SrcElt = cast<ConstantSDNode > (PermMask.getOperand(i))->getZExtValue();

      if (monotonic) {
        if (SrcElt >= V2EltIdx0) {
          if (1 >= (++EltsFromV2)) {
            V2Elt = (V2EltIdx0 - SrcElt) << 2;
          }
        } else if (CurrElt != SrcElt) {
          monotonic = false;
        }

        ++CurrElt;
      }

      if (rotate) {
        if (PrevElt > 0 && SrcElt < MaxElts) {
          if ((PrevElt == SrcElt - 1)
              || (PrevElt == MaxElts - 1 && SrcElt == 0)) {
            PrevElt = SrcElt;
            if (SrcElt == 0)
              V0Elt = i;
          } else {
            rotate = false;
          }
        } else if (PrevElt == 0) {
          // First time through, need to keep track of previous element
          PrevElt = SrcElt;
        } else {
          // This isn't a rotation, takes elements from vector 2
          rotate = false;
        }
      }
    }
  }

  if (EltsFromV2 == 1 && monotonic) {
    // Compute mask and shuffle
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
    MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
    // Initialize temporary register to 0
    SDValue InitTempReg =
      DAG.getCopyToReg(DAG.getEntryNode(), VReg, DAG.getConstant(0, PtrVT));
    // Copy register's contents as index in SHUFFLE_MASK:
    SDValue ShufMaskOp =
      DAG.getNode(SPUISD::SHUFFLE_MASK, MVT::v4i32,
                  DAG.getTargetConstant(V2Elt, MVT::i32),
                  DAG.getCopyFromReg(InitTempReg, VReg, PtrVT));
    // Use shuffle mask in SHUFB synthetic instruction:
    return DAG.getNode(SPUISD::SHUFB, V1.getValueType(), V2, V1, ShufMaskOp);
  } else if (rotate) {
    int rotamt = (MaxElts - V0Elt) * EltVT.getSizeInBits()/8;

    return DAG.getNode(SPUISD::ROTBYTES_LEFT, V1.getValueType(),
                       V1, DAG.getConstant(rotamt, MVT::i16));
  } else {
   // Convert the SHUFFLE_VECTOR mask's input element units to the
   // actual bytes.
    unsigned BytesPerElement = EltVT.getSizeInBits()/8;

    SmallVector<SDValue, 16> ResultMask;
    for (unsigned i = 0, e = PermMask.getNumOperands(); i != e; ++i) {
      unsigned SrcElt;
      if (PermMask.getOperand(i).getOpcode() == ISD::UNDEF)
        SrcElt = 0;
      else
        SrcElt = cast<ConstantSDNode>(PermMask.getOperand(i))->getZExtValue();

      for (unsigned j = 0; j < BytesPerElement; ++j) {
        ResultMask.push_back(DAG.getConstant(SrcElt*BytesPerElement+j,
                                             MVT::i8));
      }
    }

    SDValue VPermMask = DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8,
                                    &ResultMask[0], ResultMask.size());
    return DAG.getNode(SPUISD::SHUFB, V1.getValueType(), V1, V2, VPermMask);
  }
}

static SDValue LowerSCALAR_TO_VECTOR(SDValue Op, SelectionDAG &DAG) {
  SDValue Op0 = Op.getOperand(0);                     // Op0 = the scalar

  if (Op0.getNode()->getOpcode() == ISD::Constant) {
    // For a constant, build the appropriate constant vector, which will
    // eventually simplify to a vector register load.

    ConstantSDNode *CN = cast<ConstantSDNode>(Op0.getNode());
    SmallVector<SDValue, 16> ConstVecValues;
    MVT VT;
    size_t n_copies;

    // Create a constant vector:
    switch (Op.getValueType().getSimpleVT()) {
    default: assert(0 && "Unexpected constant value type in "
                         "LowerSCALAR_TO_VECTOR");
    case MVT::v16i8: n_copies = 16; VT = MVT::i8; break;
    case MVT::v8i16: n_copies = 8; VT = MVT::i16; break;
    case MVT::v4i32: n_copies = 4; VT = MVT::i32; break;
    case MVT::v4f32: n_copies = 4; VT = MVT::f32; break;
    case MVT::v2i64: n_copies = 2; VT = MVT::i64; break;
    case MVT::v2f64: n_copies = 2; VT = MVT::f64; break;
    }

    SDValue CValue = DAG.getConstant(CN->getZExtValue(), VT);
    for (size_t j = 0; j < n_copies; ++j)
      ConstVecValues.push_back(CValue);

    return DAG.getNode(ISD::BUILD_VECTOR, Op.getValueType(),
                       &ConstVecValues[0], ConstVecValues.size());
  } else {
    // Otherwise, copy the value from one register to another:
    switch (Op0.getValueType().getSimpleVT()) {
    default: assert(0 && "Unexpected value type in LowerSCALAR_TO_VECTOR");
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
    case MVT::f32:
    case MVT::f64:
      return DAG.getNode(SPUISD::PREFSLOT2VEC, Op.getValueType(), Op0, Op0);
    }
  }

  return SDValue();
}

static SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  SDValue N = Op.getOperand(0);
  SDValue Elt = Op.getOperand(1);
  SDValue retval;

  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Elt)) {
    // Constant argument:
    int EltNo = (int) C->getZExtValue();

    // sanity checks:
    if (VT == MVT::i8 && EltNo >= 16)
      assert(0 && "SPU LowerEXTRACT_VECTOR_ELT: i8 extraction slot > 15");
    else if (VT == MVT::i16 && EltNo >= 8)
      assert(0 && "SPU LowerEXTRACT_VECTOR_ELT: i16 extraction slot > 7");
    else if (VT == MVT::i32 && EltNo >= 4)
      assert(0 && "SPU LowerEXTRACT_VECTOR_ELT: i32 extraction slot > 4");
    else if (VT == MVT::i64 && EltNo >= 2)
      assert(0 && "SPU LowerEXTRACT_VECTOR_ELT: i64 extraction slot > 2");

    if (EltNo == 0 && (VT == MVT::i32 || VT == MVT::i64)) {
      // i32 and i64: Element 0 is the preferred slot
      return DAG.getNode(SPUISD::VEC2PREFSLOT, VT, N);
    }

    // Need to generate shuffle mask and extract:
    int prefslot_begin = -1, prefslot_end = -1;
    int elt_byte = EltNo * VT.getSizeInBits() / 8;

    switch (VT.getSimpleVT()) {
    default:
      assert(false && "Invalid value type!");
    case MVT::i8: {
      prefslot_begin = prefslot_end = 3;
      break;
    }
    case MVT::i16: {
      prefslot_begin = 2; prefslot_end = 3;
      break;
    }
    case MVT::i32:
    case MVT::f32: {
      prefslot_begin = 0; prefslot_end = 3;
      break;
    }
    case MVT::i64:
    case MVT::f64: {
      prefslot_begin = 0; prefslot_end = 7;
      break;
    }
    }

    assert(prefslot_begin != -1 && prefslot_end != -1 &&
           "LowerEXTRACT_VECTOR_ELT: preferred slots uninitialized");

    unsigned int ShufBytes[16];
    for (int i = 0; i < 16; ++i) {
      // zero fill uppper part of preferred slot, don't care about the
      // other slots:
      unsigned int mask_val;
      if (i <= prefslot_end) {
        mask_val =
          ((i < prefslot_begin)
           ? 0x80
           : elt_byte + (i - prefslot_begin));

        ShufBytes[i] = mask_val;
      } else
        ShufBytes[i] = ShufBytes[i % (prefslot_end + 1)];
    }

    SDValue ShufMask[4];
    for (unsigned i = 0; i < sizeof(ShufMask)/sizeof(ShufMask[0]); ++i) {
      unsigned bidx = i * 4;
      unsigned int bits = ((ShufBytes[bidx] << 24) |
                           (ShufBytes[bidx+1] << 16) |
                           (ShufBytes[bidx+2] << 8) |
                           ShufBytes[bidx+3]);
      ShufMask[i] = DAG.getConstant(bits, MVT::i32);
    }

    SDValue ShufMaskVec = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                      &ShufMask[0],
                                      sizeof(ShufMask) / sizeof(ShufMask[0]));

    retval = DAG.getNode(SPUISD::VEC2PREFSLOT, VT,
                         DAG.getNode(SPUISD::SHUFB, N.getValueType(),
                                     N, N, ShufMaskVec));
  } else {
    // Variable index: Rotate the requested element into slot 0, then replicate
    // slot 0 across the vector
    MVT VecVT = N.getValueType();
    if (!VecVT.isSimple() || !VecVT.isVector() || !VecVT.is128BitVector()) {
      cerr << "LowerEXTRACT_VECTOR_ELT: Must have a simple, 128-bit vector type!\n";
      abort();
    }

    // Make life easier by making sure the index is zero-extended to i32
    if (Elt.getValueType() != MVT::i32)
      Elt = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, Elt);

    // Scale the index to a bit/byte shift quantity
    APInt scaleFactor =
            APInt(32, uint64_t(16 / N.getValueType().getVectorNumElements()), false);
    unsigned scaleShift = scaleFactor.logBase2();
    SDValue vecShift;

    if (scaleShift > 0) {
      // Scale the shift factor:
      Elt = DAG.getNode(ISD::SHL, MVT::i32, Elt,
                        DAG.getConstant(scaleShift, MVT::i32));
    }

    vecShift = DAG.getNode(SPUISD::SHLQUAD_L_BYTES, VecVT, N, Elt);

    // Replicate the bytes starting at byte 0 across the entire vector (for
    // consistency with the notion of a unified register set)
    SDValue replicate;

    switch (VT.getSimpleVT()) {
    default:
      cerr << "LowerEXTRACT_VECTOR_ELT(varable): Unhandled vector type\n";
      abort();
      /*NOTREACHED*/
    case MVT::i8: {
      SDValue factor = DAG.getConstant(0x00000000, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, factor, factor,
                              factor, factor);
      break;
    }
    case MVT::i16: {
      SDValue factor = DAG.getConstant(0x00010001, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, factor, factor,
                              factor, factor);
      break;
    }
    case MVT::i32:
    case MVT::f32: {
      SDValue factor = DAG.getConstant(0x00010203, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, factor, factor,
                              factor, factor);
      break;
    }
    case MVT::i64:
    case MVT::f64: {
      SDValue loFactor = DAG.getConstant(0x00010203, MVT::i32);
      SDValue hiFactor = DAG.getConstant(0x04050607, MVT::i32);
      replicate = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, loFactor, hiFactor,
                              loFactor, hiFactor);
      break;
    }
    }

    retval = DAG.getNode(SPUISD::VEC2PREFSLOT, VT,
                         DAG.getNode(SPUISD::SHUFB, VecVT,
                                     vecShift, vecShift, replicate));
  }

  return retval;
}

static SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  SDValue VecOp = Op.getOperand(0);
  SDValue ValOp = Op.getOperand(1);
  SDValue IdxOp = Op.getOperand(2);
  MVT VT = Op.getValueType();

  ConstantSDNode *CN = cast<ConstantSDNode>(IdxOp);
  assert(CN != 0 && "LowerINSERT_VECTOR_ELT: Index is not constant!");

  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  // Use $sp ($1) because it's always 16-byte aligned and it's available:
  SDValue Pointer = DAG.getNode(SPUISD::IndirectAddr, PtrVT,
                                DAG.getRegister(SPU::R1, PtrVT),
                                DAG.getConstant(CN->getSExtValue(), PtrVT));
  SDValue ShufMask = DAG.getNode(SPUISD::SHUFFLE_MASK, VT, Pointer);

  SDValue result =
    DAG.getNode(SPUISD::SHUFB, VT,
                DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, ValOp),
                VecOp,
                DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, ShufMask));

  return result;
}

static SDValue LowerI8Math(SDValue Op, SelectionDAG &DAG, unsigned Opc,
                           const TargetLowering &TLI)
{
  SDValue N0 = Op.getOperand(0);      // Everything has at least one operand
  MVT ShiftVT = TLI.getShiftAmountTy();

  assert(Op.getValueType() == MVT::i8);
  switch (Opc) {
  default:
    assert(0 && "Unhandled i8 math operator");
    /*NOTREACHED*/
    break;
  case ISD::ADD: {
    // 8-bit addition: Promote the arguments up to 16-bits and truncate
    // the result:
    SDValue N1 = Op.getOperand(1);
    N0 = DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N0);
    N1 = DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N1);
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));

  }

  case ISD::SUB: {
    // 8-bit subtraction: Promote the arguments up to 16-bits and truncate
    // the result:
    SDValue N1 = Op.getOperand(1);
    N0 = DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N0);
    N1 = DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N1);
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));
  }
  case ISD::ROTR:
  case ISD::ROTL: {
    SDValue N1 = Op.getOperand(1);
    unsigned N1Opc;
    N0 = (N0.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::ZERO_EXTEND, MVT::i16, N0)
          : DAG.getConstant(cast<ConstantSDNode>(N0)->getZExtValue(),
                            MVT::i16));
    N1Opc = N1.getValueType().bitsLT(ShiftVT)
            ? ISD::ZERO_EXTEND
            : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, ShiftVT, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            TLI.getShiftAmountTy()));
    SDValue ExpandArg =
      DAG.getNode(ISD::OR, MVT::i16, N0,
                  DAG.getNode(ISD::SHL, MVT::i16,
                              N0, DAG.getConstant(8, MVT::i32)));
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, ExpandArg, N1));
  }
  case ISD::SRL:
  case ISD::SHL: {
    SDValue N1 = Op.getOperand(1);
    unsigned N1Opc;
    N0 = (N0.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::ZERO_EXTEND, MVT::i16, N0)
          : DAG.getConstant(cast<ConstantSDNode>(N0)->getZExtValue(),
                            MVT::i32));
    N1Opc = N1.getValueType().bitsLT(ShiftVT)
            ? ISD::ZERO_EXTEND
            : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, ShiftVT, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(), ShiftVT));
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));
  }
  case ISD::SRA: {
    SDValue N1 = Op.getOperand(1);
    unsigned N1Opc;
    N0 = (N0.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N0)
          : DAG.getConstant(cast<ConstantSDNode>(N0)->getSExtValue(),
                            MVT::i16));
    N1Opc = N1.getValueType().bitsLT(ShiftVT)
            ? ISD::SIGN_EXTEND
            : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, ShiftVT, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            ShiftVT));
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));
  }
  case ISD::MUL: {
    SDValue N1 = Op.getOperand(1);
    unsigned N1Opc;
    N0 = (N0.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N0)
          : DAG.getConstant(cast<ConstantSDNode>(N0)->getZExtValue(),
                            MVT::i16));
    N1Opc = N1.getValueType().bitsLT(MVT::i16) ? ISD::SIGN_EXTEND : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, MVT::i16, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getSExtValue(),
                            MVT::i16));
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));
    break;
  }
  }

  return SDValue();
}

//! Generate the carry-generate shuffle mask.
SDValue SPU::getCarryGenerateShufMask(SelectionDAG &DAG) {
  SmallVector<SDValue, 16 > ShufBytes;

  // Create the shuffle mask for "rotating" the borrow up one register slot
  // once the borrow is generated.
  ShufBytes.push_back(DAG.getConstant(0x04050607, MVT::i32));
  ShufBytes.push_back(DAG.getConstant(0x80808080, MVT::i32));
  ShufBytes.push_back(DAG.getConstant(0x0c0d0e0f, MVT::i32));
  ShufBytes.push_back(DAG.getConstant(0x80808080, MVT::i32));

  return DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                     &ShufBytes[0], ShufBytes.size());
}

//! Generate the borrow-generate shuffle mask
SDValue SPU::getBorrowGenerateShufMask(SelectionDAG &DAG) {
  SmallVector<SDValue, 16 > ShufBytes;

  // Create the shuffle mask for "rotating" the borrow up one register slot
  // once the borrow is generated.
  ShufBytes.push_back(DAG.getConstant(0x04050607, MVT::i32));
  ShufBytes.push_back(DAG.getConstant(0xc0c0c0c0, MVT::i32));
  ShufBytes.push_back(DAG.getConstant(0x0c0d0e0f, MVT::i32));
  ShufBytes.push_back(DAG.getConstant(0xc0c0c0c0, MVT::i32));

  return DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                     &ShufBytes[0], ShufBytes.size());
}

//! Lower byte immediate operations for v16i8 vectors:
static SDValue
LowerByteImmed(SDValue Op, SelectionDAG &DAG) {
  SDValue ConstVec;
  SDValue Arg;
  MVT VT = Op.getValueType();

  ConstVec = Op.getOperand(0);
  Arg = Op.getOperand(1);
  if (ConstVec.getNode()->getOpcode() != ISD::BUILD_VECTOR) {
    if (ConstVec.getNode()->getOpcode() == ISD::BIT_CONVERT) {
      ConstVec = ConstVec.getOperand(0);
    } else {
      ConstVec = Op.getOperand(1);
      Arg = Op.getOperand(0);
      if (ConstVec.getNode()->getOpcode() == ISD::BIT_CONVERT) {
        ConstVec = ConstVec.getOperand(0);
      }
    }
  }

  if (ConstVec.getNode()->getOpcode() == ISD::BUILD_VECTOR) {
    uint64_t VectorBits[2];
    uint64_t UndefBits[2];
    uint64_t SplatBits, SplatUndef;
    int SplatSize;

    if (!GetConstantBuildVectorBits(ConstVec.getNode(), VectorBits, UndefBits)
        && isConstantSplat(VectorBits, UndefBits,
                           VT.getVectorElementType().getSizeInBits(),
                           SplatBits, SplatUndef, SplatSize)) {
      SDValue tcVec[16];
      SDValue tc = DAG.getTargetConstant(SplatBits & 0xff, MVT::i8);
      const size_t tcVecSize = sizeof(tcVec) / sizeof(tcVec[0]);

      // Turn the BUILD_VECTOR into a set of target constants:
      for (size_t i = 0; i < tcVecSize; ++i)
        tcVec[i] = tc;

      return DAG.getNode(Op.getNode()->getOpcode(), VT, Arg,
                         DAG.getNode(ISD::BUILD_VECTOR, VT, tcVec, tcVecSize));
    }
  }

  // These operations (AND, OR, XOR) are legal, they just couldn't be custom
  // lowered.  Return the operation, rather than a null SDValue.
  return Op;
}

//! Custom lowering for CTPOP (count population)
/*!
  Custom lowering code that counts the number ones in the input
  operand. SPU has such an instruction, but it counts the number of
  ones per byte, which then have to be accumulated.
*/
static SDValue LowerCTPOP(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  MVT vecVT = MVT::getVectorVT(VT, (128 / VT.getSizeInBits()));

  switch (VT.getSimpleVT()) {
  default:
    assert(false && "Invalid value type!");
  case MVT::i8: {
    SDValue N = Op.getOperand(0);
    SDValue Elt0 = DAG.getConstant(0, MVT::i32);

    SDValue Promote = DAG.getNode(SPUISD::PREFSLOT2VEC, vecVT, N, N);
    SDValue CNTB = DAG.getNode(SPUISD::CNTB, vecVT, Promote);

    return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i8, CNTB, Elt0);
  }

  case MVT::i16: {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();

    unsigned CNTB_reg = RegInfo.createVirtualRegister(&SPU::R16CRegClass);

    SDValue N = Op.getOperand(0);
    SDValue Elt0 = DAG.getConstant(0, MVT::i16);
    SDValue Mask0 = DAG.getConstant(0x0f, MVT::i16);
    SDValue Shift1 = DAG.getConstant(8, MVT::i32);

    SDValue Promote = DAG.getNode(SPUISD::PREFSLOT2VEC, vecVT, N, N);
    SDValue CNTB = DAG.getNode(SPUISD::CNTB, vecVT, Promote);

    // CNTB_result becomes the chain to which all of the virtual registers
    // CNTB_reg, SUM1_reg become associated:
    SDValue CNTB_result =
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i16, CNTB, Elt0);

    SDValue CNTB_rescopy =
      DAG.getCopyToReg(CNTB_result, CNTB_reg, CNTB_result);

    SDValue Tmp1 = DAG.getCopyFromReg(CNTB_rescopy, CNTB_reg, MVT::i16);

    return DAG.getNode(ISD::AND, MVT::i16,
                       DAG.getNode(ISD::ADD, MVT::i16,
                                   DAG.getNode(ISD::SRL, MVT::i16,
                                               Tmp1, Shift1),
                                   Tmp1),
                       Mask0);
  }

  case MVT::i32: {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();

    unsigned CNTB_reg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
    unsigned SUM1_reg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);

    SDValue N = Op.getOperand(0);
    SDValue Elt0 = DAG.getConstant(0, MVT::i32);
    SDValue Mask0 = DAG.getConstant(0xff, MVT::i32);
    SDValue Shift1 = DAG.getConstant(16, MVT::i32);
    SDValue Shift2 = DAG.getConstant(8, MVT::i32);

    SDValue Promote = DAG.getNode(SPUISD::PREFSLOT2VEC, vecVT, N, N);
    SDValue CNTB = DAG.getNode(SPUISD::CNTB, vecVT, Promote);

    // CNTB_result becomes the chain to which all of the virtual registers
    // CNTB_reg, SUM1_reg become associated:
    SDValue CNTB_result =
      DAG.getNode(ISD::EXTRACT_VECTOR_ELT, MVT::i32, CNTB, Elt0);

    SDValue CNTB_rescopy =
      DAG.getCopyToReg(CNTB_result, CNTB_reg, CNTB_result);

    SDValue Comp1 =
      DAG.getNode(ISD::SRL, MVT::i32,
                  DAG.getCopyFromReg(CNTB_rescopy, CNTB_reg, MVT::i32), Shift1);

    SDValue Sum1 =
      DAG.getNode(ISD::ADD, MVT::i32,
                  Comp1, DAG.getCopyFromReg(CNTB_rescopy, CNTB_reg, MVT::i32));

    SDValue Sum1_rescopy =
      DAG.getCopyToReg(CNTB_result, SUM1_reg, Sum1);

    SDValue Comp2 =
      DAG.getNode(ISD::SRL, MVT::i32,
                  DAG.getCopyFromReg(Sum1_rescopy, SUM1_reg, MVT::i32),
                  Shift2);
    SDValue Sum2 =
      DAG.getNode(ISD::ADD, MVT::i32, Comp2,
                  DAG.getCopyFromReg(Sum1_rescopy, SUM1_reg, MVT::i32));

    return DAG.getNode(ISD::AND, MVT::i32, Sum2, Mask0);
  }

  case MVT::i64:
    break;
  }

  return SDValue();
}

//! Lower ISD::FP_TO_SINT, ISD::FP_TO_UINT for i32
/*!
 f32->i32 passes through unchanged, whereas f64->i32 expands to a libcall.
 All conversions to i64 are expanded to a libcall.
 */
static SDValue LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG,
                              SPUTargetLowering &TLI) {
  MVT OpVT = Op.getValueType();
  SDValue Op0 = Op.getOperand(0);
  MVT Op0VT = Op0.getValueType();

  if ((OpVT == MVT::i32 && Op0VT == MVT::f64)
      || OpVT == MVT::i64) {
    // Convert f32 / f64 to i32 / i64 via libcall.
    RTLIB::Libcall LC =
            (Op.getOpcode() == ISD::FP_TO_SINT)
             ? RTLIB::getFPTOSINT(Op0VT, OpVT)
             : RTLIB::getFPTOUINT(Op0VT, OpVT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpectd fp-to-int conversion!");
    SDValue Dummy;
    return ExpandLibCall(LC, Op, DAG, false, Dummy, TLI);
  }

  return Op;                    // return unmolested, legalized op
}

//! Lower ISD::SINT_TO_FP, ISD::UINT_TO_FP for i32
/*!
 i32->f32 passes through unchanged, whereas i32->f64 is expanded to a libcall.
 All conversions from i64 are expanded to a libcall.
 */
static SDValue LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG,
                              SPUTargetLowering &TLI) {
  MVT OpVT = Op.getValueType();
  SDValue Op0 = Op.getOperand(0);
  MVT Op0VT = Op0.getValueType();

  if ((OpVT == MVT::f64 && Op0VT == MVT::i32)
      || Op0VT == MVT::i64) {
    // Convert i32, i64 to f64 via libcall:
    RTLIB::Libcall LC =
            (Op.getOpcode() == ISD::SINT_TO_FP)
             ? RTLIB::getSINTTOFP(Op0VT, OpVT)
             : RTLIB::getUINTTOFP(Op0VT, OpVT);
    assert(LC != RTLIB::UNKNOWN_LIBCALL && "Unexpectd int-to-fp conversion!");
    SDValue Dummy;
    return ExpandLibCall(LC, Op, DAG, false, Dummy, TLI);
  }

  return Op;                    // return unmolested, legalized
}

//! Lower ISD::SETCC
/*!
 This handles MVT::f64 (double floating point) condition lowering
 */
static SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG,
                          const TargetLowering &TLI) {
  CondCodeSDNode *CC = dyn_cast<CondCodeSDNode>(Op.getOperand(2));
  assert(CC != 0 && "LowerSETCC: CondCodeSDNode should not be null here!\n");

  SDValue lhs = Op.getOperand(0);
  SDValue rhs = Op.getOperand(1);
  MVT lhsVT = lhs.getValueType();
  assert(lhsVT == MVT::f64 && "LowerSETCC: type other than MVT::64\n");

  MVT ccResultVT = TLI.getSetCCResultType(lhs.getValueType());
  APInt ccResultOnes = APInt::getAllOnesValue(ccResultVT.getSizeInBits());
  MVT IntVT(MVT::i64);

  // Take advantage of the fact that (truncate (sra arg, 32)) is efficiently
  // selected to a NOP:
  SDValue i64lhs = DAG.getNode(ISD::BIT_CONVERT, IntVT, lhs);
  SDValue lhsHi32 =
          DAG.getNode(ISD::TRUNCATE, MVT::i32,
                      DAG.getNode(ISD::SRL, IntVT,
                                  i64lhs, DAG.getConstant(32, MVT::i32)));
  SDValue lhsHi32abs =
          DAG.getNode(ISD::AND, MVT::i32,
                      lhsHi32, DAG.getConstant(0x7fffffff, MVT::i32));
  SDValue lhsLo32 =
          DAG.getNode(ISD::TRUNCATE, MVT::i32, i64lhs);

  // SETO and SETUO only use the lhs operand:
  if (CC->get() == ISD::SETO) {
    // Evaluates to true if Op0 is not [SQ]NaN - lowers to the inverse of
    // SETUO
    APInt ccResultAllOnes = APInt::getAllOnesValue(ccResultVT.getSizeInBits());
    return DAG.getNode(ISD::XOR, ccResultVT,
                       DAG.getSetCC(ccResultVT,
                                    lhs, DAG.getConstantFP(0.0, lhsVT),
                                    ISD::SETUO),
                       DAG.getConstant(ccResultAllOnes, ccResultVT));
  } else if (CC->get() == ISD::SETUO) {
    // Evaluates to true if Op0 is [SQ]NaN
    return DAG.getNode(ISD::AND, ccResultVT,
                       DAG.getSetCC(ccResultVT,
                                    lhsHi32abs,
                                    DAG.getConstant(0x7ff00000, MVT::i32),
                                    ISD::SETGE),
                       DAG.getSetCC(ccResultVT,
                                    lhsLo32,
                                    DAG.getConstant(0, MVT::i32),
                                    ISD::SETGT));
  }

  SDValue i64rhs = DAG.getNode(ISD::BIT_CONVERT, IntVT, rhs);
  SDValue rhsHi32 =
          DAG.getNode(ISD::TRUNCATE, MVT::i32,
                      DAG.getNode(ISD::SRL, IntVT,
                                  i64rhs, DAG.getConstant(32, MVT::i32)));

  // If a value is negative, subtract from the sign magnitude constant:
  SDValue signMag2TC = DAG.getConstant(0x8000000000000000ULL, IntVT);

  // Convert the sign-magnitude representation into 2's complement:
  SDValue lhsSelectMask = DAG.getNode(ISD::SRA, ccResultVT,
                                      lhsHi32, DAG.getConstant(31, MVT::i32));
  SDValue lhsSignMag2TC = DAG.getNode(ISD::SUB, IntVT, signMag2TC, i64lhs);
  SDValue lhsSelect =
          DAG.getNode(ISD::SELECT, IntVT,
                      lhsSelectMask, lhsSignMag2TC, i64lhs);

  SDValue rhsSelectMask = DAG.getNode(ISD::SRA, ccResultVT,
                                      rhsHi32, DAG.getConstant(31, MVT::i32));
  SDValue rhsSignMag2TC = DAG.getNode(ISD::SUB, IntVT, signMag2TC, i64rhs);
  SDValue rhsSelect =
          DAG.getNode(ISD::SELECT, IntVT,
                      rhsSelectMask, rhsSignMag2TC, i64rhs);

  unsigned compareOp;

  switch (CC->get()) {
  case ISD::SETOEQ:
  case ISD::SETUEQ:
    compareOp = ISD::SETEQ; break;
  case ISD::SETOGT:
  case ISD::SETUGT:
    compareOp = ISD::SETGT; break;
  case ISD::SETOGE:
  case ISD::SETUGE:
    compareOp = ISD::SETGE; break;
  case ISD::SETOLT:
  case ISD::SETULT:
    compareOp = ISD::SETLT; break;
  case ISD::SETOLE:
  case ISD::SETULE:
    compareOp = ISD::SETLE; break;
  case ISD::SETUNE:
  case ISD::SETONE:
    compareOp = ISD::SETNE; break;
  default:
    cerr << "CellSPU ISel Select: unimplemented f64 condition\n";
    abort();
    break;
  }

  SDValue result =
          DAG.getSetCC(ccResultVT, lhsSelect, rhsSelect, (ISD::CondCode) compareOp);

  if ((CC->get() & 0x8) == 0) {
    // Ordered comparison:
    SDValue lhsNaN = DAG.getSetCC(ccResultVT,
                                  lhs, DAG.getConstantFP(0.0, MVT::f64),
                                  ISD::SETO);
    SDValue rhsNaN = DAG.getSetCC(ccResultVT,
                                  rhs, DAG.getConstantFP(0.0, MVT::f64),
                                  ISD::SETO);
    SDValue ordered = DAG.getNode(ISD::AND, ccResultVT, lhsNaN, rhsNaN);

    result = DAG.getNode(ISD::AND, ccResultVT, ordered, result);
  }

  return result;
}

//! Lower ISD::SELECT_CC
/*!
  ISD::SELECT_CC can (generally) be implemented directly on the SPU using the
  SELB instruction.

  \note Need to revisit this in the future: if the code path through the true
  and false value computations is longer than the latency of a branch (6
  cycles), then it would be more advantageous to branch and insert a new basic
  block and branch on the condition. However, this code does not make that
  assumption, given the simplisitc uses so far.
 */

static SDValue LowerSELECT_CC(SDValue Op, SelectionDAG &DAG,
                              const TargetLowering &TLI) {
  MVT VT = Op.getValueType();
  SDValue lhs = Op.getOperand(0);
  SDValue rhs = Op.getOperand(1);
  SDValue trueval = Op.getOperand(2);
  SDValue falseval = Op.getOperand(3);
  SDValue condition = Op.getOperand(4);

  // NOTE: SELB's arguments: $rA, $rB, $mask
  //
  // SELB selects bits from $rA where bits in $mask are 0, bits from $rB
  // where bits in $mask are 1. CCond will be inverted, having 1s where the
  // condition was true and 0s where the condition was false. Hence, the
  // arguments to SELB get reversed.

  // Note: Really should be ISD::SELECT instead of SPUISD::SELB, but LLVM's
  // legalizer insists on combining SETCC/SELECT into SELECT_CC, so we end up
  // with another "cannot select select_cc" assert:

  SDValue compare = DAG.getNode(ISD::SETCC,
                                TLI.getSetCCResultType(Op.getValueType()),
                                lhs, rhs, condition);
  return DAG.getNode(SPUISD::SELB, VT, falseval, trueval, compare);
}

//! Custom lower ISD::TRUNCATE
static SDValue LowerTRUNCATE(SDValue Op, SelectionDAG &DAG)
{
  MVT VT = Op.getValueType();
  MVT::SimpleValueType simpleVT = VT.getSimpleVT();
  MVT VecVT = MVT::getVectorVT(VT, (128 / VT.getSizeInBits()));

  SDValue Op0 = Op.getOperand(0);
  MVT Op0VT = Op0.getValueType();
  MVT Op0VecVT = MVT::getVectorVT(Op0VT, (128 / Op0VT.getSizeInBits()));

  if (Op0VT.getSimpleVT() == MVT::i128 && simpleVT == MVT::i64) {
    // Create shuffle mask, least significant doubleword of quadword
    unsigned maskHigh = 0x08090a0b;
    unsigned maskLow = 0x0c0d0e0f;
    // Use a shuffle to perform the truncation
    SDValue shufMask = DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                   DAG.getConstant(maskHigh, MVT::i32),
                                   DAG.getConstant(maskLow, MVT::i32),
                                   DAG.getConstant(maskHigh, MVT::i32),
                                   DAG.getConstant(maskLow, MVT::i32));


    SDValue PromoteScalar = DAG.getNode(SPUISD::PREFSLOT2VEC, Op0VecVT, Op0);

    SDValue truncShuffle = DAG.getNode(SPUISD::SHUFB, Op0VecVT,
                                       PromoteScalar, PromoteScalar, shufMask);

    return DAG.getNode(SPUISD::VEC2PREFSLOT, VT,
                       DAG.getNode(ISD::BIT_CONVERT, VecVT, truncShuffle));
  }

  return SDValue();             // Leave the truncate unmolested
}

//! Custom (target-specific) lowering entry point
/*!
  This is where LLVM's DAG selection process calls to do target-specific
  lowering of nodes.
 */
SDValue
SPUTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG)
{
  unsigned Opc = (unsigned) Op.getOpcode();
  MVT VT = Op.getValueType();

  switch (Opc) {
  default: {
    cerr << "SPUTargetLowering::LowerOperation(): need to lower this!\n";
    cerr << "Op.getOpcode() = " << Opc << "\n";
    cerr << "*Op.getNode():\n";
    Op.getNode()->dump();
    abort();
  }
  case ISD::LOAD:
  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
    return LowerLOAD(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::STORE:
    return LowerSTORE(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::ConstantFP:
    return LowerConstantFP(Op, DAG);
  case ISD::FORMAL_ARGUMENTS:
    return LowerFORMAL_ARGUMENTS(Op, DAG, VarArgsFrameIndex);
  case ISD::CALL:
    return LowerCALL(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::RET:
    return LowerRET(Op, DAG, getTargetMachine());

  // i8, i64 math ops:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::ROTR:
  case ISD::ROTL:
  case ISD::SRL:
  case ISD::SHL:
  case ISD::SRA: {
    if (VT == MVT::i8)
      return LowerI8Math(Op, DAG, Opc, *this);
    break;
  }

  case ISD::FP_TO_SINT:
  case ISD::FP_TO_UINT:
    return LowerFP_TO_INT(Op, DAG, *this);

  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return LowerINT_TO_FP(Op, DAG, *this);

  // Vector-related lowering.
  case ISD::BUILD_VECTOR:
    return LowerBUILD_VECTOR(Op, DAG);
  case ISD::SCALAR_TO_VECTOR:
    return LowerSCALAR_TO_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
    return LowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:
    return LowerINSERT_VECTOR_ELT(Op, DAG);

  // Look for ANDBI, ORBI and XORBI opportunities and lower appropriately:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    return LowerByteImmed(Op, DAG);

  // Vector and i8 multiply:
  case ISD::MUL:
    if (VT == MVT::i8)
      return LowerI8Math(Op, DAG, Opc, *this);

  case ISD::CTPOP:
    return LowerCTPOP(Op, DAG);

  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG, *this);

  case ISD::SETCC:
    return LowerSETCC(Op, DAG, *this);

  case ISD::TRUNCATE:
    return LowerTRUNCATE(Op, DAG);
  }

  return SDValue();
}

void SPUTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue>&Results,
                                           SelectionDAG &DAG)
{
#if 0
  unsigned Opc = (unsigned) N->getOpcode();
  MVT OpVT = N->getValueType(0);

  switch (Opc) {
  default: {
    cerr << "SPUTargetLowering::ReplaceNodeResults(): need to fix this!\n";
    cerr << "Op.getOpcode() = " << Opc << "\n";
    cerr << "*Op.getNode():\n";
    N->dump();
    abort();
    /*NOTREACHED*/
  }
  }
#endif

  /* Otherwise, return unchanged */
}

//===----------------------------------------------------------------------===//
// Target Optimization Hooks
//===----------------------------------------------------------------------===//

SDValue
SPUTargetLowering::PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const
{
#if 0
  TargetMachine &TM = getTargetMachine();
#endif
  const SPUSubtarget *ST = SPUTM.getSubtargetImpl();
  SelectionDAG &DAG = DCI.DAG;
  SDValue Op0 = N->getOperand(0);       // everything has at least one operand
  MVT NodeVT = N->getValueType(0);      // The node's value type
  MVT Op0VT = Op0.getValueType();       // The first operand's result
  SDValue Result;                       // Initially, empty result

  switch (N->getOpcode()) {
  default: break;
  case ISD::ADD: {
    SDValue Op1 = N->getOperand(1);

    if (Op0.getOpcode() == SPUISD::IndirectAddr
        || Op1.getOpcode() == SPUISD::IndirectAddr) {
      // Normalize the operands to reduce repeated code
      SDValue IndirectArg = Op0, AddArg = Op1;

      if (Op1.getOpcode() == SPUISD::IndirectAddr) {
        IndirectArg = Op1;
        AddArg = Op0;
      }

      if (isa<ConstantSDNode>(AddArg)) {
        ConstantSDNode *CN0 = cast<ConstantSDNode > (AddArg);
        SDValue IndOp1 = IndirectArg.getOperand(1);

        if (CN0->isNullValue()) {
          // (add (SPUindirect <arg>, <arg>), 0) ->
          // (SPUindirect <arg>, <arg>)

#if !defined(NDEBUG)
          if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
            cerr << "\n"
                 << "Replace: (add (SPUindirect <arg>, <arg>), 0)\n"
                 << "With:    (SPUindirect <arg>, <arg>)\n";
          }
#endif

          return IndirectArg;
        } else if (isa<ConstantSDNode>(IndOp1)) {
          // (add (SPUindirect <arg>, <const>), <const>) ->
          // (SPUindirect <arg>, <const + const>)
          ConstantSDNode *CN1 = cast<ConstantSDNode > (IndOp1);
          int64_t combinedConst = CN0->getSExtValue() + CN1->getSExtValue();
          SDValue combinedValue = DAG.getConstant(combinedConst, Op0VT);

#if !defined(NDEBUG)
          if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
            cerr << "\n"
                 << "Replace: (add (SPUindirect <arg>, " << CN1->getSExtValue()
                 << "), " << CN0->getSExtValue() << ")\n"
                 << "With:    (SPUindirect <arg>, "
                 << combinedConst << ")\n";
          }
#endif

          return DAG.getNode(SPUISD::IndirectAddr, Op0VT,
                             IndirectArg, combinedValue);
        }
      }
    }
    break;
  }
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND: {
    if (Op0.getOpcode() == SPUISD::VEC2PREFSLOT && NodeVT == Op0VT) {
      // (any_extend (SPUextract_elt0 <arg>)) ->
      // (SPUextract_elt0 <arg>)
      // Types must match, however...
#if !defined(NDEBUG)
      if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
        cerr << "\nReplace: ";
        N->dump(&DAG);
        cerr << "\nWith:    ";
        Op0.getNode()->dump(&DAG);
        cerr << "\n";
      }
#endif

      return Op0;
    }
    break;
  }
  case SPUISD::IndirectAddr: {
    if (!ST->usingLargeMem() && Op0.getOpcode() == SPUISD::AFormAddr) {
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N->getOperand(1));
      if (CN != 0 && CN->getZExtValue() == 0) {
        // (SPUindirect (SPUaform <addr>, 0), 0) ->
        // (SPUaform <addr>, 0)

        DEBUG(cerr << "Replace: ");
        DEBUG(N->dump(&DAG));
        DEBUG(cerr << "\nWith:    ");
        DEBUG(Op0.getNode()->dump(&DAG));
        DEBUG(cerr << "\n");

        return Op0;
      }
    } else if (Op0.getOpcode() == ISD::ADD) {
      SDValue Op1 = N->getOperand(1);
      if (ConstantSDNode *CN1 = dyn_cast<ConstantSDNode>(Op1)) {
        // (SPUindirect (add <arg>, <arg>), 0) ->
        // (SPUindirect <arg>, <arg>)
        if (CN1->isNullValue()) {

#if !defined(NDEBUG)
          if (DebugFlag && isCurrentDebugType(DEBUG_TYPE)) {
            cerr << "\n"
                 << "Replace: (SPUindirect (add <arg>, <arg>), 0)\n"
                 << "With:    (SPUindirect <arg>, <arg>)\n";
          }
#endif

          return DAG.getNode(SPUISD::IndirectAddr, Op0VT,
                             Op0.getOperand(0), Op0.getOperand(1));
        }
      }
    }
    break;
  }
  case SPUISD::SHLQUAD_L_BITS:
  case SPUISD::SHLQUAD_L_BYTES:
  case SPUISD::VEC_SHL:
  case SPUISD::VEC_SRL:
  case SPUISD::VEC_SRA:
  case SPUISD::ROTBYTES_LEFT: {
    SDValue Op1 = N->getOperand(1);

    // Kill degenerate vector shifts:
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1)) {
      if (CN->isNullValue()) {
        Result = Op0;
      }
    }
    break;
  }
  case SPUISD::PREFSLOT2VEC: {
    switch (Op0.getOpcode()) {
    default:
      break;
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::SIGN_EXTEND: {
      // (SPUprefslot2vec (any|zero|sign_extend (SPUvec2prefslot <arg>))) ->
      // <arg>
      // but only if the SPUprefslot2vec and <arg> types match.
      SDValue Op00 = Op0.getOperand(0);
      if (Op00.getOpcode() == SPUISD::VEC2PREFSLOT) {
        SDValue Op000 = Op00.getOperand(0);
        if (Op000.getValueType() == NodeVT) {
          Result = Op000;
        }
      }
      break;
    }
    case SPUISD::VEC2PREFSLOT: {
      // (SPUprefslot2vec (SPUvec2prefslot <arg>)) ->
      // <arg>
      Result = Op0.getOperand(0);
      break;
    }
    }
    break;
  }
  }

  // Otherwise, return unchanged.
#ifndef NDEBUG
  if (Result.getNode()) {
    DEBUG(cerr << "\nReplace.SPU: ");
    DEBUG(N->dump(&DAG));
    DEBUG(cerr << "\nWith:        ");
    DEBUG(Result.getNode()->dump(&DAG));
    DEBUG(cerr << "\n");
  }
#endif

  return Result;
}

//===----------------------------------------------------------------------===//
// Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
SPUTargetLowering::ConstraintType
SPUTargetLowering::getConstraintType(const std::string &ConstraintLetter) const {
  if (ConstraintLetter.size() == 1) {
    switch (ConstraintLetter[0]) {
    default: break;
    case 'b':
    case 'r':
    case 'f':
    case 'v':
    case 'y':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(ConstraintLetter);
}

std::pair<unsigned, const TargetRegisterClass*>
SPUTargetLowering::getRegForInlineAsmConstraint(const std::string &Constraint,
                                                MVT VT) const
{
  if (Constraint.size() == 1) {
    // GCC RS6000 Constraint Letters
    switch (Constraint[0]) {
    case 'b':   // R1-R31
    case 'r':   // R0-R31
      if (VT == MVT::i64)
        return std::make_pair(0U, SPU::R64CRegisterClass);
      return std::make_pair(0U, SPU::R32CRegisterClass);
    case 'f':
      if (VT == MVT::f32)
        return std::make_pair(0U, SPU::R32FPRegisterClass);
      else if (VT == MVT::f64)
        return std::make_pair(0U, SPU::R64FPRegisterClass);
      break;
    case 'v':
      return std::make_pair(0U, SPU::GPRCRegisterClass);
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

//! Compute used/known bits for a SPU operand
void
SPUTargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                  const APInt &Mask,
                                                  APInt &KnownZero,
                                                  APInt &KnownOne,
                                                  const SelectionDAG &DAG,
                                                  unsigned Depth ) const {
#if 0
  const uint64_t uint64_sizebits = sizeof(uint64_t) * 8;

  switch (Op.getOpcode()) {
  default:
    // KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);
    break;
  case CALL:
  case SHUFB:
  case SHUFFLE_MASK:
  case CNTB:
  case SPUISD::PREFSLOT2VEC:
  case SPUISD::LDRESULT:
  case SPUISD::VEC2PREFSLOT:
  case SPUISD::SHLQUAD_L_BITS:
  case SPUISD::SHLQUAD_L_BYTES:
  case SPUISD::VEC_SHL:
  case SPUISD::VEC_SRL:
  case SPUISD::VEC_SRA:
  case SPUISD::VEC_ROTL:
  case SPUISD::VEC_ROTR:
  case SPUISD::ROTBYTES_LEFT:
  case SPUISD::SELECT_MASK:
  case SPUISD::SELB:
  }
#endif
}

unsigned
SPUTargetLowering::ComputeNumSignBitsForTargetNode(SDValue Op,
                                                   unsigned Depth) const {
  switch (Op.getOpcode()) {
  default:
    return 1;

  case ISD::SETCC: {
    MVT VT = Op.getValueType();

    if (VT != MVT::i8 && VT != MVT::i16 && VT != MVT::i32) {
      VT = MVT::i32;
    }
    return VT.getSizeInBits();
  }
  }
}

// LowerAsmOperandForConstraint
void
SPUTargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                char ConstraintLetter,
                                                bool hasMemory,
                                                std::vector<SDValue> &Ops,
                                                SelectionDAG &DAG) const {
  // Default, for the time being, to the base class handler
  TargetLowering::LowerAsmOperandForConstraint(Op, ConstraintLetter, hasMemory,
                                               Ops, DAG);
}

/// isLegalAddressImmediate - Return true if the integer value can be used
/// as the offset of the target addressing mode.
bool SPUTargetLowering::isLegalAddressImmediate(int64_t V,
                                                const Type *Ty) const {
  // SPU's addresses are 256K:
  return (V > -(1 << 18) && V < (1 << 18) - 1);
}

bool SPUTargetLowering::isLegalAddressImmediate(llvm::GlobalValue* GV) const {
  return false;
}

bool
SPUTargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The SPU target isn't yet aware of offsets.
  return false;
}
