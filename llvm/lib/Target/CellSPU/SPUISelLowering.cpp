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
#include "llvm/ADT/VectorExtras.h"
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
    const MVT        valtype;
    const int                   prefslot_byte;
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

  //! Predicate that returns true if operand is a memory target
  /*!
    \arg Op Operand to test
    \return true if the operand is a memory target (i.e., global
    address, external symbol, constant pool) or an A-form
    address.
   */
  bool isMemoryOperand(const SDValue &Op)
  {
    const unsigned Opc = Op.getOpcode();
    return (Opc == ISD::GlobalAddress
            || Opc == ISD::GlobalTLSAddress
            || Opc == ISD::JumpTable
            || Opc == ISD::ConstantPool
            || Opc == ISD::ExternalSymbol
            || Opc == ISD::TargetGlobalAddress
            || Opc == ISD::TargetGlobalTLSAddress
            || Opc == ISD::TargetJumpTable
            || Opc == ISD::TargetConstantPool
            || Opc == ISD::TargetExternalSymbol
            || Opc == SPUISD::AFormAddr);
  }

  //! Predicate that returns true if the operand is an indirect target
  bool isIndirectOperand(const SDValue &Op)
  {
    const unsigned Opc = Op.getOpcode();
    return (Opc == ISD::Register
            || Opc == SPUISD::LDRESULT);
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

  // Set up the SPU's register classes:
  addRegisterClass(MVT::i8,   SPU::R8CRegisterClass);
  addRegisterClass(MVT::i16,  SPU::R16CRegisterClass);
  addRegisterClass(MVT::i32,  SPU::R32CRegisterClass);
  addRegisterClass(MVT::i64,  SPU::R64CRegisterClass);
  addRegisterClass(MVT::f32,  SPU::R32FPRegisterClass);
  addRegisterClass(MVT::f64,  SPU::R64FPRegisterClass);
  addRegisterClass(MVT::i128, SPU::GPRCRegisterClass);

  // SPU has no sign or zero extended loads for i1, i8, i16:
  setLoadXAction(ISD::EXTLOAD,  MVT::i1, Promote);
  setLoadXAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadXAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setTruncStoreAction(MVT::i8, MVT::i1, Custom);
  setTruncStoreAction(MVT::i16, MVT::i1, Custom);
  setTruncStoreAction(MVT::i32, MVT::i1, Custom);
  setTruncStoreAction(MVT::i64, MVT::i1, Custom);
  setTruncStoreAction(MVT::i128, MVT::i1, Custom);

  setLoadXAction(ISD::EXTLOAD,  MVT::i8, Custom);
  setLoadXAction(ISD::SEXTLOAD, MVT::i8, Custom);
  setLoadXAction(ISD::ZEXTLOAD, MVT::i8, Custom);
  setTruncStoreAction(MVT::i8  , MVT::i8, Custom);
  setTruncStoreAction(MVT::i16 , MVT::i8, Custom);
  setTruncStoreAction(MVT::i32 , MVT::i8, Custom);
  setTruncStoreAction(MVT::i64 , MVT::i8, Custom);
  setTruncStoreAction(MVT::i128, MVT::i8, Custom);

  setLoadXAction(ISD::EXTLOAD,  MVT::i16, Custom);
  setLoadXAction(ISD::SEXTLOAD, MVT::i16, Custom);
  setLoadXAction(ISD::ZEXTLOAD, MVT::i16, Custom);

  // SPU constant load actions are custom lowered:
  setOperationAction(ISD::Constant,   MVT::i64, Custom);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Custom);

  // SPU's loads and stores have to be custom lowered:
  for (unsigned sctype = (unsigned) MVT::i1; sctype < (unsigned) MVT::f128;
       ++sctype) {
    MVT VT = (MVT::SimpleValueType)sctype;

    setOperationAction(ISD::LOAD, VT, Custom);
    setOperationAction(ISD::STORE, VT, Custom);
  }

  // Custom lower BRCOND for i1, i8 to "promote" the result to
  // i32 and i16, respectively.
  setOperationAction(ISD::BRCOND, MVT::Other, Custom);

  // Expand the jumptable branches
  setOperationAction(ISD::BR_JT,        MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,        MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,    MVT::Other, Expand);

  // SPU has no intrinsics for these particular operations:
  setOperationAction(ISD::MEMBARRIER, MVT::Other, Expand);

  // PowerPC has no SREM/UREM instructions
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

  // If we're enabling GP optimizations, use hardware square root
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
  // And SPU needs custom lowering for shift left/right for i64
  setOperationAction(ISD::SHL,  MVT::i64,    Custom);
  setOperationAction(ISD::SRL,  MVT::i64,    Custom);
  setOperationAction(ISD::SRA,  MVT::i64,    Custom);

  // Custom lower i8, i32 and i64 multiplications
  setOperationAction(ISD::MUL,  MVT::i8,     Custom);
  setOperationAction(ISD::MUL,  MVT::i32,    Custom);
  setOperationAction(ISD::MUL,  MVT::i64,    Custom);

  // Need to custom handle (some) common i8, i64 math ops
  setOperationAction(ISD::ADD,  MVT::i64,    Custom);
  setOperationAction(ISD::SUB,  MVT::i8,     Custom);
  setOperationAction(ISD::SUB,  MVT::i64,    Custom);

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
  setOperationAction(ISD::SELECT, MVT::i1,   Promote);
  setOperationAction(ISD::SELECT, MVT::i8,   Legal);
  setOperationAction(ISD::SELECT, MVT::i16,  Legal);
  setOperationAction(ISD::SELECT, MVT::i32,  Legal);
  setOperationAction(ISD::SELECT, MVT::i64,  Expand);

  setOperationAction(ISD::SETCC, MVT::i1,    Promote);
  setOperationAction(ISD::SETCC, MVT::i8,    Legal);
  setOperationAction(ISD::SETCC, MVT::i16,   Legal);
  setOperationAction(ISD::SETCC, MVT::i32,   Legal);
  setOperationAction(ISD::SETCC, MVT::i64,   Expand);

  // Zero extension and sign extension for i64 have to be
  // custom legalized
  setOperationAction(ISD::ZERO_EXTEND, MVT::i64, Custom);
  setOperationAction(ISD::SIGN_EXTEND, MVT::i64, Custom);
  setOperationAction(ISD::ANY_EXTEND,  MVT::i64, Custom);

  // SPU has a legal FP -> signed INT instruction
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Legal);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Legal);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Custom);

  // FDIV on SPU requires custom lowering
  setOperationAction(ISD::FDIV, MVT::f32, Custom);
  //setOperationAction(ISD::FDIV, MVT::f64, Custom);

  // SPU has [U|S]INT_TO_FP
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Legal);
  setOperationAction(ISD::SINT_TO_FP, MVT::i16, Promote);
  setOperationAction(ISD::SINT_TO_FP, MVT::i8, Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Legal);
  setOperationAction(ISD::UINT_TO_FP, MVT::i16, Promote);
  setOperationAction(ISD::UINT_TO_FP, MVT::i8, Promote);
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
  for (unsigned sctype = (unsigned) MVT::i1; sctype < (unsigned) MVT::f128;
       ++sctype) {
    MVT VT = (MVT::SimpleValueType)sctype;

    setOperationAction(ISD::GlobalAddress, VT, Custom);
    setOperationAction(ISD::ConstantPool,  VT, Custom);
    setOperationAction(ISD::JumpTable,     VT, Custom);
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

  for (unsigned i = (unsigned)MVT::FIRST_VECTOR_VALUETYPE;
       i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
    MVT VT = (MVT::SimpleValueType)i;

    // add/sub are legal for all supported vector VT's.
    setOperationAction(ISD::ADD , VT, Legal);
    setOperationAction(ISD::SUB , VT, Legal);
    // mul has to be custom lowered.
    setOperationAction(ISD::MUL , VT, Custom);

    setOperationAction(ISD::AND   , VT, Legal);
    setOperationAction(ISD::OR    , VT, Legal);
    setOperationAction(ISD::XOR   , VT, Legal);
    setOperationAction(ISD::LOAD  , VT, Legal);
    setOperationAction(ISD::SELECT, VT, Legal);
    setOperationAction(ISD::STORE,  VT, Legal);

    // These operations need to be expanded:
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);
    setOperationAction(ISD::FDIV, VT, Custom);

    // Custom lower build_vector, constant pool spills, insert and
    // extract vector elements:
    setOperationAction(ISD::BUILD_VECTOR, VT, Custom);
    setOperationAction(ISD::ConstantPool, VT, Custom);
    setOperationAction(ISD::SCALAR_TO_VECTOR, VT, Custom);
    setOperationAction(ISD::EXTRACT_VECTOR_ELT, VT, Custom);
    setOperationAction(ISD::INSERT_VECTOR_ELT, VT, Custom);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Custom);
  }

  setOperationAction(ISD::MUL, MVT::v16i8, Custom);
  setOperationAction(ISD::AND, MVT::v16i8, Custom);
  setOperationAction(ISD::OR,  MVT::v16i8, Custom);
  setOperationAction(ISD::XOR, MVT::v16i8, Custom);
  setOperationAction(ISD::SCALAR_TO_VECTOR, MVT::v4f32, Custom);

  setShiftAmountType(MVT::i32);
  setSetCCResultContents(ZeroOrOneSetCCResult);

  setStackPointerRegisterToSaveRestore(SPU::R1);

  // We have target-specific dag combine patterns for the following nodes:
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::ZERO_EXTEND);
  setTargetDAGCombine(ISD::SIGN_EXTEND);
  setTargetDAGCombine(ISD::ANY_EXTEND);

  computeRegisterProperties();
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
    node_names[(unsigned) SPUISD::INSERT_MASK] = "SPUISD::INSERT_MASK";
    node_names[(unsigned) SPUISD::CNTB] = "SPUISD::CNTB";
    node_names[(unsigned) SPUISD::PROMOTE_SCALAR] = "SPUISD::PROMOTE_SCALAR";
    node_names[(unsigned) SPUISD::EXTRACT_ELT0] = "SPUISD::EXTRACT_ELT0";
    node_names[(unsigned) SPUISD::EXTRACT_ELT0_CHAINED]
                                              = "SPUISD::EXTRACT_ELT0_CHAINED";
    node_names[(unsigned) SPUISD::EXTRACT_I1_ZEXT] = "SPUISD::EXTRACT_I1_ZEXT";
    node_names[(unsigned) SPUISD::EXTRACT_I1_SEXT] = "SPUISD::EXTRACT_I1_SEXT";
    node_names[(unsigned) SPUISD::EXTRACT_I8_ZEXT] = "SPUISD::EXTRACT_I8_ZEXT";
    node_names[(unsigned) SPUISD::EXTRACT_I8_SEXT] = "SPUISD::EXTRACT_I8_SEXT";
    node_names[(unsigned) SPUISD::MPY] = "SPUISD::MPY";
    node_names[(unsigned) SPUISD::MPYU] = "SPUISD::MPYU";
    node_names[(unsigned) SPUISD::MPYH] = "SPUISD::MPYH";
    node_names[(unsigned) SPUISD::MPYHH] = "SPUISD::MPYHH";
    node_names[(unsigned) SPUISD::SHLQUAD_L_BITS] = "SPUISD::SHLQUAD_L_BITS";
    node_names[(unsigned) SPUISD::SHLQUAD_L_BYTES] = "SPUISD::SHLQUAD_L_BYTES";
    node_names[(unsigned) SPUISD::VEC_SHL] = "SPUISD::VEC_SHL";
    node_names[(unsigned) SPUISD::VEC_SRL] = "SPUISD::VEC_SRL";
    node_names[(unsigned) SPUISD::VEC_SRA] = "SPUISD::VEC_SRA";
    node_names[(unsigned) SPUISD::VEC_ROTL] = "SPUISD::VEC_ROTL";
    node_names[(unsigned) SPUISD::VEC_ROTR] = "SPUISD::VEC_ROTR";
    node_names[(unsigned) SPUISD::ROTQUAD_RZ_BYTES] =
      "SPUISD::ROTQUAD_RZ_BYTES";
    node_names[(unsigned) SPUISD::ROTQUAD_RZ_BITS] =
      "SPUISD::ROTQUAD_RZ_BITS";
    node_names[(unsigned) SPUISD::ROTBYTES_RIGHT_S] =
      "SPUISD::ROTBYTES_RIGHT_S";
    node_names[(unsigned) SPUISD::ROTBYTES_LEFT] = "SPUISD::ROTBYTES_LEFT";
    node_names[(unsigned) SPUISD::ROTBYTES_LEFT_CHAINED] =
      "SPUISD::ROTBYTES_LEFT_CHAINED";
    node_names[(unsigned) SPUISD::ROTBYTES_LEFT_BITS] =
      "SPUISD::ROTBYTES_LEFT_BITS";
    node_names[(unsigned) SPUISD::SELECT_MASK] = "SPUISD::SELECT_MASK";
    node_names[(unsigned) SPUISD::SELB] = "SPUISD::SELB";
    node_names[(unsigned) SPUISD::ADD_EXTENDED] = "SPUISD::ADD_EXTENDED";
    node_names[(unsigned) SPUISD::CARRY_GENERATE] = "SPUISD::CARRY_GENERATE";
    node_names[(unsigned) SPUISD::SUB_EXTENDED] = "SPUISD::SUB_EXTENDED";
    node_names[(unsigned) SPUISD::BORROW_GENERATE] = "SPUISD::BORROW_GENERATE";
    node_names[(unsigned) SPUISD::FPInterp] = "SPUISD::FPInterp";
    node_names[(unsigned) SPUISD::FPRecipEst] = "SPUISD::FPRecipEst";
    node_names[(unsigned) SPUISD::SEXT32TO64] = "SPUISD::SEXT32TO64";
  }

  std::map<unsigned, const char *>::iterator i = node_names.find(Opcode);

  return ((i != node_names.end()) ? i->second : 0);
}

MVT SPUTargetLowering::getSetCCResultType(const SDValue &Op) const {
  MVT VT = Op.getValueType();
  if (VT.isInteger())
    return VT;
  else
    return MVT::i32;
}

//===----------------------------------------------------------------------===//
// Calling convention code:
//===----------------------------------------------------------------------===//

#include "SPUGenCallingConv.inc"

//===----------------------------------------------------------------------===//
//  LowerOperation implementation
//===----------------------------------------------------------------------===//

/// Aligned load common code for CellSPU
/*!
  \param[in] Op The SelectionDAG load or store operand
  \param[in] DAG The selection DAG
  \param[in] ST CellSPU subtarget information structure
  \param[in,out] alignment Caller initializes this to the load or store node's
  value from getAlignment(), may be updated while generating the aligned load
  \param[in,out] alignOffs Aligned offset; set by AlignedLoad to the aligned
  offset (divisible by 16, modulo 16 == 0)
  \param[in,out] prefSlotOffs Preferred slot offset; set by AlignedLoad to the
  offset of the preferred slot (modulo 16 != 0)
  \param[in,out] VT Caller initializes this value type to the the load or store
  node's loaded or stored value type; may be updated if an i1-extended load or
  store.
  \param[out] was16aligned true if the base pointer had 16-byte alignment,
  otherwise false. Can help to determine if the chunk needs to be rotated.

 Both load and store lowering load a block of data aligned on a 16-byte
 boundary. This is the common aligned load code shared between both.
 */
static SDValue
AlignedLoad(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST,
            LSBaseSDNode *LSN,
            unsigned &alignment, int &alignOffs, int &prefSlotOffs,
            MVT &VT, bool &was16aligned)
{
  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  const valtype_map_s *vtm = getValueTypeMapEntry(VT);
  SDValue basePtr = LSN->getBasePtr();
  SDValue chain = LSN->getChain();

  if (basePtr.getOpcode() == ISD::ADD) {
    SDValue Op1 = basePtr.getNode()->getOperand(1);

    if (Op1.getOpcode() == ISD::Constant
        || Op1.getOpcode() == ISD::TargetConstant) {
      const ConstantSDNode *CN = cast<ConstantSDNode>(basePtr.getOperand(1));

      alignOffs = (int) CN->getZExtValue();
      prefSlotOffs = (int) (alignOffs & 0xf);

      // Adjust the rotation amount to ensure that the final result ends up in
      // the preferred slot:
      prefSlotOffs -= vtm->prefslot_byte;
      basePtr = basePtr.getOperand(0);

      // Loading from memory, can we adjust alignment?
      if (basePtr.getOpcode() == SPUISD::AFormAddr) {
        SDValue APtr = basePtr.getOperand(0);
        if (APtr.getOpcode() == ISD::TargetGlobalAddress) {
          GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(APtr);
          alignment = GSDN->getGlobal()->getAlignment();
        }
      }
    } else {
      alignOffs = 0;
      prefSlotOffs = -vtm->prefslot_byte;
    }
  } else if (basePtr.getOpcode() == ISD::FrameIndex) {
    FrameIndexSDNode *FIN = cast<FrameIndexSDNode>(basePtr);
    alignOffs = int(FIN->getIndex() * SPUFrameInfo::stackSlotSize());
    prefSlotOffs = (int) (alignOffs & 0xf);
    prefSlotOffs -= vtm->prefslot_byte;
    basePtr = DAG.getRegister(SPU::R1, VT);
  } else {
    alignOffs = 0;
    prefSlotOffs = -vtm->prefslot_byte;
  }

  if (alignment == 16) {
    // Realign the base pointer as a D-Form address:
    if (!isMemoryOperand(basePtr) || (alignOffs & ~0xf) != 0) {
      basePtr = DAG.getNode(ISD::ADD, PtrVT,
                            basePtr,
                            DAG.getConstant((alignOffs & ~0xf), PtrVT));
    }

    // Emit the vector load:
    was16aligned = true;
    return DAG.getLoad(MVT::v16i8, chain, basePtr,
                       LSN->getSrcValue(), LSN->getSrcValueOffset(),
                       LSN->isVolatile(), 16);
  }

  // Unaligned load or we're using the "large memory" model, which means that
  // we have to be very pessimistic:
  if (isMemoryOperand(basePtr) || isIndirectOperand(basePtr)) {
    basePtr = DAG.getNode(SPUISD::IndirectAddr, PtrVT, basePtr,
                          DAG.getConstant(0, PtrVT));
  }

  // Add the offset
  basePtr = DAG.getNode(ISD::ADD, PtrVT, basePtr,
                        DAG.getConstant((alignOffs & ~0xf), PtrVT));
  was16aligned = false;
  return DAG.getLoad(MVT::v16i8, chain, basePtr,
                     LSN->getSrcValue(), LSN->getSrcValueOffset(),
                     LSN->isVolatile(), 16);
}

/// Custom lower loads for CellSPU
/*!
 All CellSPU loads and stores are aligned to 16-byte boundaries, so for elements
 within a 16-byte block, we have to rotate to extract the requested element.
 */
static SDValue
LowerLOAD(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  LoadSDNode *LN = cast<LoadSDNode>(Op);
  SDValue the_chain = LN->getChain();
  MVT VT = LN->getMemoryVT();
  MVT OpVT = Op.getNode()->getValueType(0);
  ISD::LoadExtType ExtType = LN->getExtensionType();
  unsigned alignment = LN->getAlignment();
  SDValue Ops[8];

  switch (LN->getAddressingMode()) {
  case ISD::UNINDEXED: {
    int offset, rotamt;
    bool was16aligned;
    SDValue result =
      AlignedLoad(Op, DAG, ST, LN,alignment, offset, rotamt, VT, was16aligned);

    if (result.getNode() == 0)
      return result;

    the_chain = result.getValue(1);
    // Rotate the chunk if necessary
    if (rotamt < 0)
      rotamt += 16;
    if (rotamt != 0 || !was16aligned) {
      SDVTList vecvts = DAG.getVTList(MVT::v16i8, MVT::Other);

      Ops[0] = the_chain;
      Ops[1] = result;
      if (was16aligned) {
        Ops[2] = DAG.getConstant(rotamt, MVT::i16);
      } else {
        MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
        LoadSDNode *LN1 = cast<LoadSDNode>(result);
        Ops[2] = DAG.getNode(ISD::ADD, PtrVT, LN1->getBasePtr(),
                             DAG.getConstant(rotamt, PtrVT));
      }

      result = DAG.getNode(SPUISD::ROTBYTES_LEFT_CHAINED, vecvts, Ops, 3);
      the_chain = result.getValue(1);
    }

    if (VT == OpVT || ExtType == ISD::EXTLOAD) {
      SDVTList scalarvts;
      MVT vecVT = MVT::v16i8;

      // Convert the loaded v16i8 vector to the appropriate vector type
      // specified by the operand:
      if (OpVT == VT) {
        if (VT != MVT::i1)
          vecVT = MVT::getVectorVT(VT, (128 / VT.getSizeInBits()));
      } else
        vecVT = MVT::getVectorVT(OpVT, (128 / OpVT.getSizeInBits()));

      Ops[0] = the_chain;
      Ops[1] = DAG.getNode(ISD::BIT_CONVERT, vecVT, result);
      scalarvts = DAG.getVTList((OpVT == VT ? VT : OpVT), MVT::Other);
      result = DAG.getNode(SPUISD::EXTRACT_ELT0_CHAINED, scalarvts, Ops, 2);
      the_chain = result.getValue(1);
    } else {
      // Handle the sign and zero-extending loads for i1 and i8:
      unsigned NewOpC;

      if (ExtType == ISD::SEXTLOAD) {
        NewOpC = (OpVT == MVT::i1
                  ? SPUISD::EXTRACT_I1_SEXT
                  : SPUISD::EXTRACT_I8_SEXT);
      } else {
        assert(ExtType == ISD::ZEXTLOAD);
        NewOpC = (OpVT == MVT::i1
                  ? SPUISD::EXTRACT_I1_ZEXT
                  : SPUISD::EXTRACT_I8_ZEXT);
      }

      result = DAG.getNode(NewOpC, OpVT, result);
    }

    SDVTList retvts = DAG.getVTList(OpVT, MVT::Other);
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
    int chunk_offset, slot_offset;
    bool was16aligned;

    // The vector type we really want to load from the 16-byte chunk, except
    // in the case of MVT::i1, which has to be v16i8.
    MVT vecVT, stVecVT = MVT::v16i8;

    if (StVT != MVT::i1)
      stVecVT = MVT::getVectorVT(StVT, (128 / StVT.getSizeInBits()));
    vecVT = MVT::getVectorVT(VT, (128 / VT.getSizeInBits()));

    SDValue alignLoadVec =
      AlignedLoad(Op, DAG, ST, SN, alignment,
                  chunk_offset, slot_offset, VT, was16aligned);

    if (alignLoadVec.getNode() == 0)
      return alignLoadVec;

    LoadSDNode *LN = cast<LoadSDNode>(alignLoadVec);
    SDValue basePtr = LN->getBasePtr();
    SDValue the_chain = alignLoadVec.getValue(1);
    SDValue theValue = SN->getValue();
    SDValue result;

    if (StVT != VT
        && (theValue.getOpcode() == ISD::AssertZext
            || theValue.getOpcode() == ISD::AssertSext)) {
      // Drill down and get the value for zero- and sign-extended
      // quantities
      theValue = theValue.getOperand(0);
    }

    chunk_offset &= 0xf;

    SDValue insertEltOffs = DAG.getConstant(chunk_offset, PtrVT);
    SDValue insertEltPtr;
    SDValue insertEltOp;

    // If the base pointer is already a D-form address, then just create
    // a new D-form address with a slot offset and the orignal base pointer.
    // Otherwise generate a D-form address with the slot offset relative
    // to the stack pointer, which is always aligned.
    DEBUG(cerr << "CellSPU LowerSTORE: basePtr = ");
    DEBUG(basePtr.getNode()->dump(&DAG));
    DEBUG(cerr << "\n");

    if (basePtr.getOpcode() == SPUISD::IndirectAddr ||
        (basePtr.getOpcode() == ISD::ADD
         && basePtr.getOperand(0).getOpcode() == SPUISD::IndirectAddr)) {
      insertEltPtr = basePtr;
    } else {
      insertEltPtr = DAG.getNode(ISD::ADD, PtrVT, basePtr, insertEltOffs);
    }

    insertEltOp = DAG.getNode(SPUISD::INSERT_MASK, stVecVT, insertEltPtr);
    result = DAG.getNode(SPUISD::SHUFB, vecVT,
                         DAG.getNode(ISD::SCALAR_TO_VECTOR, vecVT, theValue),
                         alignLoadVec,
                         DAG.getNode(ISD::BIT_CONVERT, vecVT, insertEltOp));

    result = DAG.getStore(the_chain, result, basePtr,
                          LN->getSrcValue(), LN->getSrcValueOffset(),
                          LN->isVolatile(), LN->getAlignment());

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

/// Generate the address of a constant pool entry.
static SDValue
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

//! Custom lower i64 integer constants
/*!
 This code inserts all of the necessary juggling that needs to occur to load
 a 64-bit constant into a register.
 */
static SDValue
LowerConstant(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  ConstantSDNode *CN = cast<ConstantSDNode>(Op.getNode());

  if (VT == MVT::i64) {
    SDValue T = DAG.getConstant(CN->getZExtValue(), MVT::i64);
    return DAG.getNode(SPUISD::EXTRACT_ELT0, VT,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v2i64, T, T));
  } else {
    cerr << "LowerConstant: unhandled constant type "
         << VT.getMVTString()
         << "\n";
    abort();
    /*NOTREACHED*/
  }

  return SDValue();
}

//! Custom lower double precision floating point constants
static SDValue
LowerConstantFP(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  ConstantFPSDNode *FP = cast<ConstantFPSDNode>(Op.getNode());

  assert((FP != 0) &&
         "LowerConstantFP: Node is not ConstantFPSDNode");

  if (VT == MVT::f64) {
    uint64_t dbits = DoubleToBits(FP->getValueAPF().convertToDouble());
    return DAG.getNode(ISD::BIT_CONVERT, VT,
                       LowerConstant(DAG.getConstant(dbits, MVT::i64), DAG));
  }

  return SDValue();
}

//! Lower MVT::i1, MVT::i8 brcond to a promoted type (MVT::i32, MVT::i16)
static SDValue
LowerBRCOND(SDValue Op, SelectionDAG &DAG)
{
  SDValue Cond = Op.getOperand(1);
  MVT CondVT = Cond.getValueType();
  MVT CondNVT;

  if (CondVT == MVT::i1 || CondVT == MVT::i8) {
    CondNVT = (CondVT == MVT::i1 ? MVT::i32 : MVT::i16);
    return DAG.getNode(ISD::BRCOND, Op.getValueType(),
                      Op.getOperand(0),
                      DAG.getNode(ISD::ZERO_EXTEND, CondNVT, Op.getOperand(1)),
                      Op.getOperand(2));
  } else
    return SDValue();                // Unchanged
}

static SDValue
LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG, int &VarArgsFrameIndex)
{
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SmallVector<SDValue, 8> ArgValues;
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
    SDValue ArgVal;
    bool needsLoad = false;
    MVT ObjectVT = Op.getValue(ArgNo).getValueType();
    unsigned ObjSize = ObjectVT.getSizeInBits()/8;

    switch (ObjectVT.getSimpleVT()) {
    default: {
      cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
           << ObjectVT.getMVTString()
           << "\n";
      abort();
    }
    case MVT::i8:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R8CRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::i8);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::i16:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R16CRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::i16);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::i32:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32CRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::i32);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::i64:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R64CRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::i64);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::f32:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R32FPRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::f32);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::f64:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::R64FPRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, MVT::f64);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    case MVT::v2f64:
    case MVT::v4f32:
    case MVT::v2i64:
    case MVT::v4i32:
    case MVT::v8i16:
    case MVT::v16i8:
      if (!isVarArg && ArgRegIdx < NumArgRegs) {
        unsigned VReg = RegInfo.createVirtualRegister(&SPU::VECREGRegClass);
        RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
        ArgVal = DAG.getCopyFromReg(Root, VReg, ObjectVT);
        ++ArgRegIdx;
      } else {
        needsLoad = true;
      }
      break;
    }

    // We need to load the argument to a virtual register if we determined above
    // that we ran out of physical registers of the appropriate type
    if (needsLoad) {
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDValue FIN = DAG.getFrameIndex(FI, PtrVT);
      ArgVal = DAG.getLoad(ObjectVT, Root, FIN, NULL, 0);
      ArgOffset += StackSlotSize;
    }

    ArgValues.push_back(ArgVal);
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (isVarArg) {
    VarArgsFrameIndex = MFI->CreateFixedObject(PtrVT.getSizeInBits()/8,
                                               ArgOffset);
    SDValue FIN = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
    // If this function is vararg, store any remaining integer argument regs to
    // their spots on the stack so that they may be loaded by deferencing the
    // result of va_next.
    SmallVector<SDValue, 8> MemOps;
    for (; ArgRegIdx != NumArgRegs; ++ArgRegIdx) {
      unsigned VReg = RegInfo.createVirtualRegister(&SPU::GPRCRegClass);
      RegInfo.addLiveIn(ArgRegs[ArgRegIdx], VReg);
      SDValue Val = DAG.getCopyFromReg(Root, VReg, PtrVT);
      SDValue Store = DAG.getStore(Val.getValue(1), Val, FIN, NULL, 0);
      MemOps.push_back(Store);
      // Increment the address by four for the next argument to store
      SDValue PtrOff = DAG.getConstant(PtrVT.getSizeInBits()/8, PtrVT);
      FIN = DAG.getNode(ISD::ADD, PtrOff.getValueType(), FIN, PtrOff);
    }
    if (!MemOps.empty())
      Root = DAG.getNode(ISD::TokenFactor, MVT::Other,&MemOps[0],MemOps.size());
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getMergeValues(Op.getNode()->getVTList(), &ArgValues[0],
                            ArgValues.size());
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

static
SDValue
LowerCALL(SDValue Op, SelectionDAG &DAG, const SPUSubtarget *ST) {
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  SDValue Chain = TheCall->getChain();
#if 0
  bool isVarArg   = TheCall->isVarArg();
  bool isTailCall = TheCall->isTailCall();
#endif
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
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getExternalSymbol(S->getSymbol(), Callee.getValueType());
  else if (SDNode *Dest = isLSAAddress(Callee, DAG)) {
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
  case MVT::f32:
  case MVT::f64:
    Chain = DAG.getCopyFromReg(Chain, SPU::R3, TheCall->getValueType(0),
                               InFlag).getValue(1);
    ResultVals[0] = Chain.getValue(0);
    NumResults = 1;
    break;
  case MVT::v2f64:
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
      return DAG.getConstant(Value, ValueType);
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
      return DAG.getConstant(Value, ValueType);
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
      return DAG.getConstant(Value, ValueType);
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
      return DAG.getConstant(Value & 0xff, ValueType);
    else if (ValueType == MVT::i8
             && (Value & 0xff) == Value)
      return DAG.getConstant(Value, ValueType);
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
      return DAG.getConstant(Value >> 16, ValueType);
  }

  return SDValue();
}

/// get_v4i32_imm - Catch-all for general 32-bit constant vectors
SDValue SPU::get_v4i32_imm(SDNode *N, SelectionDAG &DAG) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    return DAG.getConstant((unsigned) CN->getZExtValue(), MVT::i32);
  }

  return SDValue();
}

/// get_v4i32_imm - Catch-all for general 64-bit constant vectors
SDValue SPU::get_v2i64_imm(SDNode *N, SelectionDAG &DAG) {
  if (ConstantSDNode *CN = getVecImm(N)) {
    return DAG.getConstant((unsigned) CN->getZExtValue(), MVT::i64);
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

// If this is a case we can't handle, return null and let the default
// expansion code take care of it.  If we CAN select this case, and if it
// selects to a single instruction, return Op.  Otherwise, if we can codegen
// this case more efficiently than a constant pool load, lower it to the
// sequence of ops that should be used.
static SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) {
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
  case MVT::v4f32: {
    uint32_t Value32 = SplatBits;
    assert(SplatSize == 4
           && "LowerBUILD_VECTOR: Unexpected floating point vector element.");
    // NOTE: pretend the constant is an integer. LLVM won't load FP constants
    SDValue T = DAG.getConstant(Value32, MVT::i32);
    return DAG.getNode(ISD::BIT_CONVERT, MVT::v4f32,
                       DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32, T, T, T, T));
    break;
  }
  case MVT::v2f64: {
    uint64_t f64val = SplatBits;
    assert(SplatSize == 8
           && "LowerBUILD_VECTOR: 64-bit float vector element: unexpected size.");
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
  case MVT::v2i64: {
    uint64_t val = SplatBits;
    uint32_t upper = uint32_t(val >> 32);
    uint32_t lower = uint32_t(val);

    if (upper == lower) {
      // Magic constant that can be matched by IL, ILA, et. al.
      SDValue Val = DAG.getTargetConstant(val, MVT::i64);
      return DAG.getNode(ISD::BUILD_VECTOR, VT, Val, Val);
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
        LO32 = DAG.getNode(ISD::BIT_CONVERT, VT,
                           DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                       LO32C, LO32C, LO32C, LO32C));
      }

      // Create upper vector if not a special pattern
      if (!upper_special) {
        SDValue HI32C = DAG.getConstant(upper, MVT::i32);
        HI32 = DAG.getNode(ISD::BIT_CONVERT, VT,
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

      return DAG.getNode(SPUISD::SHUFB, VT, HI32, LO32,
                         DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                     &ShufBytes[0], ShufBytes.size()));
    }
  }
  }

  return SDValue();
}

/// LowerVECTOR_SHUFFLE - Lower a vector shuffle (V1, V2, V3) to something on
/// which the Cell can operate. The code inspects V3 to ascertain whether the
/// permutation vector, V3, is monotonically increasing with one "exception"
/// element, e.g., (0, 1, _, 3). If this is the case, then generate a
/// INSERT_MASK synthetic instruction. Otherwise, spill V3 to the constant pool.
/// In either case, the net result is going to eventually invoke SHUFB to
/// permute/shuffle the bytes from V1 and V2.
/// \note
/// INSERT_MASK is eventually selected as one of the C*D instructions, generate
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
  MVT EltVT = V1.getValueType().getVectorElementType();
  unsigned EltsFromV2 = 0;
  unsigned V2Elt = 0;
  unsigned V2EltIdx0 = 0;
  unsigned CurrElt = 0;
  bool monotonic = true;
  if (EltVT == MVT::i8)
    V2EltIdx0 = 16;
  else if (EltVT == MVT::i16)
    V2EltIdx0 = 8;
  else if (EltVT == MVT::i32)
    V2EltIdx0 = 4;
  else
    assert(0 && "Unhandled vector type in LowerVECTOR_SHUFFLE");

  for (unsigned i = 0, e = PermMask.getNumOperands();
       EltsFromV2 <= 1 && monotonic && i != e;
       ++i) {
    unsigned SrcElt;
    if (PermMask.getOperand(i).getOpcode() == ISD::UNDEF)
      SrcElt = 0;
    else
      SrcElt = cast<ConstantSDNode>(PermMask.getOperand(i))->getZExtValue();

    if (SrcElt >= V2EltIdx0) {
      ++EltsFromV2;
      V2Elt = (V2EltIdx0 - SrcElt) << 2;
    } else if (CurrElt != SrcElt) {
      monotonic = false;
    }

    ++CurrElt;
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
    // Copy register's contents as index in INSERT_MASK:
    SDValue ShufMaskOp =
      DAG.getNode(SPUISD::INSERT_MASK, V1.getValueType(),
                  DAG.getTargetConstant(V2Elt, MVT::i32),
                  DAG.getCopyFromReg(InitTempReg, VReg, PtrVT));
    // Use shuffle mask in SHUFB synthetic instruction:
    return DAG.getNode(SPUISD::SHUFB, V1.getValueType(), V2, V1, ShufMaskOp);
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
      return DAG.getNode(SPUISD::PROMOTE_SCALAR, Op.getValueType(), Op0, Op0);
    }
  }

  return SDValue();
}

static SDValue LowerVectorMUL(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getValueType().getSimpleVT()) {
  default:
    cerr << "CellSPU: Unknown vector multiplication, got "
         << Op.getValueType().getMVTString()
         << "\n";
    abort();
    /*NOTREACHED*/

  case MVT::v4i32: {
    SDValue rA = Op.getOperand(0);
    SDValue rB = Op.getOperand(1);
    SDValue HiProd1 = DAG.getNode(SPUISD::MPYH, MVT::v4i32, rA, rB);
    SDValue HiProd2 = DAG.getNode(SPUISD::MPYH, MVT::v4i32, rB, rA);
    SDValue LoProd = DAG.getNode(SPUISD::MPYU, MVT::v4i32, rA, rB);
    SDValue Residual1 = DAG.getNode(ISD::ADD, MVT::v4i32, LoProd, HiProd1);

    return DAG.getNode(ISD::ADD, MVT::v4i32, Residual1, HiProd2);
    break;
  }

  // Multiply two v8i16 vectors (pipeline friendly version):
  // a) multiply lower halves, mask off upper 16-bit of 32-bit product
  // b) multiply upper halves, rotate left by 16 bits (inserts 16 lower zeroes)
  // c) Use SELB to select upper and lower halves from the intermediate results
  //
  // NOTE: We really want to move the SELECT_MASK to earlier to actually get the
  // dual-issue. This code does manage to do this, even if it's a little on
  // the wacky side
  case MVT::v8i16: {
    MachineFunction &MF = DAG.getMachineFunction();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    SDValue Chain = Op.getOperand(0);
    SDValue rA = Op.getOperand(0);
    SDValue rB = Op.getOperand(1);
    unsigned FSMBIreg = RegInfo.createVirtualRegister(&SPU::VECREGRegClass);
    unsigned HiProdReg = RegInfo.createVirtualRegister(&SPU::VECREGRegClass);

    SDValue FSMBOp =
      DAG.getCopyToReg(Chain, FSMBIreg,
                       DAG.getNode(SPUISD::SELECT_MASK, MVT::v8i16,
                                   DAG.getConstant(0xcccc, MVT::i16)));

    SDValue HHProd =
      DAG.getCopyToReg(FSMBOp, HiProdReg,
                       DAG.getNode(SPUISD::MPYHH, MVT::v8i16, rA, rB));

    SDValue HHProd_v4i32 =
      DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32,
                  DAG.getCopyFromReg(HHProd, HiProdReg, MVT::v4i32));

    return DAG.getNode(SPUISD::SELB, MVT::v8i16,
                       DAG.getNode(SPUISD::MPY, MVT::v8i16, rA, rB),
                       DAG.getNode(ISD::BIT_CONVERT, Op.getValueType(),
                                   DAG.getNode(SPUISD::VEC_SHL, MVT::v4i32,
                                               HHProd_v4i32,
                                               DAG.getConstant(16, MVT::i16))),
                       DAG.getCopyFromReg(FSMBOp, FSMBIreg, MVT::v4i32));
  }

  // This M00sE is N@stI! (apologies to Monty Python)
  //
  // SPU doesn't know how to do any 8-bit multiplication, so the solution
  // is to break it all apart, sign extend, and reassemble the various
  // intermediate products.
  case MVT::v16i8: {
    SDValue rA = Op.getOperand(0);
    SDValue rB = Op.getOperand(1);
    SDValue c8 = DAG.getConstant(8, MVT::i32);
    SDValue c16 = DAG.getConstant(16, MVT::i32);

    SDValue LLProd =
      DAG.getNode(SPUISD::MPY, MVT::v8i16,
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, rA),
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, rB));

    SDValue rALH = DAG.getNode(SPUISD::VEC_SRA, MVT::v8i16, rA, c8);

    SDValue rBLH = DAG.getNode(SPUISD::VEC_SRA, MVT::v8i16, rB, c8);

    SDValue LHProd =
      DAG.getNode(SPUISD::VEC_SHL, MVT::v8i16,
                  DAG.getNode(SPUISD::MPY, MVT::v8i16, rALH, rBLH), c8);

    SDValue FSMBmask = DAG.getNode(SPUISD::SELECT_MASK, MVT::v8i16,
                                     DAG.getConstant(0x2222, MVT::i16));

    SDValue LoProdParts =
      DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32,
                  DAG.getNode(SPUISD::SELB, MVT::v8i16,
                              LLProd, LHProd, FSMBmask));

    SDValue LoProdMask = DAG.getConstant(0xffff, MVT::i32);

    SDValue LoProd =
      DAG.getNode(ISD::AND, MVT::v4i32,
                  LoProdParts,
                  DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                              LoProdMask, LoProdMask,
                              LoProdMask, LoProdMask));

    SDValue rAH =
      DAG.getNode(SPUISD::VEC_SRA, MVT::v4i32,
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, rA), c16);

    SDValue rBH =
      DAG.getNode(SPUISD::VEC_SRA, MVT::v4i32,
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32, rB), c16);

    SDValue HLProd =
      DAG.getNode(SPUISD::MPY, MVT::v8i16,
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, rAH),
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16, rBH));

    SDValue HHProd_1 =
      DAG.getNode(SPUISD::MPY, MVT::v8i16,
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16,
                              DAG.getNode(SPUISD::VEC_SRA,
                                          MVT::v4i32, rAH, c8)),
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v8i16,
                              DAG.getNode(SPUISD::VEC_SRA,
                                          MVT::v4i32, rBH, c8)));

    SDValue HHProd =
      DAG.getNode(SPUISD::SELB, MVT::v8i16,
                  HLProd,
                  DAG.getNode(SPUISD::VEC_SHL, MVT::v8i16, HHProd_1, c8),
                  FSMBmask);

    SDValue HiProd =
      DAG.getNode(SPUISD::VEC_SHL, MVT::v4i32, HHProd, c16);

    return DAG.getNode(ISD::BIT_CONVERT, MVT::v16i8,
                       DAG.getNode(ISD::OR, MVT::v4i32,
                                   LoProd, HiProd));
  }
  }

  return SDValue();
}

static SDValue LowerFDIVf32(SDValue Op, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  SDValue A = Op.getOperand(0);
  SDValue B = Op.getOperand(1);
  MVT VT = Op.getValueType();

  unsigned VRegBR, VRegC;

  if (VT == MVT::f32) {
    VRegBR = RegInfo.createVirtualRegister(&SPU::R32FPRegClass);
    VRegC = RegInfo.createVirtualRegister(&SPU::R32FPRegClass);
  } else {
    VRegBR = RegInfo.createVirtualRegister(&SPU::VECREGRegClass);
    VRegC = RegInfo.createVirtualRegister(&SPU::VECREGRegClass);
  }
  // TODO: make sure we're feeding FPInterp the right arguments
  // Right now: fi B, frest(B)

  // Computes BRcpl =
  // (Floating Interpolate (FP Reciprocal Estimate B))
  SDValue BRcpl =
      DAG.getCopyToReg(DAG.getEntryNode(), VRegBR,
                       DAG.getNode(SPUISD::FPInterp, VT, B,
                                DAG.getNode(SPUISD::FPRecipEst, VT, B)));

  // Computes A * BRcpl and stores in a temporary register
  SDValue AxBRcpl =
      DAG.getCopyToReg(BRcpl, VRegC,
                 DAG.getNode(ISD::FMUL, VT, A,
                        DAG.getCopyFromReg(BRcpl, VRegBR, VT)));
  // What's the Chain variable do? It's magic!
  // TODO: set Chain = Op(0).getEntryNode()

  return DAG.getNode(ISD::FADD, VT,
                DAG.getCopyFromReg(AxBRcpl, VRegC, VT),
                DAG.getNode(ISD::FMUL, VT,
                        DAG.getCopyFromReg(AxBRcpl, VRegBR, VT),
                        DAG.getNode(ISD::FSUB, VT, A,
                            DAG.getNode(ISD::FMUL, VT, B,
                            DAG.getCopyFromReg(AxBRcpl, VRegC, VT)))));
}

static SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  MVT VT = Op.getValueType();
  SDValue N = Op.getOperand(0);
  SDValue Elt = Op.getOperand(1);
  SDValue ShufMask[16];
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(Elt);

  assert(C != 0 && "LowerEXTRACT_VECTOR_ELT expecting constant SDNode");

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
    return DAG.getNode(SPUISD::EXTRACT_ELT0, VT, N);
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
  case MVT::i32: {
    prefslot_begin = 0; prefslot_end = 3;
    break;
  }
  case MVT::i64: {
    prefslot_begin = 0; prefslot_end = 7;
    break;
  }
  }

  assert(prefslot_begin != -1 && prefslot_end != -1 &&
         "LowerEXTRACT_VECTOR_ELT: preferred slots uninitialized");

  for (int i = 0; i < 16; ++i) {
    // zero fill uppper part of preferred slot, don't care about the
    // other slots:
    unsigned int mask_val;

    if (i <= prefslot_end) {
      mask_val =
        ((i < prefslot_begin)
         ? 0x80
         : elt_byte + (i - prefslot_begin));

      ShufMask[i] = DAG.getConstant(mask_val, MVT::i8);
    } else
      ShufMask[i] = ShufMask[i % (prefslot_end + 1)];
  }

  SDValue ShufMaskVec =
    DAG.getNode(ISD::BUILD_VECTOR, MVT::v16i8,
                &ShufMask[0],
                sizeof(ShufMask) / sizeof(ShufMask[0]));

  return DAG.getNode(SPUISD::EXTRACT_ELT0, VT,
                     DAG.getNode(SPUISD::SHUFB, N.getValueType(),
                                 N, N, ShufMaskVec));

}

static SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) {
  SDValue VecOp = Op.getOperand(0);
  SDValue ValOp = Op.getOperand(1);
  SDValue IdxOp = Op.getOperand(2);
  MVT VT = Op.getValueType();

  ConstantSDNode *CN = cast<ConstantSDNode>(IdxOp);
  assert(CN != 0 && "LowerINSERT_VECTOR_ELT: Index is not constant!");

  MVT PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  // Use $2 because it's always 16-byte aligned and it's available:
  SDValue PtrBase = DAG.getRegister(SPU::R2, PtrVT);

  SDValue result =
    DAG.getNode(SPUISD::SHUFB, VT,
                DAG.getNode(ISD::SCALAR_TO_VECTOR, VT, ValOp),
                VecOp,
                DAG.getNode(SPUISD::INSERT_MASK, VT,
                            DAG.getNode(ISD::ADD, PtrVT,
                                        PtrBase,
                                        DAG.getConstant(CN->getZExtValue(),
                                                        PtrVT))));

  return result;
}

static SDValue LowerI8Math(SDValue Op, SelectionDAG &DAG, unsigned Opc)
{
  SDValue N0 = Op.getOperand(0);      // Everything has at least one operand

  assert(Op.getValueType() == MVT::i8);
  switch (Opc) {
  default:
    assert(0 && "Unhandled i8 math operator");
    /*NOTREACHED*/
    break;
  case ISD::SUB: {
    // 8-bit subtraction: Promote the arguments up to 16-bits and truncate
    // the result:
    SDValue N1 = Op.getOperand(1);
    N0 = (N0.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N0)
          : DAG.getConstant(cast<ConstantSDNode>(N0)->getZExtValue(),
                            MVT::i16));
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            MVT::i16));
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
    N1Opc = N1.getValueType().bitsLT(MVT::i16)
            ? ISD::ZERO_EXTEND
            : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, MVT::i16, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            MVT::i16));
    SDValue ExpandArg =
      DAG.getNode(ISD::OR, MVT::i16, N0,
                  DAG.getNode(ISD::SHL, MVT::i16,
                              N0, DAG.getConstant(8, MVT::i16)));
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
                            MVT::i16));
    N1Opc = N1.getValueType().bitsLT(MVT::i16)
            ? ISD::ZERO_EXTEND
            : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, MVT::i16, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            MVT::i16));
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));
  }
  case ISD::SRA: {
    SDValue N1 = Op.getOperand(1);
    unsigned N1Opc;
    N0 = (N0.getOpcode() != ISD::Constant
          ? DAG.getNode(ISD::SIGN_EXTEND, MVT::i16, N0)
          : DAG.getConstant(cast<ConstantSDNode>(N0)->getZExtValue(),
                            MVT::i16));
    N1Opc = N1.getValueType().bitsLT(MVT::i16)
            ? ISD::SIGN_EXTEND
            : ISD::TRUNCATE;
    N1 = (N1.getOpcode() != ISD::Constant
          ? DAG.getNode(N1Opc, MVT::i16, N1)
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            MVT::i16));
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
          : DAG.getConstant(cast<ConstantSDNode>(N1)->getZExtValue(),
                            MVT::i16));
    return DAG.getNode(ISD::TRUNCATE, MVT::i8,
                       DAG.getNode(Opc, MVT::i16, N0, N1));
    break;
  }
  }

  return SDValue();
}

static SDValue LowerI64Math(SDValue Op, SelectionDAG &DAG, unsigned Opc)
{
  MVT VT = Op.getValueType();
  MVT VecVT = MVT::getVectorVT(VT, (128 / VT.getSizeInBits()));

  SDValue Op0 = Op.getOperand(0);

  switch (Opc) {
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND: {
    MVT Op0VT = Op0.getValueType();
    MVT Op0VecVT = MVT::getVectorVT(Op0VT, (128 / Op0VT.getSizeInBits()));

    assert(Op0VT == MVT::i32
           && "CellSPU: Zero/sign extending something other than i32");
    DEBUG(cerr << "CellSPU: LowerI64Math custom lowering zero/sign/any extend\n");

    unsigned NewOpc = (Opc == ISD::SIGN_EXTEND
                      ? SPUISD::ROTBYTES_RIGHT_S
                      : SPUISD::ROTQUAD_RZ_BYTES);
    SDValue PromoteScalar =
      DAG.getNode(SPUISD::PROMOTE_SCALAR, Op0VecVT, Op0);

    return DAG.getNode(SPUISD::EXTRACT_ELT0, VT,
                       DAG.getNode(ISD::BIT_CONVERT, VecVT,
                                   DAG.getNode(NewOpc, Op0VecVT,
                                               PromoteScalar,
                                               DAG.getConstant(4, MVT::i32))));
  }

  case ISD::ADD: {
    // Turn operands into vectors to satisfy type checking (shufb works on
    // vectors)
    SDValue Op0 =
      DAG.getNode(SPUISD::PROMOTE_SCALAR, MVT::v2i64, Op.getOperand(0));
    SDValue Op1 =
      DAG.getNode(SPUISD::PROMOTE_SCALAR, MVT::v2i64, Op.getOperand(1));
    SmallVector<SDValue, 16> ShufBytes;

    // Create the shuffle mask for "rotating" the borrow up one register slot
    // once the borrow is generated.
    ShufBytes.push_back(DAG.getConstant(0x04050607, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x80808080, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x0c0d0e0f, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x80808080, MVT::i32));

    SDValue CarryGen =
      DAG.getNode(SPUISD::CARRY_GENERATE, MVT::v2i64, Op0, Op1);
    SDValue ShiftedCarry =
      DAG.getNode(SPUISD::SHUFB, MVT::v2i64,
                  CarryGen, CarryGen,
                  DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                              &ShufBytes[0], ShufBytes.size()));

    return DAG.getNode(SPUISD::EXTRACT_ELT0, MVT::i64,
                       DAG.getNode(SPUISD::ADD_EXTENDED, MVT::v2i64,
                                   Op0, Op1, ShiftedCarry));
  }

  case ISD::SUB: {
    // Turn operands into vectors to satisfy type checking (shufb works on
    // vectors)
    SDValue Op0 =
      DAG.getNode(SPUISD::PROMOTE_SCALAR, MVT::v2i64, Op.getOperand(0));
    SDValue Op1 =
      DAG.getNode(SPUISD::PROMOTE_SCALAR, MVT::v2i64, Op.getOperand(1));
    SmallVector<SDValue, 16> ShufBytes;

    // Create the shuffle mask for "rotating" the borrow up one register slot
    // once the borrow is generated.
    ShufBytes.push_back(DAG.getConstant(0x04050607, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0xc0c0c0c0, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x0c0d0e0f, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0xc0c0c0c0, MVT::i32));

    SDValue BorrowGen =
      DAG.getNode(SPUISD::BORROW_GENERATE, MVT::v2i64, Op0, Op1);
    SDValue ShiftedBorrow =
      DAG.getNode(SPUISD::SHUFB, MVT::v2i64,
                  BorrowGen, BorrowGen,
                  DAG.getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                              &ShufBytes[0], ShufBytes.size()));

    return DAG.getNode(SPUISD::EXTRACT_ELT0, MVT::i64,
                       DAG.getNode(SPUISD::SUB_EXTENDED, MVT::v2i64,
                                   Op0, Op1, ShiftedBorrow));
  }

  case ISD::SHL: {
    SDValue ShiftAmt = Op.getOperand(1);
    MVT ShiftAmtVT = ShiftAmt.getValueType();
    SDValue Op0Vec = DAG.getNode(SPUISD::PROMOTE_SCALAR, VecVT, Op0);
    SDValue MaskLower =
      DAG.getNode(SPUISD::SELB, VecVT,
                  Op0Vec,
                  DAG.getConstant(0, VecVT),
                  DAG.getNode(SPUISD::SELECT_MASK, VecVT,
                              DAG.getConstant(0xff00ULL, MVT::i16)));
    SDValue ShiftAmtBytes =
      DAG.getNode(ISD::SRL, ShiftAmtVT,
                  ShiftAmt,
                  DAG.getConstant(3, ShiftAmtVT));
    SDValue ShiftAmtBits =
      DAG.getNode(ISD::AND, ShiftAmtVT,
                  ShiftAmt,
                  DAG.getConstant(7, ShiftAmtVT));

    return DAG.getNode(SPUISD::EXTRACT_ELT0, VT,
                       DAG.getNode(SPUISD::SHLQUAD_L_BITS, VecVT,
                                   DAG.getNode(SPUISD::SHLQUAD_L_BYTES, VecVT,
                                               MaskLower, ShiftAmtBytes),
                                   ShiftAmtBits));
  }

  case ISD::SRL: {
    MVT VT = Op.getValueType();
    SDValue ShiftAmt = Op.getOperand(1);
    MVT ShiftAmtVT = ShiftAmt.getValueType();
    SDValue ShiftAmtBytes =
      DAG.getNode(ISD::SRL, ShiftAmtVT,
                  ShiftAmt,
                  DAG.getConstant(3, ShiftAmtVT));
    SDValue ShiftAmtBits =
      DAG.getNode(ISD::AND, ShiftAmtVT,
                  ShiftAmt,
                  DAG.getConstant(7, ShiftAmtVT));

    return DAG.getNode(SPUISD::ROTQUAD_RZ_BITS, VT,
                       DAG.getNode(SPUISD::ROTQUAD_RZ_BYTES, VT,
                                   Op0, ShiftAmtBytes),
                       ShiftAmtBits);
  }

  case ISD::SRA: {
    // Promote Op0 to vector
    SDValue Op0 =
      DAG.getNode(SPUISD::PROMOTE_SCALAR, MVT::v2i64, Op.getOperand(0));
    SDValue ShiftAmt = Op.getOperand(1);
    MVT ShiftVT = ShiftAmt.getValueType();

    // Negate variable shift amounts
    if (!isa<ConstantSDNode>(ShiftAmt)) {
      ShiftAmt = DAG.getNode(ISD::SUB, ShiftVT,
                             DAG.getConstant(0, ShiftVT), ShiftAmt);
    }

    SDValue UpperHalfSign =
      DAG.getNode(SPUISD::EXTRACT_ELT0, MVT::i32,
                  DAG.getNode(ISD::BIT_CONVERT, MVT::v4i32,
                              DAG.getNode(SPUISD::VEC_SRA, MVT::v2i64,
                                          Op0, DAG.getConstant(31, MVT::i32))));
    SDValue UpperHalfSignMask =
      DAG.getNode(SPUISD::SELECT_MASK, MVT::v2i64, UpperHalfSign);
    SDValue UpperLowerMask =
      DAG.getNode(SPUISD::SELECT_MASK, MVT::v2i64,
                  DAG.getConstant(0xff00, MVT::i16));
    SDValue UpperLowerSelect =
      DAG.getNode(SPUISD::SELB, MVT::v2i64,
                  UpperHalfSignMask, Op0, UpperLowerMask);
    SDValue RotateLeftBytes =
      DAG.getNode(SPUISD::ROTBYTES_LEFT_BITS, MVT::v2i64,
                  UpperLowerSelect, ShiftAmt);
    SDValue RotateLeftBits =
      DAG.getNode(SPUISD::ROTBYTES_LEFT, MVT::v2i64,
                  RotateLeftBytes, ShiftAmt);

    return DAG.getNode(SPUISD::EXTRACT_ELT0, MVT::i64,
                       RotateLeftBits);
  }
  }

  return SDValue();
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

//! Lower i32 multiplication
static SDValue LowerMUL(SDValue Op, SelectionDAG &DAG, MVT VT,
                          unsigned Opc) {
  switch (VT.getSimpleVT()) {
  default:
    cerr << "CellSPU: Unknown LowerMUL value type, got "
         << Op.getValueType().getMVTString()
         << "\n";
    abort();
    /*NOTREACHED*/

  case MVT::i32: {
    SDValue rA = Op.getOperand(0);
    SDValue rB = Op.getOperand(1);

    return DAG.getNode(ISD::ADD, MVT::i32,
                       DAG.getNode(ISD::ADD, MVT::i32,
                                   DAG.getNode(SPUISD::MPYH, MVT::i32, rA, rB),
                                   DAG.getNode(SPUISD::MPYH, MVT::i32, rB, rA)),
                       DAG.getNode(SPUISD::MPYU, MVT::i32, rA, rB));
  }
  }

  return SDValue();
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

    SDValue Promote = DAG.getNode(SPUISD::PROMOTE_SCALAR, vecVT, N, N);
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
    SDValue Shift1 = DAG.getConstant(8, MVT::i16);

    SDValue Promote = DAG.getNode(SPUISD::PROMOTE_SCALAR, vecVT, N, N);
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

    SDValue Promote = DAG.getNode(SPUISD::PROMOTE_SCALAR, vecVT, N, N);
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

/// LowerOperation - Provide custom lowering hooks for some operations.
///
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
  case ISD::Constant:
    return LowerConstant(Op, DAG);
  case ISD::ConstantFP:
    return LowerConstantFP(Op, DAG);
  case ISD::BRCOND:
    return LowerBRCOND(Op, DAG);
  case ISD::FORMAL_ARGUMENTS:
    return LowerFORMAL_ARGUMENTS(Op, DAG, VarArgsFrameIndex);
  case ISD::CALL:
    return LowerCALL(Op, DAG, SPUTM.getSubtargetImpl());
  case ISD::RET:
    return LowerRET(Op, DAG, getTargetMachine());


  // i8, i64 math ops:
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::ROTR:
  case ISD::ROTL:
  case ISD::SRL:
  case ISD::SHL:
  case ISD::SRA: {
    if (VT == MVT::i8)
      return LowerI8Math(Op, DAG, Opc);
    else if (VT == MVT::i64)
      return LowerI64Math(Op, DAG, Opc);
    break;
  }

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
    if (VT.isVector())
      return LowerVectorMUL(Op, DAG);
    else if (VT == MVT::i8)
      return LowerI8Math(Op, DAG, Opc);
    else
      return LowerMUL(Op, DAG, VT, Opc);

  case ISD::FDIV:
    if (VT == MVT::f32 || VT == MVT::v4f32)
      return LowerFDIVf32(Op, DAG);
//    else if (Op.getValueType() == MVT::f64)
//      return LowerFDIVf64(Op, DAG);
    else
      assert(0 && "Calling FDIV on unsupported MVT");

  case ISD::CTPOP:
    return LowerCTPOP(Op, DAG);
  }

  return SDValue();
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
  SDValue Op0 = N->getOperand(0);      // everything has at least one operand
  SDValue Result;                     // Initially, NULL result

  switch (N->getOpcode()) {
  default: break;
  case ISD::ADD: {
    SDValue Op1 = N->getOperand(1);

    if (isa<ConstantSDNode>(Op1) && Op0.getOpcode() == SPUISD::IndirectAddr) {
      SDValue Op01 = Op0.getOperand(1);
      if (Op01.getOpcode() == ISD::Constant
          || Op01.getOpcode() == ISD::TargetConstant) {
        // (add <const>, (SPUindirect <arg>, <const>)) ->
        // (SPUindirect <arg>, <const + const>)
        ConstantSDNode *CN0 = cast<ConstantSDNode>(Op1);
        ConstantSDNode *CN1 = cast<ConstantSDNode>(Op01);
        SDValue combinedConst =
          DAG.getConstant(CN0->getZExtValue() + CN1->getZExtValue(),
                          Op0.getValueType());

        DEBUG(cerr << "Replace: (add " << CN0->getZExtValue() << ", "
                   << "(SPUindirect <arg>, " << CN1->getZExtValue() << "))\n");
        DEBUG(cerr << "With:    (SPUindirect <arg>, "
                   << CN0->getZExtValue() + CN1->getZExtValue() << ")\n");
        return DAG.getNode(SPUISD::IndirectAddr, Op0.getValueType(),
                           Op0.getOperand(0), combinedConst);
      }
    } else if (isa<ConstantSDNode>(Op0)
               && Op1.getOpcode() == SPUISD::IndirectAddr) {
      SDValue Op11 = Op1.getOperand(1);
      if (Op11.getOpcode() == ISD::Constant
          || Op11.getOpcode() == ISD::TargetConstant) {
        // (add (SPUindirect <arg>, <const>), <const>) ->
        // (SPUindirect <arg>, <const + const>)
        ConstantSDNode *CN0 = cast<ConstantSDNode>(Op0);
        ConstantSDNode *CN1 = cast<ConstantSDNode>(Op11);
        SDValue combinedConst =
          DAG.getConstant(CN0->getZExtValue() + CN1->getZExtValue(),
                          Op0.getValueType());

        DEBUG(cerr << "Replace: (add " << CN0->getZExtValue() << ", "
                   << "(SPUindirect <arg>, " << CN1->getZExtValue() << "))\n");
        DEBUG(cerr << "With:    (SPUindirect <arg>, "
                   << CN0->getZExtValue() + CN1->getZExtValue() << ")\n");

        return DAG.getNode(SPUISD::IndirectAddr, Op1.getValueType(),
                           Op1.getOperand(0), combinedConst);
      }
    }
    break;
  }
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND: {
    if (Op0.getOpcode() == SPUISD::EXTRACT_ELT0 &&
        N->getValueType(0) == Op0.getValueType()) {
      // (any_extend (SPUextract_elt0 <arg>)) ->
      // (SPUextract_elt0 <arg>)
      // Types must match, however...
      DEBUG(cerr << "Replace: ");
      DEBUG(N->dump(&DAG));
      DEBUG(cerr << "\nWith:    ");
      DEBUG(Op0.getNode()->dump(&DAG));
      DEBUG(cerr << "\n");

      return Op0;
    }
    break;
  }
  case SPUISD::IndirectAddr: {
    if (!ST->usingLargeMem() && Op0.getOpcode() == SPUISD::AFormAddr) {
      ConstantSDNode *CN = cast<ConstantSDNode>(N->getOperand(1));
      if (CN->getZExtValue() == 0) {
        // (SPUindirect (SPUaform <addr>, 0), 0) ->
        // (SPUaform <addr>, 0)

        DEBUG(cerr << "Replace: ");
        DEBUG(N->dump(&DAG));
        DEBUG(cerr << "\nWith:    ");
        DEBUG(Op0.getNode()->dump(&DAG));
        DEBUG(cerr << "\n");

        return Op0;
      }
    }
    break;
  }
  case SPUISD::SHLQUAD_L_BITS:
  case SPUISD::SHLQUAD_L_BYTES:
  case SPUISD::VEC_SHL:
  case SPUISD::VEC_SRL:
  case SPUISD::VEC_SRA:
  case SPUISD::ROTQUAD_RZ_BYTES:
  case SPUISD::ROTQUAD_RZ_BITS: {
    SDValue Op1 = N->getOperand(1);

    if (isa<ConstantSDNode>(Op1)) {
      // Kill degenerate vector shifts:
      ConstantSDNode *CN = cast<ConstantSDNode>(Op1);

      if (CN->getZExtValue() == 0) {
        Result = Op0;
      }
    }
    break;
  }
  case SPUISD::PROMOTE_SCALAR: {
    switch (Op0.getOpcode()) {
    default:
      break;
    case ISD::ANY_EXTEND:
    case ISD::ZERO_EXTEND:
    case ISD::SIGN_EXTEND: {
      // (SPUpromote_scalar (any|sign|zero_extend (SPUextract_elt0 <arg>))) ->
      // <arg>
      // but only if the SPUpromote_scalar and <arg> types match.
      SDValue Op00 = Op0.getOperand(0);
      if (Op00.getOpcode() == SPUISD::EXTRACT_ELT0) {
        SDValue Op000 = Op00.getOperand(0);
        if (Op000.getValueType() == N->getValueType(0)) {
          Result = Op000;
        }
      }
      break;
    }
    case SPUISD::EXTRACT_ELT0: {
      // (SPUpromote_scalar (SPUextract_elt0 <arg>)) ->
      // <arg>
      Result = Op0.getOperand(0);
      break;
    }
    }
    break;
  }
  }
  // Otherwise, return unchanged.
#if 1
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
#endif

  switch (Op.getOpcode()) {
  default:
    // KnownZero = KnownOne = APInt(Mask.getBitWidth(), 0);
    break;

#if 0
  case CALL:
  case SHUFB:
  case INSERT_MASK:
  case CNTB:
#endif

  case SPUISD::PROMOTE_SCALAR: {
    SDValue Op0 = Op.getOperand(0);
    MVT Op0VT = Op0.getValueType();
    unsigned Op0VTBits = Op0VT.getSizeInBits();
    uint64_t InMask = Op0VT.getIntegerVTBitMask();
    KnownZero |= APInt(Op0VTBits, ~InMask, false);
    KnownOne |= APInt(Op0VTBits, InMask, false);
    break;
  }

  case SPUISD::LDRESULT:
  case SPUISD::EXTRACT_ELT0:
  case SPUISD::EXTRACT_ELT0_CHAINED: {
    MVT OpVT = Op.getValueType();
    unsigned OpVTBits = OpVT.getSizeInBits();
    uint64_t InMask = OpVT.getIntegerVTBitMask();
    KnownZero |= APInt(OpVTBits, ~InMask, false);
    KnownOne |= APInt(OpVTBits, InMask, false);
    break;
  }

#if 0
  case EXTRACT_I1_ZEXT:
  case EXTRACT_I1_SEXT:
  case EXTRACT_I8_ZEXT:
  case EXTRACT_I8_SEXT:
  case MPY:
  case MPYU:
  case MPYH:
  case MPYHH:
  case SPUISD::SHLQUAD_L_BITS:
  case SPUISD::SHLQUAD_L_BYTES:
  case SPUISD::VEC_SHL:
  case SPUISD::VEC_SRL:
  case SPUISD::VEC_SRA:
  case SPUISD::VEC_ROTL:
  case SPUISD::VEC_ROTR:
  case SPUISD::ROTQUAD_RZ_BYTES:
  case SPUISD::ROTQUAD_RZ_BITS:
  case SPUISD::ROTBYTES_RIGHT_S:
  case SPUISD::ROTBYTES_LEFT:
  case SPUISD::ROTBYTES_LEFT_CHAINED:
  case SPUISD::SELECT_MASK:
  case SPUISD::SELB:
  case SPUISD::FPInterp:
  case SPUISD::FPRecipEst:
  case SPUISD::SEXT32TO64:
#endif
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
