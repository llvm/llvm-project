//===-- SPUISelDAGToDAG.cpp - CellSPU pattern matching inst selector ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for the Cell SPU,
// converting from a legalized dag to a SPU-target dag.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPUTargetMachine.h"
#include "SPUISelLowering.h"
#include "SPUHazardRecognizers.h"
#include "SPUFrameInfo.h"
#include "SPURegisterNames.h"
#include "SPUTargetMachine.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  //! ConstantSDNode predicate for i32 sign-extended, 10-bit immediates
  bool
  isI64IntS10Immediate(ConstantSDNode *CN)
  {
    return isS10Constant(CN->getSExtValue());
  }

  //! ConstantSDNode predicate for i32 sign-extended, 10-bit immediates
  bool
  isI32IntS10Immediate(ConstantSDNode *CN)
  {
    return isS10Constant(CN->getSExtValue());
  }

  //! ConstantSDNode predicate for i32 unsigned 10-bit immediate values
  bool
  isI32IntU10Immediate(ConstantSDNode *CN)
  {
    return isU10Constant(CN->getSExtValue());
  }

  //! ConstantSDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntS10Immediate(ConstantSDNode *CN)
  {
    return isS10Constant(CN->getSExtValue());
  }

  //! SDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntS10Immediate(SDNode *N)
  {
    ConstantSDNode *CN = dyn_cast<ConstantSDNode>(N);
    return (CN != 0 && isI16IntS10Immediate(CN));
  }

  //! ConstantSDNode predicate for i16 unsigned 10-bit immediate values
  bool
  isI16IntU10Immediate(ConstantSDNode *CN)
  {
    return isU10Constant((short) CN->getZExtValue());
  }

  //! SDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntU10Immediate(SDNode *N)
  {
    return (N->getOpcode() == ISD::Constant
            && isI16IntU10Immediate(cast<ConstantSDNode>(N)));
  }

  //! ConstantSDNode predicate for signed 16-bit values
  /*!
    \arg CN The constant SelectionDAG node holding the value
    \arg Imm The returned 16-bit value, if returning true

    This predicate tests the value in \a CN to see whether it can be
    represented as a 16-bit, sign-extended quantity. Returns true if
    this is the case.
   */
  bool
  isIntS16Immediate(ConstantSDNode *CN, short &Imm)
  {
    MVT vt = CN->getValueType(0);
    Imm = (short) CN->getZExtValue();
    if (vt.getSimpleVT() >= MVT::i1 && vt.getSimpleVT() <= MVT::i16) {
      return true;
    } else if (vt == MVT::i32) {
      int32_t i_val = (int32_t) CN->getZExtValue();
      short s_val = (short) i_val;
      return i_val == s_val;
    } else {
      int64_t i_val = (int64_t) CN->getZExtValue();
      short s_val = (short) i_val;
      return i_val == s_val;
    }

    return false;
  }

  //! SDNode predicate for signed 16-bit values.
  bool
  isIntS16Immediate(SDNode *N, short &Imm)
  {
    return (N->getOpcode() == ISD::Constant
            && isIntS16Immediate(cast<ConstantSDNode>(N), Imm));
  }

  //! ConstantFPSDNode predicate for representing floats as 16-bit sign ext.
  static bool
  isFPS16Immediate(ConstantFPSDNode *FPN, short &Imm)
  {
    MVT vt = FPN->getValueType(0);
    if (vt == MVT::f32) {
      int val = FloatToBits(FPN->getValueAPF().convertToFloat());
      int sval = (int) ((val << 16) >> 16);
      Imm = (short) val;
      return val == sval;
    }

    return false;
  }

  bool
  isHighLow(const SDValue &Op)
  {
    return (Op.getOpcode() == SPUISD::IndirectAddr
            && ((Op.getOperand(0).getOpcode() == SPUISD::Hi
                 && Op.getOperand(1).getOpcode() == SPUISD::Lo)
                || (Op.getOperand(0).getOpcode() == SPUISD::Lo
                    && Op.getOperand(1).getOpcode() == SPUISD::Hi)));
  }

  //===------------------------------------------------------------------===//
  //! MVT to "useful stuff" mapping structure:

  struct valtype_map_s {
    MVT VT;
    unsigned ldresult_ins;      /// LDRESULT instruction (0 = undefined)
    bool ldresult_imm;          /// LDRESULT instruction requires immediate?
    unsigned lrinst;            /// LR instruction
  };

  const valtype_map_s valtype_map[] = {
    { MVT::i8,    SPU::ORBIr8,  true,  SPU::LRr8 },
    { MVT::i16,   SPU::ORHIr16, true,  SPU::LRr16 },
    { MVT::i32,   SPU::ORIr32,  true,  SPU::LRr32 },
    { MVT::i64,   SPU::ORr64,   false, SPU::LRr64 },
    { MVT::f32,   SPU::ORf32,   false, SPU::LRf32 },
    { MVT::f64,   SPU::ORf64,   false, SPU::LRf64 },
    // vector types... (sigh!)
    { MVT::v16i8, 0,            false, SPU::LRv16i8 },
    { MVT::v8i16, 0,            false, SPU::LRv8i16 },
    { MVT::v4i32, 0,            false, SPU::LRv4i32 },
    { MVT::v2i64, 0,            false, SPU::LRv2i64 },
    { MVT::v4f32, 0,            false, SPU::LRv4f32 },
    { MVT::v2f64, 0,            false, SPU::LRv2f64 }
  };

  const size_t n_valtype_map = sizeof(valtype_map) / sizeof(valtype_map[0]);

  const valtype_map_s *getValueTypeMapEntry(MVT VT)
  {
    const valtype_map_s *retval = 0;
    for (size_t i = 0; i < n_valtype_map; ++i) {
      if (valtype_map[i].VT == VT) {
        retval = valtype_map + i;
        break;
      }
    }


#ifndef NDEBUG
    if (retval == 0) {
      cerr << "SPUISelDAGToDAG.cpp: getValueTypeMapEntry returns NULL for "
           << VT.getMVTString()
           << "\n";
      abort();
    }
#endif

    return retval;
  }
}

namespace {

//===--------------------------------------------------------------------===//
/// SPUDAGToDAGISel - Cell SPU-specific code to select SPU machine
/// instructions for SelectionDAG operations.
///
class SPUDAGToDAGISel :
  public SelectionDAGISel
{
  SPUTargetMachine &TM;
  SPUTargetLowering &SPUtli;
  unsigned GlobalBaseReg;

public:
  explicit SPUDAGToDAGISel(SPUTargetMachine &tm) :
    SelectionDAGISel(tm),
    TM(tm),
    SPUtli(*tm.getTargetLowering())
  { }

  virtual bool runOnFunction(Function &Fn) {
    // Make sure we re-emit a set of the global base reg if necessary
    GlobalBaseReg = 0;
    SelectionDAGISel::runOnFunction(Fn);
    return true;
  }

  /// getI32Imm - Return a target constant with the specified value, of type
  /// i32.
  inline SDValue getI32Imm(uint32_t Imm) {
    return CurDAG->getTargetConstant(Imm, MVT::i32);
  }

  /// getI64Imm - Return a target constant with the specified value, of type
  /// i64.
  inline SDValue getI64Imm(uint64_t Imm) {
    return CurDAG->getTargetConstant(Imm, MVT::i64);
  }

  /// getSmallIPtrImm - Return a target constant of pointer type.
  inline SDValue getSmallIPtrImm(unsigned Imm) {
    return CurDAG->getTargetConstant(Imm, SPUtli.getPointerTy());
    }

  SDNode *emitBuildVector(SDValue build_vec) {
    MVT vecVT = build_vec.getValueType();
    SDNode *bvNode = build_vec.getNode();

    // Check to see if this vector can be represented as a CellSPU immediate
    // constant by invoking all of the instruction selection predicates:
    if (((vecVT == MVT::v8i16) &&
         (SPU::get_vec_i16imm(bvNode, *CurDAG, MVT::i16).getNode() != 0)) ||
        ((vecVT == MVT::v4i32) &&
         ((SPU::get_vec_i16imm(bvNode, *CurDAG, MVT::i32).getNode() != 0) ||
          (SPU::get_ILHUvec_imm(bvNode, *CurDAG, MVT::i32).getNode() != 0) ||
          (SPU::get_vec_u18imm(bvNode, *CurDAG, MVT::i32).getNode() != 0) ||
          (SPU::get_v4i32_imm(bvNode, *CurDAG).getNode() != 0))) ||
        ((vecVT == MVT::v2i64) &&
         ((SPU::get_vec_i16imm(bvNode, *CurDAG, MVT::i64).getNode() != 0) ||
          (SPU::get_ILHUvec_imm(bvNode, *CurDAG, MVT::i64).getNode() != 0) ||
          (SPU::get_vec_u18imm(bvNode, *CurDAG, MVT::i64).getNode() != 0))))
      return Select(build_vec);

    // No, need to emit a constant pool spill:
    std::vector<Constant*> CV;

    for (size_t i = 0; i < build_vec.getNumOperands(); ++i) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode > (build_vec.getOperand(i));
      CV.push_back(const_cast<ConstantInt *> (V->getConstantIntValue()));
    }

    Constant *CP = ConstantVector::get(CV);
    SDValue CPIdx = CurDAG->getConstantPool(CP, SPUtli.getPointerTy());
    unsigned Alignment = 1 << cast<ConstantPoolSDNode > (CPIdx)->getAlignment();
    SDValue CGPoolOffset =
            SPU::LowerConstantPool(CPIdx, *CurDAG,
                                   SPUtli.getSPUTargetMachine());
    return SelectCode(CurDAG->getLoad(build_vec.getValueType(),
                                      CurDAG->getEntryNode(), CGPoolOffset,
                                      PseudoSourceValue::getConstantPool(), 0,
                                      false, Alignment));
  }

  /// Select - Convert the specified operand from a target-independent to a
  /// target-specific node if it hasn't already been changed.
  SDNode *Select(SDValue Op);

  //! Emit the instruction sequence for i64 shl
  SDNode *SelectSHLi64(SDValue &Op, MVT OpVT);

  //! Emit the instruction sequence for i64 srl
  SDNode *SelectSRLi64(SDValue &Op, MVT OpVT);

  //! Emit the instruction sequence for i64 sra
  SDNode *SelectSRAi64(SDValue &Op, MVT OpVT);

  //! Emit the necessary sequence for loading i64 constants:
  SDNode *SelectI64Constant(SDValue &Op, MVT OpVT);

  //! Returns true if the address N is an A-form (local store) address
  bool SelectAFormAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Index);

  //! D-form address predicate
  bool SelectDFormAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Index);

  /// Alternate D-form address using i7 offset predicate
  bool SelectDForm2Addr(SDValue Op, SDValue N, SDValue &Disp,
                        SDValue &Base);

  /// D-form address selection workhorse
  bool DFormAddressPredicate(SDValue Op, SDValue N, SDValue &Disp,
                             SDValue &Base, int minOffset, int maxOffset);

  //! Address predicate if N can be expressed as an indexed [r+r] operation.
  bool SelectXFormAddr(SDValue Op, SDValue N, SDValue &Base,
                       SDValue &Index);

  /// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
  /// inline asm expressions.
  virtual bool SelectInlineAsmMemoryOperand(const SDValue &Op,
                                            char ConstraintCode,
                                            std::vector<SDValue> &OutOps) {
    SDValue Op0, Op1;
    switch (ConstraintCode) {
    default: return true;
    case 'm':   // memory
      if (!SelectDFormAddr(Op, Op, Op0, Op1)
          && !SelectAFormAddr(Op, Op, Op0, Op1))
        SelectXFormAddr(Op, Op, Op0, Op1);
      break;
    case 'o':   // offsetable
      if (!SelectDFormAddr(Op, Op, Op0, Op1)
          && !SelectAFormAddr(Op, Op, Op0, Op1)) {
        Op0 = Op;
        Op1 = getSmallIPtrImm(0);
      }
      break;
    case 'v':   // not offsetable
#if 1
      assert(0 && "InlineAsmMemoryOperand 'v' constraint not handled.");
#else
      SelectAddrIdxOnly(Op, Op, Op0, Op1);
#endif
      break;
    }

    OutOps.push_back(Op0);
    OutOps.push_back(Op1);
    return false;
  }

  /// InstructionSelect - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelect();

  virtual const char *getPassName() const {
    return "Cell SPU DAG->DAG Pattern Instruction Selection";
  }

  /// CreateTargetHazardRecognizer - Return the hazard recognizer to use for
  /// this target when scheduling the DAG.
  virtual ScheduleHazardRecognizer *CreateTargetHazardRecognizer() {
    const TargetInstrInfo *II = TM.getInstrInfo();
    assert(II && "No InstrInfo?");
    return new SPUHazardRecognizer(*II);
  }

  // Include the pieces autogenerated from the target description.
#include "SPUGenDAGISel.inc"
};

}

/// InstructionSelect - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void
SPUDAGToDAGISel::InstructionSelect()
{
  DEBUG(BB->dump());

  // Select target instructions for the DAG.
  SelectRoot(*CurDAG);
  CurDAG->RemoveDeadNodes();
}

/*!
 \arg Op The ISD instruction operand
 \arg N The address to be tested
 \arg Base The base address
 \arg Index The base address index
 */
bool
SPUDAGToDAGISel::SelectAFormAddr(SDValue Op, SDValue N, SDValue &Base,
                    SDValue &Index) {
  // These match the addr256k operand type:
  MVT OffsVT = MVT::i16;
  SDValue Zero = CurDAG->getTargetConstant(0, OffsVT);

  switch (N.getOpcode()) {
  case ISD::Constant:
  case ISD::ConstantPool:
  case ISD::GlobalAddress:
    cerr << "SPU SelectAFormAddr: Constant/Pool/Global not lowered.\n";
    abort();
    /*NOTREACHED*/

  case ISD::TargetConstant:
  case ISD::TargetGlobalAddress:
  case ISD::TargetJumpTable:
    cerr << "SPUSelectAFormAddr: Target Constant/Pool/Global not wrapped as "
         << "A-form address.\n";
    abort();
    /*NOTREACHED*/

  case SPUISD::AFormAddr:
    // Just load from memory if there's only a single use of the location,
    // otherwise, this will get handled below with D-form offset addresses
    if (N.hasOneUse()) {
      SDValue Op0 = N.getOperand(0);
      switch (Op0.getOpcode()) {
      case ISD::TargetConstantPool:
      case ISD::TargetJumpTable:
        Base = Op0;
        Index = Zero;
        return true;

      case ISD::TargetGlobalAddress: {
        GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op0);
        GlobalValue *GV = GSDN->getGlobal();
        if (GV->getAlignment() == 16) {
          Base = Op0;
          Index = Zero;
          return true;
        }
        break;
      }
      }
    }
    break;
  }
  return false;
}

bool
SPUDAGToDAGISel::SelectDForm2Addr(SDValue Op, SDValue N, SDValue &Disp,
                                  SDValue &Base) {
  const int minDForm2Offset = -(1 << 7);
  const int maxDForm2Offset = (1 << 7) - 1;
  return DFormAddressPredicate(Op, N, Disp, Base, minDForm2Offset,
                               maxDForm2Offset);
}

/*!
  \arg Op The ISD instruction (ignored)
  \arg N The address to be tested
  \arg Base Base address register/pointer
  \arg Index Base address index

  Examine the input address by a base register plus a signed 10-bit
  displacement, [r+I10] (D-form address).

  \return true if \a N is a D-form address with \a Base and \a Index set
  to non-empty SDValue instances.
*/
bool
SPUDAGToDAGISel::SelectDFormAddr(SDValue Op, SDValue N, SDValue &Base,
                                 SDValue &Index) {
  return DFormAddressPredicate(Op, N, Base, Index,
                               SPUFrameInfo::minFrameOffset(),
                               SPUFrameInfo::maxFrameOffset());
}

bool
SPUDAGToDAGISel::DFormAddressPredicate(SDValue Op, SDValue N, SDValue &Base,
                                      SDValue &Index, int minOffset,
                                      int maxOffset) {
  unsigned Opc = N.getOpcode();
  MVT PtrTy = SPUtli.getPointerTy();

  if (Opc == ISD::FrameIndex) {
    // Stack frame index must be less than 512 (divided by 16):
    FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N);
    int FI = int(FIN->getIndex());
    DEBUG(cerr << "SelectDFormAddr: ISD::FrameIndex = "
               << FI << "\n");
    if (SPUFrameInfo::FItoStackOffset(FI) < maxOffset) {
      Base = CurDAG->getTargetConstant(0, PtrTy);
      Index = CurDAG->getTargetFrameIndex(FI, PtrTy);
      return true;
    }
  } else if (Opc == ISD::ADD) {
    // Generated by getelementptr
    const SDValue Op0 = N.getOperand(0);
    const SDValue Op1 = N.getOperand(1);

    if ((Op0.getOpcode() == SPUISD::Hi && Op1.getOpcode() == SPUISD::Lo)
        || (Op1.getOpcode() == SPUISD::Hi && Op0.getOpcode() == SPUISD::Lo)) {
      Base = CurDAG->getTargetConstant(0, PtrTy);
      Index = N;
      return true;
    } else if (Op1.getOpcode() == ISD::Constant
               || Op1.getOpcode() == ISD::TargetConstant) {
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op1);
      int32_t offset = int32_t(CN->getSExtValue());

      if (Op0.getOpcode() == ISD::FrameIndex) {
        FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Op0);
        int FI = int(FIN->getIndex());
        DEBUG(cerr << "SelectDFormAddr: ISD::ADD offset = " << offset
                   << " frame index = " << FI << "\n");

        if (SPUFrameInfo::FItoStackOffset(FI) < maxOffset) {
          Base = CurDAG->getTargetConstant(offset, PtrTy);
          Index = CurDAG->getTargetFrameIndex(FI, PtrTy);
          return true;
        }
      } else if (offset > minOffset && offset < maxOffset) {
        Base = CurDAG->getTargetConstant(offset, PtrTy);
        Index = Op0;
        return true;
      }
    } else if (Op0.getOpcode() == ISD::Constant
               || Op0.getOpcode() == ISD::TargetConstant) {
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op0);
      int32_t offset = int32_t(CN->getSExtValue());

      if (Op1.getOpcode() == ISD::FrameIndex) {
        FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Op1);
        int FI = int(FIN->getIndex());
        DEBUG(cerr << "SelectDFormAddr: ISD::ADD offset = " << offset
                   << " frame index = " << FI << "\n");

        if (SPUFrameInfo::FItoStackOffset(FI) < maxOffset) {
          Base = CurDAG->getTargetConstant(offset, PtrTy);
          Index = CurDAG->getTargetFrameIndex(FI, PtrTy);
          return true;
        }
      } else if (offset > minOffset && offset < maxOffset) {
        Base = CurDAG->getTargetConstant(offset, PtrTy);
        Index = Op1;
        return true;
      }
    }
  } else if (Opc == SPUISD::IndirectAddr) {
    // Indirect with constant offset -> D-Form address
    const SDValue Op0 = N.getOperand(0);
    const SDValue Op1 = N.getOperand(1);

    if (Op0.getOpcode() == SPUISD::Hi
        && Op1.getOpcode() == SPUISD::Lo) {
      // (SPUindirect (SPUhi <arg>, 0), (SPUlo <arg>, 0))
      Base = CurDAG->getTargetConstant(0, PtrTy);
      Index = N;
      return true;
    } else if (isa<ConstantSDNode>(Op0) || isa<ConstantSDNode>(Op1)) {
      int32_t offset = 0;
      SDValue idxOp;

      if (isa<ConstantSDNode>(Op1)) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op1);
        offset = int32_t(CN->getSExtValue());
        idxOp = Op0;
      } else if (isa<ConstantSDNode>(Op0)) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op0);
        offset = int32_t(CN->getSExtValue());
        idxOp = Op1;
      }

      if (offset >= minOffset && offset <= maxOffset) {
        Base = CurDAG->getTargetConstant(offset, PtrTy);
        Index = idxOp;
        return true;
      }
    }
  } else if (Opc == SPUISD::AFormAddr) {
    Base = CurDAG->getTargetConstant(0, N.getValueType());
    Index = N;
    return true;
  } else if (Opc == SPUISD::LDRESULT) {
    Base = CurDAG->getTargetConstant(0, N.getValueType());
    Index = N;
    return true;
  } else if (Opc == ISD::Register || Opc == ISD::CopyFromReg) {
    unsigned OpOpc = Op.getOpcode();

    if (OpOpc == ISD::STORE || OpOpc == ISD::LOAD) {
      // Direct load/store without getelementptr
      SDValue Addr, Offs;

      // Get the register from CopyFromReg
      if (Opc == ISD::CopyFromReg)
        Addr = N.getOperand(1);
      else
        Addr = N;                       // Register

      Offs = ((OpOpc == ISD::STORE) ? Op.getOperand(3) : Op.getOperand(2));

      if (Offs.getOpcode() == ISD::Constant || Offs.getOpcode() == ISD::UNDEF) {
        if (Offs.getOpcode() == ISD::UNDEF)
          Offs = CurDAG->getTargetConstant(0, Offs.getValueType());

        Base = Offs;
        Index = Addr;
        return true;
      }
    } else {
      /* If otherwise unadorned, default to D-form address with 0 offset: */
      if (Opc == ISD::CopyFromReg) {
        Index = N.getOperand(1);
      } else {
        Index = N;
      }

      Base = CurDAG->getTargetConstant(0, Index.getValueType());
      return true;
    }
  }

  return false;
}

/*!
  \arg Op The ISD instruction operand
  \arg N The address operand
  \arg Base The base pointer operand
  \arg Index The offset/index operand

  If the address \a N can be expressed as an A-form or D-form address, returns
  false.  Otherwise, creates two operands, Base and Index that will become the
  (r)(r) X-form address.
*/
bool
SPUDAGToDAGISel::SelectXFormAddr(SDValue Op, SDValue N, SDValue &Base,
                                 SDValue &Index) {
  if (!SelectAFormAddr(Op, N, Base, Index)
      && !SelectDFormAddr(Op, N, Base, Index)) {
    // If the address is neither A-form or D-form, punt and use an X-form
    // address:
    Base = N.getOperand(1);
    Index = N.getOperand(0);
    return true;
  }

  return false;
}

//! Convert the operand from a target-independent to a target-specific node
/*!
 */
SDNode *
SPUDAGToDAGISel::Select(SDValue Op) {
  SDNode *N = Op.getNode();
  unsigned Opc = N->getOpcode();
  int n_ops = -1;
  unsigned NewOpc;
  MVT OpVT = Op.getValueType();
  SDValue Ops[8];

  if (N->isMachineOpcode()) {
    return NULL;   // Already selected.
  }

  if (Opc == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, Op.getValueType());
    SDValue Imm0 = CurDAG->getTargetConstant(0, Op.getValueType());

    if (FI < 128) {
      NewOpc = SPU::AIr32;
      Ops[0] = TFI;
      Ops[1] = Imm0;
      n_ops = 2;
    } else {
      NewOpc = SPU::Ar32;
      Ops[0] = CurDAG->getRegister(SPU::R1, Op.getValueType());
      Ops[1] = SDValue(CurDAG->getTargetNode(SPU::ILAr32, Op.getValueType(),
                                             TFI, Imm0), 0);
      n_ops = 2;
    }
  } else if (Opc == ISD::Constant && OpVT == MVT::i64) {
    // Catch the i64 constants that end up here. Note: The backend doesn't
    // attempt to legalize the constant (it's useless because DAGCombiner
    // will insert 64-bit constants and we can't stop it).
    return SelectI64Constant(Op, OpVT);
  } else if ((Opc == ISD::ZERO_EXTEND || Opc == ISD::ANY_EXTEND)
             && OpVT == MVT::i64) {
    SDValue Op0 = Op.getOperand(0);
    MVT Op0VT = Op0.getValueType();
    MVT Op0VecVT = MVT::getVectorVT(Op0VT, (128 / Op0VT.getSizeInBits()));
    MVT OpVecVT = MVT::getVectorVT(OpVT, (128 / OpVT.getSizeInBits()));
    SDValue shufMask;

    switch (Op0VT.getSimpleVT()) {
    default:
      cerr << "CellSPU Select: Unhandled zero/any extend MVT\n";
      abort();
      /*NOTREACHED*/
      break;
    case MVT::i32:
      shufMask = CurDAG->getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x00010203, MVT::i32),
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x08090a0b, MVT::i32));
      break;

    case MVT::i16:
      shufMask = CurDAG->getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x80800203, MVT::i32),
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x80800a0b, MVT::i32));
      break;

    case MVT::i8:
      shufMask = CurDAG->getNode(ISD::BUILD_VECTOR, MVT::v4i32,
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x80808003, MVT::i32),
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x8080800b, MVT::i32));
      break;
    }

    SDNode *shufMaskLoad = emitBuildVector(shufMask);
    SDNode *PromoteScalar =
            SelectCode(CurDAG->getNode(SPUISD::PREFSLOT2VEC, Op0VecVT, Op0));

    SDValue zextShuffle =
            CurDAG->getNode(SPUISD::SHUFB, OpVecVT,
                            SDValue(PromoteScalar, 0),
                            SDValue(PromoteScalar, 0),
                            SDValue(shufMaskLoad, 0));

    // N.B.: BIT_CONVERT replaces and updates the zextShuffle node, so we
    // re-use it in the VEC2PREFSLOT selection without needing to explicitly
    // call SelectCode (it's already done for us.)
    SelectCode(CurDAG->getNode(ISD::BIT_CONVERT, OpVecVT, zextShuffle));
    return SelectCode(CurDAG->getNode(SPUISD::VEC2PREFSLOT, OpVT,
                                      zextShuffle));
  } else if (Opc == ISD::ADD && (OpVT == MVT::i64 || OpVT == MVT::v2i64)) {
    SDNode *CGLoad =
            emitBuildVector(SPU::getCarryGenerateShufMask(*CurDAG));

    return SelectCode(CurDAG->getNode(SPUISD::ADD64_MARKER, OpVT,
                                      Op.getOperand(0), Op.getOperand(1),
                                      SDValue(CGLoad, 0)));
  } else if (Opc == ISD::SUB && (OpVT == MVT::i64 || OpVT == MVT::v2i64)) {
    SDNode *CGLoad =
            emitBuildVector(SPU::getBorrowGenerateShufMask(*CurDAG));

    return SelectCode(CurDAG->getNode(SPUISD::SUB64_MARKER, OpVT,
                                      Op.getOperand(0), Op.getOperand(1),
                                      SDValue(CGLoad, 0)));
  } else if (Opc == ISD::MUL && (OpVT == MVT::i64 || OpVT == MVT::v2i64)) {
    SDNode *CGLoad =
            emitBuildVector(SPU::getCarryGenerateShufMask(*CurDAG));

    return SelectCode(CurDAG->getNode(SPUISD::MUL64_MARKER, OpVT,
                                      Op.getOperand(0), Op.getOperand(1),
                                      SDValue(CGLoad, 0)));
  } else if (Opc == ISD::TRUNCATE) {
    SDValue Op0 = Op.getOperand(0);
    if ((Op0.getOpcode() == ISD::SRA || Op0.getOpcode() == ISD::SRL)
        && OpVT == MVT::i32
        && Op0.getValueType() == MVT::i64) {
      // Catch (truncate:i32 ([sra|srl]:i64 arg, c), where c >= 32
      //
      // Take advantage of the fact that the upper 32 bits are in the
      // i32 preferred slot and avoid shuffle gymnastics:
      ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op0.getOperand(1));
      if (CN != 0) {
        unsigned shift_amt = unsigned(CN->getZExtValue());

        if (shift_amt >= 32) {
          SDNode *hi32 =
                  CurDAG->getTargetNode(SPU::ORr32_r64, OpVT, Op0.getOperand(0));

          shift_amt -= 32;
          if (shift_amt > 0) {
            // Take care of the additional shift, if present:
            SDValue shift = CurDAG->getTargetConstant(shift_amt, MVT::i32);
            unsigned Opc = SPU::ROTMAIr32_i32;

            if (Op0.getOpcode() == ISD::SRL)
              Opc = SPU::ROTMr32;

            hi32 = CurDAG->getTargetNode(Opc, OpVT, SDValue(hi32, 0), shift);
          }

          return hi32;
        }
      }
    }
  } else if (Opc == ISD::SHL) {
    if (OpVT == MVT::i64) {
      return SelectSHLi64(Op, OpVT);
    }
  } else if (Opc == ISD::SRL) {
    if (OpVT == MVT::i64) {
      return SelectSRLi64(Op, OpVT);
    }
  } else if (Opc == ISD::SRA) {
    if (OpVT == MVT::i64) {
      return SelectSRAi64(Op, OpVT);
    }
  } else if (Opc == SPUISD::LDRESULT) {
    // Custom select instructions for LDRESULT
    MVT VT = N->getValueType(0);
    SDValue Arg = N->getOperand(0);
    SDValue Chain = N->getOperand(1);
    SDNode *Result;
    const valtype_map_s *vtm = getValueTypeMapEntry(VT);

    if (vtm->ldresult_ins == 0) {
      cerr << "LDRESULT for unsupported type: "
           << VT.getMVTString()
           << "\n";
      abort();
    }

    Opc = vtm->ldresult_ins;
    if (vtm->ldresult_imm) {
      SDValue Zero = CurDAG->getTargetConstant(0, VT);

      Result = CurDAG->getTargetNode(Opc, VT, MVT::Other, Arg, Zero, Chain);
    } else {
      Result = CurDAG->getTargetNode(Opc, VT, MVT::Other, Arg, Arg, Chain);
    }

    return Result;
  } else if (Opc == SPUISD::IndirectAddr) {
    // Look at the operands: SelectCode() will catch the cases that aren't
    // specifically handled here.
    //
    // SPUInstrInfo catches the following patterns:
    // (SPUindirect (SPUhi ...), (SPUlo ...))
    // (SPUindirect $sp, imm)
    MVT VT = Op.getValueType();
    SDValue Op0 = N->getOperand(0);
    SDValue Op1 = N->getOperand(1);
    RegisterSDNode *RN;

    if ((Op0.getOpcode() != SPUISD::Hi && Op1.getOpcode() != SPUISD::Lo)
        || (Op0.getOpcode() == ISD::Register
            && ((RN = dyn_cast<RegisterSDNode>(Op0.getNode())) != 0
                && RN->getReg() != SPU::R1))) {
      NewOpc = SPU::Ar32;
      if (Op1.getOpcode() == ISD::Constant) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op1);
        Op1 = CurDAG->getTargetConstant(CN->getSExtValue(), VT);
        NewOpc = (isI32IntS10Immediate(CN) ? SPU::AIr32 : SPU::Ar32);
      }
      Ops[0] = Op0;
      Ops[1] = Op1;
      n_ops = 2;
    }
  }

  if (n_ops > 0) {
    if (N->hasOneUse())
      return CurDAG->SelectNodeTo(N, NewOpc, OpVT, Ops, n_ops);
    else
      return CurDAG->getTargetNode(NewOpc, OpVT, Ops, n_ops);
  } else
    return SelectCode(Op);
}

/*!
 * Emit the instruction sequence for i64 left shifts. The basic algorithm
 * is to fill the bottom two word slots with zeros so that zeros are shifted
 * in as the entire quadword is shifted left.
 *
 * \note This code could also be used to implement v2i64 shl.
 *
 * @param Op The shl operand
 * @param OpVT Op's machine value value type (doesn't need to be passed, but
 * makes life easier.)
 * @return The SDNode with the entire instruction sequence
 */
SDNode *
SPUDAGToDAGISel::SelectSHLi64(SDValue &Op, MVT OpVT) {
  SDValue Op0 = Op.getOperand(0);
  MVT VecVT = MVT::getVectorVT(OpVT, (128 / OpVT.getSizeInBits()));
  SDValue ShiftAmt = Op.getOperand(1);
  MVT ShiftAmtVT = ShiftAmt.getValueType();
  SDNode *VecOp0, *SelMask, *ZeroFill, *Shift = 0;
  SDValue SelMaskVal;

  VecOp0 = CurDAG->getTargetNode(SPU::ORv2i64_i64, VecVT, Op0);
  SelMaskVal = CurDAG->getTargetConstant(0xff00ULL, MVT::i16);
  SelMask = CurDAG->getTargetNode(SPU::FSMBIv2i64, VecVT, SelMaskVal);
  ZeroFill = CurDAG->getTargetNode(SPU::ILv2i64, VecVT,
                                   CurDAG->getTargetConstant(0, OpVT));
  VecOp0 = CurDAG->getTargetNode(SPU::SELBv2i64, VecVT,
                                 SDValue(ZeroFill, 0),
                                 SDValue(VecOp0, 0),
                                 SDValue(SelMask, 0));

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(ShiftAmt)) {
    unsigned bytes = unsigned(CN->getZExtValue()) >> 3;
    unsigned bits = unsigned(CN->getZExtValue()) & 7;

    if (bytes > 0) {
      Shift =
        CurDAG->getTargetNode(SPU::SHLQBYIv2i64, VecVT,
                              SDValue(VecOp0, 0),
                              CurDAG->getTargetConstant(bytes, ShiftAmtVT));
    }

    if (bits > 0) {
      Shift =
        CurDAG->getTargetNode(SPU::SHLQBIIv2i64, VecVT,
                              SDValue((Shift != 0 ? Shift : VecOp0), 0),
                              CurDAG->getTargetConstant(bits, ShiftAmtVT));
    }
  } else {
    SDNode *Bytes =
      CurDAG->getTargetNode(SPU::ROTMIr32, ShiftAmtVT,
                            ShiftAmt,
                            CurDAG->getTargetConstant(3, ShiftAmtVT));
    SDNode *Bits =
      CurDAG->getTargetNode(SPU::ANDIr32, ShiftAmtVT,
                            ShiftAmt,
                            CurDAG->getTargetConstant(7, ShiftAmtVT));
    Shift =
      CurDAG->getTargetNode(SPU::SHLQBYv2i64, VecVT,
                            SDValue(VecOp0, 0), SDValue(Bytes, 0));
    Shift =
      CurDAG->getTargetNode(SPU::SHLQBIv2i64, VecVT,
                            SDValue(Shift, 0), SDValue(Bits, 0));
  }

  return CurDAG->getTargetNode(SPU::ORi64_v2i64, OpVT, SDValue(Shift, 0));
}

/*!
 * Emit the instruction sequence for i64 logical right shifts.
 *
 * @param Op The shl operand
 * @param OpVT Op's machine value value type (doesn't need to be passed, but
 * makes life easier.)
 * @return The SDNode with the entire instruction sequence
 */
SDNode *
SPUDAGToDAGISel::SelectSRLi64(SDValue &Op, MVT OpVT) {
  SDValue Op0 = Op.getOperand(0);
  MVT VecVT = MVT::getVectorVT(OpVT, (128 / OpVT.getSizeInBits()));
  SDValue ShiftAmt = Op.getOperand(1);
  MVT ShiftAmtVT = ShiftAmt.getValueType();
  SDNode *VecOp0, *Shift = 0;

  VecOp0 = CurDAG->getTargetNode(SPU::ORv2i64_i64, VecVT, Op0);

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(ShiftAmt)) {
    unsigned bytes = unsigned(CN->getZExtValue()) >> 3;
    unsigned bits = unsigned(CN->getZExtValue()) & 7;

    if (bytes > 0) {
      Shift =
        CurDAG->getTargetNode(SPU::ROTQMBYIv2i64, VecVT,
                              SDValue(VecOp0, 0),
                              CurDAG->getTargetConstant(bytes, ShiftAmtVT));
    }

    if (bits > 0) {
      Shift =
        CurDAG->getTargetNode(SPU::ROTQMBIIv2i64, VecVT,
                              SDValue((Shift != 0 ? Shift : VecOp0), 0),
                              CurDAG->getTargetConstant(bits, ShiftAmtVT));
    }
  } else {
    SDNode *Bytes =
      CurDAG->getTargetNode(SPU::ROTMIr32, ShiftAmtVT,
                            ShiftAmt,
                            CurDAG->getTargetConstant(3, ShiftAmtVT));
    SDNode *Bits =
      CurDAG->getTargetNode(SPU::ANDIr32, ShiftAmtVT,
                            ShiftAmt,
                            CurDAG->getTargetConstant(7, ShiftAmtVT));

    // Ensure that the shift amounts are negated!
    Bytes = CurDAG->getTargetNode(SPU::SFIr32, ShiftAmtVT,
                                  SDValue(Bytes, 0),
                                  CurDAG->getTargetConstant(0, ShiftAmtVT));

    Bits = CurDAG->getTargetNode(SPU::SFIr32, ShiftAmtVT,
                                 SDValue(Bits, 0),
                                 CurDAG->getTargetConstant(0, ShiftAmtVT));

    Shift =
      CurDAG->getTargetNode(SPU::ROTQMBYv2i64, VecVT,
                            SDValue(VecOp0, 0), SDValue(Bytes, 0));
    Shift =
      CurDAG->getTargetNode(SPU::ROTQMBIv2i64, VecVT,
                            SDValue(Shift, 0), SDValue(Bits, 0));
  }

  return CurDAG->getTargetNode(SPU::ORi64_v2i64, OpVT, SDValue(Shift, 0));
}

/*!
 * Emit the instruction sequence for i64 arithmetic right shifts.
 *
 * @param Op The shl operand
 * @param OpVT Op's machine value value type (doesn't need to be passed, but
 * makes life easier.)
 * @return The SDNode with the entire instruction sequence
 */
SDNode *
SPUDAGToDAGISel::SelectSRAi64(SDValue &Op, MVT OpVT) {
  // Promote Op0 to vector
  MVT VecVT = MVT::getVectorVT(OpVT, (128 / OpVT.getSizeInBits()));
  SDValue ShiftAmt = Op.getOperand(1);
  MVT ShiftAmtVT = ShiftAmt.getValueType();

  SDNode *VecOp0 =
    CurDAG->getTargetNode(SPU::ORv2i64_i64, VecVT, Op.getOperand(0));

  SDValue SignRotAmt = CurDAG->getTargetConstant(31, ShiftAmtVT);
  SDNode *SignRot =
    CurDAG->getTargetNode(SPU::ROTMAIv2i64_i32, MVT::v2i64,
                          SDValue(VecOp0, 0), SignRotAmt);
  SDNode *UpperHalfSign =
    CurDAG->getTargetNode(SPU::ORi32_v4i32, MVT::i32, SDValue(SignRot, 0));

  SDNode *UpperHalfSignMask =
    CurDAG->getTargetNode(SPU::FSM64r32, VecVT, SDValue(UpperHalfSign, 0));
  SDNode *UpperLowerMask =
    CurDAG->getTargetNode(SPU::FSMBIv2i64, VecVT,
                          CurDAG->getTargetConstant(0xff00ULL, MVT::i16));
  SDNode *UpperLowerSelect =
    CurDAG->getTargetNode(SPU::SELBv2i64, VecVT,
                          SDValue(UpperHalfSignMask, 0),
                          SDValue(VecOp0, 0),
                          SDValue(UpperLowerMask, 0));

  SDNode *Shift = 0;

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(ShiftAmt)) {
    unsigned bytes = unsigned(CN->getZExtValue()) >> 3;
    unsigned bits = unsigned(CN->getZExtValue()) & 7;

    if (bytes > 0) {
      bytes = 31 - bytes;
      Shift =
        CurDAG->getTargetNode(SPU::ROTQBYIv2i64, VecVT,
                              SDValue(UpperLowerSelect, 0),
                              CurDAG->getTargetConstant(bytes, ShiftAmtVT));
    }

    if (bits > 0) {
      bits = 8 - bits;
      Shift =
        CurDAG->getTargetNode(SPU::ROTQBIIv2i64, VecVT,
                              SDValue((Shift != 0 ? Shift : UpperLowerSelect), 0),
                              CurDAG->getTargetConstant(bits, ShiftAmtVT));
    }
  } else {
    SDNode *NegShift =
      CurDAG->getTargetNode(SPU::SFIr32, ShiftAmtVT,
                            ShiftAmt, CurDAG->getTargetConstant(0, ShiftAmtVT));

    Shift =
      CurDAG->getTargetNode(SPU::ROTQBYBIv2i64_r32, VecVT,
                            SDValue(UpperLowerSelect, 0), SDValue(NegShift, 0));
    Shift =
      CurDAG->getTargetNode(SPU::ROTQBIv2i64, VecVT,
                            SDValue(Shift, 0), SDValue(NegShift, 0));
  }

  return CurDAG->getTargetNode(SPU::ORi64_v2i64, OpVT, SDValue(Shift, 0));
}

/*!
 Do the necessary magic necessary to load a i64 constant
 */
SDNode *SPUDAGToDAGISel::SelectI64Constant(SDValue& Op, MVT OpVT) {
  ConstantSDNode *CN = cast<ConstantSDNode>(Op.getNode());
  MVT OpVecVT = MVT::getVectorVT(OpVT, 2);
  SDValue i64vec =
          SPU::LowerSplat_v2i64(OpVecVT, *CurDAG, CN->getZExtValue());

  // Here's where it gets interesting, because we have to parse out the
  // subtree handed back in i64vec:

  if (i64vec.getOpcode() == ISD::BIT_CONVERT) {
    // The degenerate case where the upper and lower bits in the splat are
    // identical:
    SDValue Op0 = i64vec.getOperand(0);

    ReplaceUses(i64vec, Op0);
    return CurDAG->getTargetNode(SPU::ORi64_v2i64, OpVT,
                                 SDValue(emitBuildVector(Op0), 0));
  } else if (i64vec.getOpcode() == SPUISD::SHUFB) {
    SDValue lhs = i64vec.getOperand(0);
    SDValue rhs = i64vec.getOperand(1);
    SDValue shufmask = i64vec.getOperand(2);

    if (lhs.getOpcode() == ISD::BIT_CONVERT) {
      ReplaceUses(lhs, lhs.getOperand(0));
      lhs = lhs.getOperand(0);
    }

    SDNode *lhsNode = (lhs.getNode()->isMachineOpcode()
                       ? lhs.getNode()
                       : emitBuildVector(lhs));

    if (rhs.getOpcode() == ISD::BIT_CONVERT) {
      ReplaceUses(rhs, rhs.getOperand(0));
      rhs = rhs.getOperand(0);
    }

    SDNode *rhsNode = (rhs.getNode()->isMachineOpcode()
                       ? rhs.getNode()
                       : emitBuildVector(rhs));

    if (shufmask.getOpcode() == ISD::BIT_CONVERT) {
      ReplaceUses(shufmask, shufmask.getOperand(0));
      shufmask = shufmask.getOperand(0);
    }

    SDNode *shufMaskNode = (shufmask.getNode()->isMachineOpcode()
                            ? shufmask.getNode()
                            : emitBuildVector(shufmask));

    SDNode *shufNode =
            Select(CurDAG->getNode(SPUISD::SHUFB, OpVecVT,
                                   SDValue(lhsNode, 0), SDValue(rhsNode, 0),
                                   SDValue(shufMaskNode, 0)));

    return CurDAG->getTargetNode(SPU::ORi64_v2i64, OpVT, SDValue(shufNode, 0));
  } else {
    cerr << "SPUDAGToDAGISel::SelectI64Constant: Unhandled i64vec condition\n";
    abort();
  }
}

/// createSPUISelDag - This pass converts a legalized DAG into a
/// SPU-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSPUISelDag(SPUTargetMachine &TM) {
  return new SPUDAGToDAGISel(TM);
}
