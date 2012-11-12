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
#include "SPUHazardRecognizers.h"
#include "SPUFrameLowering.h"
#include "SPUTargetMachine.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Constants.h"
#include "llvm/GlobalValue.h"
#include "llvm/Intrinsics.h"
#include "llvm/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  //! ConstantSDNode predicate for i32 sign-extended, 10-bit immediates
  bool
  isI32IntS10Immediate(ConstantSDNode *CN)
  {
    return isInt<10>(CN->getSExtValue());
  }

  //! ConstantSDNode predicate for i32 unsigned 10-bit immediate values
  bool
  isI32IntU10Immediate(ConstantSDNode *CN)
  {
    return isUInt<10>(CN->getSExtValue());
  }

  //! ConstantSDNode predicate for i16 sign-extended, 10-bit immediate values
  bool
  isI16IntS10Immediate(ConstantSDNode *CN)
  {
    return isInt<10>(CN->getSExtValue());
  }

  //! ConstantSDNode predicate for i16 unsigned 10-bit immediate values
  bool
  isI16IntU10Immediate(ConstantSDNode *CN)
  {
    return isUInt<10>((short) CN->getZExtValue());
  }

  //! ConstantSDNode predicate for signed 16-bit values
  /*!
    \param CN The constant SelectionDAG node holding the value
    \param Imm The returned 16-bit value, if returning true

    This predicate tests the value in \a CN to see whether it can be
    represented as a 16-bit, sign-extended quantity. Returns true if
    this is the case.
   */
  bool
  isIntS16Immediate(ConstantSDNode *CN, short &Imm)
  {
    EVT vt = CN->getValueType(0);
    Imm = (short) CN->getZExtValue();
    if (vt.getSimpleVT() >= MVT::i1 && vt.getSimpleVT() <= MVT::i16) {
      return true;
    } else if (vt == MVT::i32) {
      int32_t i_val = (int32_t) CN->getZExtValue();
      return i_val == SignExtend32<16>(i_val);
    } else {
      int64_t i_val = (int64_t) CN->getZExtValue();
      return i_val == SignExtend64<16>(i_val);
    }
  }

  //! ConstantFPSDNode predicate for representing floats as 16-bit sign ext.
  static bool
  isFPS16Immediate(ConstantFPSDNode *FPN, short &Imm)
  {
    EVT vt = FPN->getValueType(0);
    if (vt == MVT::f32) {
      int val = FloatToBits(FPN->getValueAPF().convertToFloat());
      if (val == SignExtend32<16>(val)) {
        Imm = (short) val;
        return true;
      }
    }

    return false;
  }

  //! Generate the carry-generate shuffle mask.
  SDValue getCarryGenerateShufMask(SelectionDAG &DAG, DebugLoc dl) {
    SmallVector<SDValue, 16 > ShufBytes;

    // Create the shuffle mask for "rotating" the borrow up one register slot
    // once the borrow is generated.
    ShufBytes.push_back(DAG.getConstant(0x04050607, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x80808080, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x0c0d0e0f, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x80808080, MVT::i32));

    return DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                       &ShufBytes[0], ShufBytes.size());
  }

  //! Generate the borrow-generate shuffle mask
  SDValue getBorrowGenerateShufMask(SelectionDAG &DAG, DebugLoc dl) {
    SmallVector<SDValue, 16 > ShufBytes;

    // Create the shuffle mask for "rotating" the borrow up one register slot
    // once the borrow is generated.
    ShufBytes.push_back(DAG.getConstant(0x04050607, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0xc0c0c0c0, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0x0c0d0e0f, MVT::i32));
    ShufBytes.push_back(DAG.getConstant(0xc0c0c0c0, MVT::i32));

    return DAG.getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                       &ShufBytes[0], ShufBytes.size());
  }

  //===------------------------------------------------------------------===//
  /// SPUDAGToDAGISel - Cell SPU-specific code to select SPU machine
  /// instructions for SelectionDAG operations.
  ///
  class SPUDAGToDAGISel :
    public SelectionDAGISel
  {
    const SPUTargetMachine &TM;
    const SPUTargetLowering &SPUtli;
    unsigned GlobalBaseReg;

  public:
    explicit SPUDAGToDAGISel(SPUTargetMachine &tm) :
      SelectionDAGISel(tm),
      TM(tm),
      SPUtli(*tm.getTargetLowering())
    { }

    virtual bool runOnMachineFunction(MachineFunction &MF) {
      // Make sure we re-emit a set of the global base reg if necessary
      GlobalBaseReg = 0;
      SelectionDAGISel::runOnMachineFunction(MF);
      return true;
    }

    /// getI32Imm - Return a target constant with the specified value, of type
    /// i32.
    inline SDValue getI32Imm(uint32_t Imm) {
      return CurDAG->getTargetConstant(Imm, MVT::i32);
    }

    /// getSmallIPtrImm - Return a target constant of pointer type.
    inline SDValue getSmallIPtrImm(unsigned Imm) {
      return CurDAG->getTargetConstant(Imm, SPUtli.getPointerTy());
    }

    SDNode *emitBuildVector(SDNode *bvNode) {
      EVT vecVT = bvNode->getValueType(0);
      DebugLoc dl = bvNode->getDebugLoc();

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
            (SPU::get_vec_u18imm(bvNode, *CurDAG, MVT::i64).getNode() != 0)))) {
        HandleSDNode Dummy(SDValue(bvNode, 0));
        if (SDNode *N = Select(bvNode))
          return N;
        return Dummy.getValue().getNode();
      }

      // No, need to emit a constant pool spill:
      std::vector<Constant*> CV;

      for (size_t i = 0; i < bvNode->getNumOperands(); ++i) {
        ConstantSDNode *V = cast<ConstantSDNode > (bvNode->getOperand(i));
        CV.push_back(const_cast<ConstantInt *>(V->getConstantIntValue()));
      }

      const Constant *CP = ConstantVector::get(CV);
      SDValue CPIdx = CurDAG->getConstantPool(CP, SPUtli.getPointerTy());
      unsigned Alignment = cast<ConstantPoolSDNode>(CPIdx)->getAlignment();
      SDValue CGPoolOffset =
              SPU::LowerConstantPool(CPIdx, *CurDAG, TM);

      HandleSDNode Dummy(CurDAG->getLoad(vecVT, dl,
                                         CurDAG->getEntryNode(), CGPoolOffset,
                                         MachinePointerInfo::getConstantPool(),
                                         false, false, false, Alignment));
      CurDAG->ReplaceAllUsesWith(SDValue(bvNode, 0), Dummy.getValue());
      if (SDNode *N = SelectCode(Dummy.getValue().getNode()))
        return N;
      return Dummy.getValue().getNode();
    }

    /// Select - Convert the specified operand from a target-independent to a
    /// target-specific node if it hasn't already been changed.
    SDNode *Select(SDNode *N);

    //! Emit the instruction sequence for i64 shl
    SDNode *SelectSHLi64(SDNode *N, EVT OpVT);

    //! Emit the instruction sequence for i64 srl
    SDNode *SelectSRLi64(SDNode *N, EVT OpVT);

    //! Emit the instruction sequence for i64 sra
    SDNode *SelectSRAi64(SDNode *N, EVT OpVT);

    //! Emit the necessary sequence for loading i64 constants:
    SDNode *SelectI64Constant(SDNode *N, EVT OpVT, DebugLoc dl);

    //! Alternate instruction emit sequence for loading i64 constants
    SDNode *SelectI64Constant(uint64_t i64const, EVT OpVT, DebugLoc dl);

    //! Returns true if the address N is an A-form (local store) address
    bool SelectAFormAddr(SDNode *Op, SDValue N, SDValue &Base,
                         SDValue &Index);

    //! D-form address predicate
    bool SelectDFormAddr(SDNode *Op, SDValue N, SDValue &Base,
                         SDValue &Index);

    /// Alternate D-form address using i7 offset predicate
    bool SelectDForm2Addr(SDNode *Op, SDValue N, SDValue &Disp,
                          SDValue &Base);

    /// D-form address selection workhorse
    bool DFormAddressPredicate(SDNode *Op, SDValue N, SDValue &Disp,
                               SDValue &Base, int minOffset, int maxOffset);

    //! Address predicate if N can be expressed as an indexed [r+r] operation.
    bool SelectXFormAddr(SDNode *Op, SDValue N, SDValue &Base,
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
        if (!SelectDFormAddr(Op.getNode(), Op, Op0, Op1)
            && !SelectAFormAddr(Op.getNode(), Op, Op0, Op1))
          SelectXFormAddr(Op.getNode(), Op, Op0, Op1);
        break;
      case 'o':   // offsetable
        if (!SelectDFormAddr(Op.getNode(), Op, Op0, Op1)
            && !SelectAFormAddr(Op.getNode(), Op, Op0, Op1)) {
          Op0 = Op;
          Op1 = getSmallIPtrImm(0);
        }
        break;
      case 'v':   // not offsetable
#if 1
        llvm_unreachable("InlineAsmMemoryOperand 'v' constraint not handled.");
#else
        SelectAddrIdxOnly(Op, Op, Op0, Op1);
        break;
#endif
      }

      OutOps.push_back(Op0);
      OutOps.push_back(Op1);
      return false;
    }

    virtual const char *getPassName() const {
      return "Cell SPU DAG->DAG Pattern Instruction Selection";
    }

  private:
    SDValue getRC( MVT );

    // Include the pieces autogenerated from the target description.
#include "SPUGenDAGISel.inc"
  };
}

/*!
 \param Op The ISD instruction operand
 \param N The address to be tested
 \param Base The base address
 \param Index The base address index
 */
bool
SPUDAGToDAGISel::SelectAFormAddr(SDNode *Op, SDValue N, SDValue &Base,
                    SDValue &Index) {
  // These match the addr256k operand type:
  EVT OffsVT = MVT::i16;
  SDValue Zero = CurDAG->getTargetConstant(0, OffsVT);
  int64_t val;

  switch (N.getOpcode()) {
  case ISD::Constant:
    val = dyn_cast<ConstantSDNode>(N.getNode())->getSExtValue();
    Base = CurDAG->getTargetConstant( val , MVT::i32);
    Index = Zero;
    return true;
  case ISD::ConstantPool:
  case ISD::GlobalAddress:
    report_fatal_error("SPU SelectAFormAddr: Pool/Global not lowered.");
    /*NOTREACHED*/

  case ISD::TargetConstant:
  case ISD::TargetGlobalAddress:
  case ISD::TargetJumpTable:
    report_fatal_error("SPUSelectAFormAddr: Target Constant/Pool/Global "
                      "not wrapped as A-form address.");
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
        const GlobalValue *GV = GSDN->getGlobal();
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
SPUDAGToDAGISel::SelectDForm2Addr(SDNode *Op, SDValue N, SDValue &Disp,
                                  SDValue &Base) {
  const int minDForm2Offset = -(1 << 7);
  const int maxDForm2Offset = (1 << 7) - 1;
  return DFormAddressPredicate(Op, N, Disp, Base, minDForm2Offset,
                               maxDForm2Offset);
}

/*!
  \param Op The ISD instruction (ignored)
  \param N The address to be tested
  \param Base Base address register/pointer
  \param Index Base address index

  Examine the input address by a base register plus a signed 10-bit
  displacement, [r+I10] (D-form address).

  \return true if \a N is a D-form address with \a Base and \a Index set
  to non-empty SDValue instances.
*/
bool
SPUDAGToDAGISel::SelectDFormAddr(SDNode *Op, SDValue N, SDValue &Base,
                                 SDValue &Index) {
  return DFormAddressPredicate(Op, N, Base, Index,
                               SPUFrameLowering::minFrameOffset(),
                               SPUFrameLowering::maxFrameOffset());
}

bool
SPUDAGToDAGISel::DFormAddressPredicate(SDNode *Op, SDValue N, SDValue &Base,
                                      SDValue &Index, int minOffset,
                                      int maxOffset) {
  unsigned Opc = N.getOpcode();
  EVT PtrTy = SPUtli.getPointerTy();

  if (Opc == ISD::FrameIndex) {
    // Stack frame index must be less than 512 (divided by 16):
    FrameIndexSDNode *FIN = cast<FrameIndexSDNode>(N);
    int FI = int(FIN->getIndex());
    DEBUG(errs() << "SelectDFormAddr: ISD::FrameIndex = "
               << FI << "\n");
    if (SPUFrameLowering::FItoStackOffset(FI) < maxOffset) {
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
      ConstantSDNode *CN = cast<ConstantSDNode>(Op1);
      int32_t offset = int32_t(CN->getSExtValue());

      if (Op0.getOpcode() == ISD::FrameIndex) {
        FrameIndexSDNode *FIN = cast<FrameIndexSDNode>(Op0);
        int FI = int(FIN->getIndex());
        DEBUG(errs() << "SelectDFormAddr: ISD::ADD offset = " << offset
                   << " frame index = " << FI << "\n");

        if (SPUFrameLowering::FItoStackOffset(FI) < maxOffset) {
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
      ConstantSDNode *CN = cast<ConstantSDNode>(Op0);
      int32_t offset = int32_t(CN->getSExtValue());

      if (Op1.getOpcode() == ISD::FrameIndex) {
        FrameIndexSDNode *FIN = cast<FrameIndexSDNode>(Op1);
        int FI = int(FIN->getIndex());
        DEBUG(errs() << "SelectDFormAddr: ISD::ADD offset = " << offset
                   << " frame index = " << FI << "\n");

        if (SPUFrameLowering::FItoStackOffset(FI) < maxOffset) {
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
  } else if (Opc == ISD::Register
           ||Opc == ISD::CopyFromReg
           ||Opc == ISD::UNDEF
           ||Opc == ISD::Constant) {
    unsigned OpOpc = Op->getOpcode();

    if (OpOpc == ISD::STORE || OpOpc == ISD::LOAD) {
      // Direct load/store without getelementptr
      SDValue Offs;

      Offs = ((OpOpc == ISD::STORE) ? Op->getOperand(3) : Op->getOperand(2));

      if (Offs.getOpcode() == ISD::Constant || Offs.getOpcode() == ISD::UNDEF) {
        if (Offs.getOpcode() == ISD::UNDEF)
          Offs = CurDAG->getTargetConstant(0, Offs.getValueType());

        Base = Offs;
        Index = N;
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
  \param Op The ISD instruction operand
  \param N The address operand
  \param Base The base pointer operand
  \param Index The offset/index operand

  If the address \a N can be expressed as an A-form or D-form address, returns
  false.  Otherwise, creates two operands, Base and Index that will become the
  (r)(r) X-form address.
*/
bool
SPUDAGToDAGISel::SelectXFormAddr(SDNode *Op, SDValue N, SDValue &Base,
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

/*!
 Utility function to use with COPY_TO_REGCLASS instructions. Returns a SDValue
 to be used as the last parameter of a
CurDAG->getMachineNode(COPY_TO_REGCLASS,..., ) function call
 \param VT the value type for which we want a register class
*/
SDValue SPUDAGToDAGISel::getRC( MVT VT ) {
  switch( VT.SimpleTy ) {
  case MVT::i8:
    return CurDAG->getTargetConstant(SPU::R8CRegClass.getID(), MVT::i32);
  case MVT::i16:
    return CurDAG->getTargetConstant(SPU::R16CRegClass.getID(), MVT::i32);
  case MVT::i32:
    return CurDAG->getTargetConstant(SPU::R32CRegClass.getID(), MVT::i32);
  case MVT::f32:
    return CurDAG->getTargetConstant(SPU::R32FPRegClass.getID(), MVT::i32);
  case MVT::i64:
    return CurDAG->getTargetConstant(SPU::R64CRegClass.getID(), MVT::i32);
  case MVT::i128:
    return CurDAG->getTargetConstant(SPU::GPRCRegClass.getID(), MVT::i32);
  case MVT::v16i8:
  case MVT::v8i16:
  case MVT::v4i32:
  case MVT::v4f32:
  case MVT::v2i64:
  case MVT::v2f64:
    return CurDAG->getTargetConstant(SPU::VECREGRegClass.getID(), MVT::i32);
  default:
    assert( false && "add a new case here" );
    return SDValue();
  }
}

//! Convert the operand from a target-independent to a target-specific node
/*!
 */
SDNode *
SPUDAGToDAGISel::Select(SDNode *N) {
  unsigned Opc = N->getOpcode();
  int n_ops = -1;
  unsigned NewOpc = 0;
  EVT OpVT = N->getValueType(0);
  SDValue Ops[8];
  DebugLoc dl = N->getDebugLoc();

  if (N->isMachineOpcode())
    return NULL;   // Already selected.

  if (Opc == ISD::FrameIndex) {
    int FI = cast<FrameIndexSDNode>(N)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, N->getValueType(0));
    SDValue Imm0 = CurDAG->getTargetConstant(0, N->getValueType(0));

    if (FI < 128) {
      NewOpc = SPU::AIr32;
      Ops[0] = TFI;
      Ops[1] = Imm0;
      n_ops = 2;
    } else {
      NewOpc = SPU::Ar32;
      Ops[0] = CurDAG->getRegister(SPU::R1, N->getValueType(0));
      Ops[1] = SDValue(CurDAG->getMachineNode(SPU::ILAr32, dl,
                                              N->getValueType(0), TFI),
                       0);
      n_ops = 2;
    }
  } else if (Opc == ISD::Constant && OpVT == MVT::i64) {
    // Catch the i64 constants that end up here. Note: The backend doesn't
    // attempt to legalize the constant (it's useless because DAGCombiner
    // will insert 64-bit constants and we can't stop it).
    return SelectI64Constant(N, OpVT, N->getDebugLoc());
  } else if ((Opc == ISD::ZERO_EXTEND || Opc == ISD::ANY_EXTEND)
             && OpVT == MVT::i64) {
    SDValue Op0 = N->getOperand(0);
    EVT Op0VT = Op0.getValueType();
    EVT Op0VecVT = EVT::getVectorVT(*CurDAG->getContext(),
                                    Op0VT, (128 / Op0VT.getSizeInBits()));
    EVT OpVecVT = EVT::getVectorVT(*CurDAG->getContext(),
                                   OpVT, (128 / OpVT.getSizeInBits()));
    SDValue shufMask;

    switch (Op0VT.getSimpleVT().SimpleTy) {
    default:
      report_fatal_error("CellSPU Select: Unhandled zero/any extend EVT");
      /*NOTREACHED*/
    case MVT::i32:
      shufMask = CurDAG->getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x00010203, MVT::i32),
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x08090a0b, MVT::i32));
      break;

    case MVT::i16:
      shufMask = CurDAG->getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x80800203, MVT::i32),
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x80800a0b, MVT::i32));
      break;

    case MVT::i8:
      shufMask = CurDAG->getNode(ISD::BUILD_VECTOR, dl, MVT::v4i32,
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x80808003, MVT::i32),
                                 CurDAG->getConstant(0x80808080, MVT::i32),
                                 CurDAG->getConstant(0x8080800b, MVT::i32));
      break;
    }

    SDNode *shufMaskLoad = emitBuildVector(shufMask.getNode());

    HandleSDNode PromoteScalar(CurDAG->getNode(SPUISD::PREFSLOT2VEC, dl,
                                               Op0VecVT, Op0));

    SDValue PromScalar;
    if (SDNode *N = SelectCode(PromoteScalar.getValue().getNode()))
      PromScalar = SDValue(N, 0);
    else
      PromScalar = PromoteScalar.getValue();

    SDValue zextShuffle =
            CurDAG->getNode(SPUISD::SHUFB, dl, OpVecVT,
                            PromScalar, PromScalar,
                            SDValue(shufMaskLoad, 0));

    HandleSDNode Dummy2(zextShuffle);
    if (SDNode *N = SelectCode(Dummy2.getValue().getNode()))
      zextShuffle = SDValue(N, 0);
    else
      zextShuffle = Dummy2.getValue();
    HandleSDNode Dummy(CurDAG->getNode(SPUISD::VEC2PREFSLOT, dl, OpVT,
                                       zextShuffle));

    CurDAG->ReplaceAllUsesWith(N, Dummy.getValue().getNode());
    SelectCode(Dummy.getValue().getNode());
    return Dummy.getValue().getNode();
  } else if (Opc == ISD::ADD && (OpVT == MVT::i64 || OpVT == MVT::v2i64)) {
    SDNode *CGLoad =
            emitBuildVector(getCarryGenerateShufMask(*CurDAG, dl).getNode());

    HandleSDNode Dummy(CurDAG->getNode(SPUISD::ADD64_MARKER, dl, OpVT,
                                       N->getOperand(0), N->getOperand(1),
                                       SDValue(CGLoad, 0)));

    CurDAG->ReplaceAllUsesWith(N, Dummy.getValue().getNode());
    if (SDNode *N = SelectCode(Dummy.getValue().getNode()))
      return N;
    return Dummy.getValue().getNode();
  } else if (Opc == ISD::SUB && (OpVT == MVT::i64 || OpVT == MVT::v2i64)) {
    SDNode *CGLoad =
            emitBuildVector(getBorrowGenerateShufMask(*CurDAG, dl).getNode());

    HandleSDNode Dummy(CurDAG->getNode(SPUISD::SUB64_MARKER, dl, OpVT,
                                       N->getOperand(0), N->getOperand(1),
                                       SDValue(CGLoad, 0)));

    CurDAG->ReplaceAllUsesWith(N, Dummy.getValue().getNode());
    if (SDNode *N = SelectCode(Dummy.getValue().getNode()))
      return N;
    return Dummy.getValue().getNode();
  } else if (Opc == ISD::MUL && (OpVT == MVT::i64 || OpVT == MVT::v2i64)) {
    SDNode *CGLoad =
            emitBuildVector(getCarryGenerateShufMask(*CurDAG, dl).getNode());

    HandleSDNode Dummy(CurDAG->getNode(SPUISD::MUL64_MARKER, dl, OpVT,
                                       N->getOperand(0), N->getOperand(1),
                                       SDValue(CGLoad, 0)));
    CurDAG->ReplaceAllUsesWith(N, Dummy.getValue().getNode());
    if (SDNode *N = SelectCode(Dummy.getValue().getNode()))
      return N;
    return Dummy.getValue().getNode();
  } else if (Opc == ISD::TRUNCATE) {
    SDValue Op0 = N->getOperand(0);
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
                  CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl, OpVT,
                                         Op0.getOperand(0), getRC(MVT::i32));

          shift_amt -= 32;
          if (shift_amt > 0) {
            // Take care of the additional shift, if present:
            SDValue shift = CurDAG->getTargetConstant(shift_amt, MVT::i32);
            unsigned Opc = SPU::ROTMAIr32_i32;

            if (Op0.getOpcode() == ISD::SRL)
              Opc = SPU::ROTMr32;

            hi32 = CurDAG->getMachineNode(Opc, dl, OpVT, SDValue(hi32, 0),
                                          shift);
          }

          return hi32;
        }
      }
    }
  } else if (Opc == ISD::SHL) {
    if (OpVT == MVT::i64)
      return SelectSHLi64(N, OpVT);
  } else if (Opc == ISD::SRL) {
    if (OpVT == MVT::i64)
      return SelectSRLi64(N, OpVT);
  } else if (Opc == ISD::SRA) {
    if (OpVT == MVT::i64)
      return SelectSRAi64(N, OpVT);
  } else if (Opc == ISD::FNEG
             && (OpVT == MVT::f64 || OpVT == MVT::v2f64)) {
    DebugLoc dl = N->getDebugLoc();
    // Check if the pattern is a special form of DFNMS:
    // (fneg (fsub (fmul R64FP:$rA, R64FP:$rB), R64FP:$rC))
    SDValue Op0 = N->getOperand(0);
    if (Op0.getOpcode() == ISD::FSUB) {
      SDValue Op00 = Op0.getOperand(0);
      if (Op00.getOpcode() == ISD::FMUL) {
        unsigned Opc = SPU::DFNMSf64;
        if (OpVT == MVT::v2f64)
          Opc = SPU::DFNMSv2f64;

        return CurDAG->getMachineNode(Opc, dl, OpVT,
                                      Op00.getOperand(0),
                                      Op00.getOperand(1),
                                      Op0.getOperand(1));
      }
    }

    SDValue negConst = CurDAG->getConstant(0x8000000000000000ULL, MVT::i64);
    SDNode *signMask = 0;
    unsigned Opc = SPU::XORfneg64;

    if (OpVT == MVT::f64) {
      signMask = SelectI64Constant(negConst.getNode(), MVT::i64, dl);
    } else if (OpVT == MVT::v2f64) {
      Opc = SPU::XORfnegvec;
      signMask = emitBuildVector(CurDAG->getNode(ISD::BUILD_VECTOR, dl,
                                                 MVT::v2i64,
                                                 negConst, negConst).getNode());
    }

    return CurDAG->getMachineNode(Opc, dl, OpVT,
                                  N->getOperand(0), SDValue(signMask, 0));
  } else if (Opc == ISD::FABS) {
    if (OpVT == MVT::f64) {
      SDNode *signMask = SelectI64Constant(0x7fffffffffffffffULL, MVT::i64, dl);
      return CurDAG->getMachineNode(SPU::ANDfabs64, dl, OpVT,
                                    N->getOperand(0), SDValue(signMask, 0));
    } else if (OpVT == MVT::v2f64) {
      SDValue absConst = CurDAG->getConstant(0x7fffffffffffffffULL, MVT::i64);
      SDValue absVec = CurDAG->getNode(ISD::BUILD_VECTOR, dl, MVT::v2i64,
                                       absConst, absConst);
      SDNode *signMask = emitBuildVector(absVec.getNode());
      return CurDAG->getMachineNode(SPU::ANDfabsvec, dl, OpVT,
                                    N->getOperand(0), SDValue(signMask, 0));
    }
  } else if (Opc == SPUISD::LDRESULT) {
    // Custom select instructions for LDRESULT
    EVT VT = N->getValueType(0);
    SDValue Arg = N->getOperand(0);
    SDValue Chain = N->getOperand(1);
    SDNode *Result;

    Result = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl, VT,
                                    MVT::Other, Arg,
                                    getRC( VT.getSimpleVT()), Chain);
    return Result;

  } else if (Opc == SPUISD::IndirectAddr) {
    // Look at the operands: SelectCode() will catch the cases that aren't
    // specifically handled here.
    //
    // SPUInstrInfo catches the following patterns:
    // (SPUindirect (SPUhi ...), (SPUlo ...))
    // (SPUindirect $sp, imm)
    EVT VT = N->getValueType(0);
    SDValue Op0 = N->getOperand(0);
    SDValue Op1 = N->getOperand(1);
    RegisterSDNode *RN;

    if ((Op0.getOpcode() != SPUISD::Hi && Op1.getOpcode() != SPUISD::Lo)
        || (Op0.getOpcode() == ISD::Register
            && ((RN = dyn_cast<RegisterSDNode>(Op0.getNode())) != 0
                && RN->getReg() != SPU::R1))) {
      NewOpc = SPU::Ar32;
      Ops[1] = Op1;
      if (Op1.getOpcode() == ISD::Constant) {
        ConstantSDNode *CN = cast<ConstantSDNode>(Op1);
        Op1 = CurDAG->getTargetConstant(CN->getSExtValue(), VT);
        if (isInt<10>(CN->getSExtValue())) {
          NewOpc = SPU::AIr32;
          Ops[1] = Op1;
        } else {
          Ops[1] = SDValue(CurDAG->getMachineNode(SPU::ILr32, dl,
                                                  N->getValueType(0),
                                                  Op1),
                           0);
        }
      }
      Ops[0] = Op0;
      n_ops = 2;
    }
  }

  if (n_ops > 0) {
    if (N->hasOneUse())
      return CurDAG->SelectNodeTo(N, NewOpc, OpVT, Ops, n_ops);
    else
      return CurDAG->getMachineNode(NewOpc, dl, OpVT, Ops, n_ops);
  } else
    return SelectCode(N);
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
SPUDAGToDAGISel::SelectSHLi64(SDNode *N, EVT OpVT) {
  SDValue Op0 = N->getOperand(0);
  EVT VecVT = EVT::getVectorVT(*CurDAG->getContext(),
                               OpVT, (128 / OpVT.getSizeInBits()));
  SDValue ShiftAmt = N->getOperand(1);
  EVT ShiftAmtVT = ShiftAmt.getValueType();
  SDNode *VecOp0, *SelMask, *ZeroFill, *Shift = 0;
  SDValue SelMaskVal;
  DebugLoc dl = N->getDebugLoc();

  VecOp0 = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl, VecVT,
                                  Op0, getRC(MVT::v2i64) );
  SelMaskVal = CurDAG->getTargetConstant(0xff00ULL, MVT::i16);
  SelMask = CurDAG->getMachineNode(SPU::FSMBIv2i64, dl, VecVT, SelMaskVal);
  ZeroFill = CurDAG->getMachineNode(SPU::ILv2i64, dl, VecVT,
                                    CurDAG->getTargetConstant(0, OpVT));
  VecOp0 = CurDAG->getMachineNode(SPU::SELBv2i64, dl, VecVT,
                                  SDValue(ZeroFill, 0),
                                  SDValue(VecOp0, 0),
                                  SDValue(SelMask, 0));

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(ShiftAmt)) {
    unsigned bytes = unsigned(CN->getZExtValue()) >> 3;
    unsigned bits = unsigned(CN->getZExtValue()) & 7;

    if (bytes > 0) {
      Shift =
        CurDAG->getMachineNode(SPU::SHLQBYIv2i64, dl, VecVT,
                               SDValue(VecOp0, 0),
                               CurDAG->getTargetConstant(bytes, ShiftAmtVT));
    }

    if (bits > 0) {
      Shift =
        CurDAG->getMachineNode(SPU::SHLQBIIv2i64, dl, VecVT,
                               SDValue((Shift != 0 ? Shift : VecOp0), 0),
                               CurDAG->getTargetConstant(bits, ShiftAmtVT));
    }
  } else {
    SDNode *Bytes =
      CurDAG->getMachineNode(SPU::ROTMIr32, dl, ShiftAmtVT,
                             ShiftAmt,
                             CurDAG->getTargetConstant(3, ShiftAmtVT));
    SDNode *Bits =
      CurDAG->getMachineNode(SPU::ANDIr32, dl, ShiftAmtVT,
                             ShiftAmt,
                             CurDAG->getTargetConstant(7, ShiftAmtVT));
    Shift =
      CurDAG->getMachineNode(SPU::SHLQBYv2i64, dl, VecVT,
                             SDValue(VecOp0, 0), SDValue(Bytes, 0));
    Shift =
      CurDAG->getMachineNode(SPU::SHLQBIv2i64, dl, VecVT,
                             SDValue(Shift, 0), SDValue(Bits, 0));
  }

  return CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl,
                                OpVT, SDValue(Shift, 0), getRC(MVT::i64));
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
SPUDAGToDAGISel::SelectSRLi64(SDNode *N, EVT OpVT) {
  SDValue Op0 = N->getOperand(0);
  EVT VecVT = EVT::getVectorVT(*CurDAG->getContext(),
                               OpVT, (128 / OpVT.getSizeInBits()));
  SDValue ShiftAmt = N->getOperand(1);
  EVT ShiftAmtVT = ShiftAmt.getValueType();
  SDNode *VecOp0, *Shift = 0;
  DebugLoc dl = N->getDebugLoc();

  VecOp0 = CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl, VecVT,
                                  Op0, getRC(MVT::v2i64) );

  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(ShiftAmt)) {
    unsigned bytes = unsigned(CN->getZExtValue()) >> 3;
    unsigned bits = unsigned(CN->getZExtValue()) & 7;

    if (bytes > 0) {
      Shift =
        CurDAG->getMachineNode(SPU::ROTQMBYIv2i64, dl, VecVT,
                               SDValue(VecOp0, 0),
                               CurDAG->getTargetConstant(bytes, ShiftAmtVT));
    }

    if (bits > 0) {
      Shift =
        CurDAG->getMachineNode(SPU::ROTQMBIIv2i64, dl, VecVT,
                               SDValue((Shift != 0 ? Shift : VecOp0), 0),
                               CurDAG->getTargetConstant(bits, ShiftAmtVT));
    }
  } else {
    SDNode *Bytes =
      CurDAG->getMachineNode(SPU::ROTMIr32, dl, ShiftAmtVT,
                             ShiftAmt,
                             CurDAG->getTargetConstant(3, ShiftAmtVT));
    SDNode *Bits =
      CurDAG->getMachineNode(SPU::ANDIr32, dl, ShiftAmtVT,
                             ShiftAmt,
                             CurDAG->getTargetConstant(7, ShiftAmtVT));

    // Ensure that the shift amounts are negated!
    Bytes = CurDAG->getMachineNode(SPU::SFIr32, dl, ShiftAmtVT,
                                   SDValue(Bytes, 0),
                                   CurDAG->getTargetConstant(0, ShiftAmtVT));

    Bits = CurDAG->getMachineNode(SPU::SFIr32, dl, ShiftAmtVT,
                                  SDValue(Bits, 0),
                                  CurDAG->getTargetConstant(0, ShiftAmtVT));

    Shift =
      CurDAG->getMachineNode(SPU::ROTQMBYv2i64, dl, VecVT,
                             SDValue(VecOp0, 0), SDValue(Bytes, 0));
    Shift =
      CurDAG->getMachineNode(SPU::ROTQMBIv2i64, dl, VecVT,
                             SDValue(Shift, 0), SDValue(Bits, 0));
  }

  return CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl,
                                OpVT, SDValue(Shift, 0), getRC(MVT::i64));
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
SPUDAGToDAGISel::SelectSRAi64(SDNode *N, EVT OpVT) {
  // Promote Op0 to vector
  EVT VecVT = EVT::getVectorVT(*CurDAG->getContext(),
                               OpVT, (128 / OpVT.getSizeInBits()));
  SDValue ShiftAmt = N->getOperand(1);
  EVT ShiftAmtVT = ShiftAmt.getValueType();
  DebugLoc dl = N->getDebugLoc();

  SDNode *VecOp0 =
    CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl,
                           VecVT, N->getOperand(0), getRC(MVT::v2i64));

  SDValue SignRotAmt = CurDAG->getTargetConstant(31, ShiftAmtVT);
  SDNode *SignRot =
    CurDAG->getMachineNode(SPU::ROTMAIv2i64_i32, dl, MVT::v2i64,
                           SDValue(VecOp0, 0), SignRotAmt);
  SDNode *UpperHalfSign =
    CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl,
                           MVT::i32, SDValue(SignRot, 0), getRC(MVT::i32));

  SDNode *UpperHalfSignMask =
    CurDAG->getMachineNode(SPU::FSM64r32, dl, VecVT, SDValue(UpperHalfSign, 0));
  SDNode *UpperLowerMask =
    CurDAG->getMachineNode(SPU::FSMBIv2i64, dl, VecVT,
                           CurDAG->getTargetConstant(0xff00ULL, MVT::i16));
  SDNode *UpperLowerSelect =
    CurDAG->getMachineNode(SPU::SELBv2i64, dl, VecVT,
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
        CurDAG->getMachineNode(SPU::ROTQBYIv2i64, dl, VecVT,
                               SDValue(UpperLowerSelect, 0),
                               CurDAG->getTargetConstant(bytes, ShiftAmtVT));
    }

    if (bits > 0) {
      bits = 8 - bits;
      Shift =
        CurDAG->getMachineNode(SPU::ROTQBIIv2i64, dl, VecVT,
                               SDValue((Shift != 0 ? Shift : UpperLowerSelect), 0),
                               CurDAG->getTargetConstant(bits, ShiftAmtVT));
    }
  } else {
    SDNode *NegShift =
      CurDAG->getMachineNode(SPU::SFIr32, dl, ShiftAmtVT,
                             ShiftAmt, CurDAG->getTargetConstant(0, ShiftAmtVT));

    Shift =
      CurDAG->getMachineNode(SPU::ROTQBYBIv2i64_r32, dl, VecVT,
                             SDValue(UpperLowerSelect, 0), SDValue(NegShift, 0));
    Shift =
      CurDAG->getMachineNode(SPU::ROTQBIv2i64, dl, VecVT,
                             SDValue(Shift, 0), SDValue(NegShift, 0));
  }

  return CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl,
                                OpVT, SDValue(Shift, 0), getRC(MVT::i64));
}

/*!
 Do the necessary magic necessary to load a i64 constant
 */
SDNode *SPUDAGToDAGISel::SelectI64Constant(SDNode *N, EVT OpVT,
                                           DebugLoc dl) {
  ConstantSDNode *CN = cast<ConstantSDNode>(N);
  return SelectI64Constant(CN->getZExtValue(), OpVT, dl);
}

SDNode *SPUDAGToDAGISel::SelectI64Constant(uint64_t Value64, EVT OpVT,
                                           DebugLoc dl) {
  EVT OpVecVT = EVT::getVectorVT(*CurDAG->getContext(), OpVT, 2);
  SDValue i64vec =
          SPU::LowerV2I64Splat(OpVecVT, *CurDAG, Value64, dl);

  // Here's where it gets interesting, because we have to parse out the
  // subtree handed back in i64vec:

  if (i64vec.getOpcode() == ISD::BITCAST) {
    // The degenerate case where the upper and lower bits in the splat are
    // identical:
    SDValue Op0 = i64vec.getOperand(0);

    ReplaceUses(i64vec, Op0);
    return CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl, OpVT,
                                  SDValue(emitBuildVector(Op0.getNode()), 0),
                                  getRC(MVT::i64));
  } else if (i64vec.getOpcode() == SPUISD::SHUFB) {
    SDValue lhs = i64vec.getOperand(0);
    SDValue rhs = i64vec.getOperand(1);
    SDValue shufmask = i64vec.getOperand(2);

    if (lhs.getOpcode() == ISD::BITCAST) {
      ReplaceUses(lhs, lhs.getOperand(0));
      lhs = lhs.getOperand(0);
    }

    SDNode *lhsNode = (lhs.getNode()->isMachineOpcode()
                       ? lhs.getNode()
                       : emitBuildVector(lhs.getNode()));

    if (rhs.getOpcode() == ISD::BITCAST) {
      ReplaceUses(rhs, rhs.getOperand(0));
      rhs = rhs.getOperand(0);
    }

    SDNode *rhsNode = (rhs.getNode()->isMachineOpcode()
                       ? rhs.getNode()
                       : emitBuildVector(rhs.getNode()));

    if (shufmask.getOpcode() == ISD::BITCAST) {
      ReplaceUses(shufmask, shufmask.getOperand(0));
      shufmask = shufmask.getOperand(0);
    }

    SDNode *shufMaskNode = (shufmask.getNode()->isMachineOpcode()
                            ? shufmask.getNode()
                            : emitBuildVector(shufmask.getNode()));

   SDValue shufNode =
            CurDAG->getNode(SPUISD::SHUFB, dl, OpVecVT,
                                   SDValue(lhsNode, 0), SDValue(rhsNode, 0),
                                   SDValue(shufMaskNode, 0));
    HandleSDNode Dummy(shufNode);
    SDNode *SN = SelectCode(Dummy.getValue().getNode());
    if (SN == 0) SN = Dummy.getValue().getNode();

    return CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl,
                                  OpVT, SDValue(SN, 0), getRC(MVT::i64));
  } else if (i64vec.getOpcode() == ISD::BUILD_VECTOR) {
    return CurDAG->getMachineNode(TargetOpcode::COPY_TO_REGCLASS, dl, OpVT,
                                  SDValue(emitBuildVector(i64vec.getNode()), 0),
                                  getRC(MVT::i64));
  } else {
    report_fatal_error("SPUDAGToDAGISel::SelectI64Constant: Unhandled i64vec"
                      "condition");
  }
}

/// createSPUISelDag - This pass converts a legalized DAG into a
/// SPU-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSPUISelDag(SPUTargetMachine &TM) {
  return new SPUDAGToDAGISel(TM);
}
