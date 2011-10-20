//===-- llvm/Target/TargetLowering.h - Target Lowering Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes how to lower LLVM code to machine code.  This has two
// main components:
//
//  1. Which ValueTypes are natively supported by the target.
//  2. Which operations are supported for supported ValueTypes.
//  3. Cost thresholds for alternative implementations of certain operations.
//
// In addition it has a few other components, like information about FP
// immediates.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETLOWERING_H
#define LLVM_TARGET_TARGETLOWERING_H

#include "llvm/CallingConv.h"
#include "llvm/InlineAsm.h"
#include "llvm/Attributes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Target/TargetCallingConv.h"
#include "llvm/Target/TargetMachine.h"
#include <climits>
#include <map>
#include <vector>

namespace llvm {
  class AllocaInst;
  class APFloat;
  class CallInst;
  class CCState;
  class Function;
  class FastISel;
  class FunctionLoweringInfo;
  class ImmutableCallSite;
  class MachineBasicBlock;
  class MachineFunction;
  class MachineFrameInfo;
  class MachineInstr;
  class MachineJumpTableInfo;
  class MCContext;
  class MCExpr;
  class SDNode;
  class SDValue;
  class SelectionDAG;
  template<typename T> class SmallVectorImpl;
  class TargetData;
  class TargetMachine;
  class TargetRegisterClass;
  class TargetLoweringObjectFile;
  class Value;

  // FIXME: should this be here?
  namespace TLSModel {
    enum Model {
      GeneralDynamic,
      LocalDynamic,
      InitialExec,
      LocalExec
    };
  }
  TLSModel::Model getTLSModel(const GlobalValue *GV, Reloc::Model reloc);


//===----------------------------------------------------------------------===//
/// TargetLowering - This class defines information used to lower LLVM code to
/// legal SelectionDAG operators that the target instruction selector can accept
/// natively.
///
/// This class also defines callbacks that targets must implement to lower
/// target-specific constructs to SelectionDAG operators.
///
class TargetLowering {
  TargetLowering(const TargetLowering&);  // DO NOT IMPLEMENT
  void operator=(const TargetLowering&);  // DO NOT IMPLEMENT
public:
  /// LegalizeAction - This enum indicates whether operations are valid for a
  /// target, and if not, what action should be used to make them valid.
  enum LegalizeAction {
    Legal,      // The target natively supports this operation.
    Promote,    // This operation should be executed in a larger type.
    Expand,     // Try to expand this to other ops, otherwise use a libcall.
    Custom      // Use the LowerOperation hook to implement custom lowering.
  };

  /// LegalizeAction - This enum indicates whether a types are legal for a
  /// target, and if not, what action should be used to make them valid.
  enum LegalizeTypeAction {
    TypeLegal,           // The target natively supports this type.
    TypePromoteInteger,  // Replace this integer with a larger one.
    TypeExpandInteger,   // Split this integer into two of half the size.
    TypeSoftenFloat,     // Convert this float to a same size integer type.
    TypeExpandFloat,     // Split this float into two of half the size.
    TypeScalarizeVector, // Replace this one-element vector with its element.
    TypeSplitVector,     // Split this vector into two of half the size.
    TypeWidenVector      // This vector should be widened into a larger vector.
  };

  enum BooleanContent { // How the target represents true/false values.
    UndefinedBooleanContent,    // Only bit 0 counts, the rest can hold garbage.
    ZeroOrOneBooleanContent,        // All bits zero except for bit 0.
    ZeroOrNegativeOneBooleanContent // All bits equal to bit 0.
  };

  static ISD::NodeType getExtendForContent(BooleanContent Content) {
    switch (Content) {
    default:
      assert(false && "Unknown BooleanContent!");
    case UndefinedBooleanContent:
      // Extend by adding rubbish bits.
      return ISD::ANY_EXTEND;
    case ZeroOrOneBooleanContent:
      // Extend by adding zero bits.
      return ISD::ZERO_EXTEND;
    case ZeroOrNegativeOneBooleanContent:
      // Extend by copying the sign bit.
      return ISD::SIGN_EXTEND;
    }
  }

  /// NOTE: The constructor takes ownership of TLOF.
  explicit TargetLowering(const TargetMachine &TM,
                          const TargetLoweringObjectFile *TLOF);
  virtual ~TargetLowering();

  const TargetMachine &getTargetMachine() const { return TM; }
  const TargetData *getTargetData() const { return TD; }
  const TargetLoweringObjectFile &getObjFileLowering() const { return TLOF; }

  bool isBigEndian() const { return !IsLittleEndian; }
  bool isLittleEndian() const { return IsLittleEndian; }
  MVT getPointerTy() const { return PointerTy; }
  virtual MVT getShiftAmountTy(EVT LHSTy) const;

  /// isSelectExpensive - Return true if the select operation is expensive for
  /// this target.
  bool isSelectExpensive() const { return SelectIsExpensive; }

  /// isIntDivCheap() - Return true if integer divide is usually cheaper than
  /// a sequence of several shifts, adds, and multiplies for this target.
  bool isIntDivCheap() const { return IntDivIsCheap; }

  /// isPow2DivCheap() - Return true if pow2 div is cheaper than a chain of
  /// srl/add/sra.
  bool isPow2DivCheap() const { return Pow2DivIsCheap; }

  /// isJumpExpensive() - Return true if Flow Control is an expensive operation
  /// that should be avoided.
  bool isJumpExpensive() const { return JumpIsExpensive; }

  /// getSetCCResultType - Return the ValueType of the result of SETCC
  /// operations.  Also used to obtain the target's preferred type for
  /// the condition operand of SELECT and BRCOND nodes.  In the case of
  /// BRCOND the argument passed is MVT::Other since there are no other
  /// operands to get a type hint from.
  virtual EVT getSetCCResultType(EVT VT) const;

  /// getCmpLibcallReturnType - Return the ValueType for comparison
  /// libcalls. Comparions libcalls include floating point comparion calls,
  /// and Ordered/Unordered check calls on floating point numbers.
  virtual
  MVT::SimpleValueType getCmpLibcallReturnType() const;

  /// getBooleanContents - For targets without i1 registers, this gives the
  /// nature of the high-bits of boolean values held in types wider than i1.
  /// "Boolean values" are special true/false values produced by nodes like
  /// SETCC and consumed (as the condition) by nodes like SELECT and BRCOND.
  /// Not to be confused with general values promoted from i1.
  /// Some cpus distinguish between vectors of boolean and scalars; the isVec
  /// parameter selects between the two kinds.  For example on X86 a scalar
  /// boolean should be zero extended from i1, while the elements of a vector
  /// of booleans should be sign extended from i1.
  BooleanContent getBooleanContents(bool isVec) const {
    return isVec ? BooleanVectorContents : BooleanContents;
  }

  /// getSchedulingPreference - Return target scheduling preference.
  Sched::Preference getSchedulingPreference() const {
    return SchedPreferenceInfo;
  }

  /// getSchedulingPreference - Some scheduler, e.g. hybrid, can switch to
  /// different scheduling heuristics for different nodes. This function returns
  /// the preference (or none) for the given node.
  virtual Sched::Preference getSchedulingPreference(SDNode *) const {
    return Sched::None;
  }

  /// getRegClassFor - Return the register class that should be used for the
  /// specified value type.
  virtual TargetRegisterClass *getRegClassFor(EVT VT) const {
    assert(VT.isSimple() && "getRegClassFor called on illegal type!");
    TargetRegisterClass *RC = RegClassForVT[VT.getSimpleVT().SimpleTy];
    assert(RC && "This value type is not natively supported!");
    return RC;
  }

  /// getRepRegClassFor - Return the 'representative' register class for the
  /// specified value type. The 'representative' register class is the largest
  /// legal super-reg register class for the register class of the value type.
  /// For example, on i386 the rep register class for i8, i16, and i32 are GR32;
  /// while the rep register class is GR64 on x86_64.
  virtual const TargetRegisterClass *getRepRegClassFor(EVT VT) const {
    assert(VT.isSimple() && "getRepRegClassFor called on illegal type!");
    const TargetRegisterClass *RC = RepRegClassForVT[VT.getSimpleVT().SimpleTy];
    return RC;
  }

  /// getRepRegClassCostFor - Return the cost of the 'representative' register
  /// class for the specified value type.
  virtual uint8_t getRepRegClassCostFor(EVT VT) const {
    assert(VT.isSimple() && "getRepRegClassCostFor called on illegal type!");
    return RepRegClassCostForVT[VT.getSimpleVT().SimpleTy];
  }

  /// isTypeLegal - Return true if the target has native support for the
  /// specified value type.  This means that it has a register that directly
  /// holds it without promotions or expansions.
  bool isTypeLegal(EVT VT) const {
    assert(!VT.isSimple() ||
           (unsigned)VT.getSimpleVT().SimpleTy < array_lengthof(RegClassForVT));
    return VT.isSimple() && RegClassForVT[VT.getSimpleVT().SimpleTy] != 0;
  }

  class ValueTypeActionImpl {
    /// ValueTypeActions - For each value type, keep a LegalizeTypeAction enum
    /// that indicates how instruction selection should deal with the type.
    uint8_t ValueTypeActions[MVT::LAST_VALUETYPE];

  public:
    ValueTypeActionImpl() {
      std::fill(ValueTypeActions, array_endof(ValueTypeActions), 0);
    }

    LegalizeTypeAction getTypeAction(MVT VT) const {
      return (LegalizeTypeAction)ValueTypeActions[VT.SimpleTy];
    }

    void setTypeAction(EVT VT, LegalizeTypeAction Action) {
      unsigned I = VT.getSimpleVT().SimpleTy;
      ValueTypeActions[I] = Action;
    }
  };

  const ValueTypeActionImpl &getValueTypeActions() const {
    return ValueTypeActions;
  }

  /// getTypeAction - Return how we should legalize values of this type, either
  /// it is already legal (return 'Legal') or we need to promote it to a larger
  /// type (return 'Promote'), or we need to expand it into multiple registers
  /// of smaller integer type (return 'Expand').  'Custom' is not an option.
  LegalizeTypeAction getTypeAction(LLVMContext &Context, EVT VT) const {
    return getTypeConversion(Context, VT).first;
  }
  LegalizeTypeAction getTypeAction(MVT VT) const {
    return ValueTypeActions.getTypeAction(VT);
  }

  /// getTypeToTransformTo - For types supported by the target, this is an
  /// identity function.  For types that must be promoted to larger types, this
  /// returns the larger type to promote to.  For integer types that are larger
  /// than the largest integer register, this contains one step in the expansion
  /// to get to the smaller register. For illegal floating point types, this
  /// returns the integer type to transform to.
  EVT getTypeToTransformTo(LLVMContext &Context, EVT VT) const {
    return getTypeConversion(Context, VT).second;
  }

  /// getTypeToExpandTo - For types supported by the target, this is an
  /// identity function.  For types that must be expanded (i.e. integer types
  /// that are larger than the largest integer register or illegal floating
  /// point types), this returns the largest legal type it will be expanded to.
  EVT getTypeToExpandTo(LLVMContext &Context, EVT VT) const {
    assert(!VT.isVector());
    while (true) {
      switch (getTypeAction(Context, VT)) {
      case TypeLegal:
        return VT;
      case TypeExpandInteger:
        VT = getTypeToTransformTo(Context, VT);
        break;
      default:
        assert(false && "Type is not legal nor is it to be expanded!");
        return VT;
      }
    }
    return VT;
  }

  /// getVectorTypeBreakdown - Vector types are broken down into some number of
  /// legal first class types.  For example, EVT::v8f32 maps to 2 EVT::v4f32
  /// with Altivec or SSE1, or 8 promoted EVT::f64 values with the X86 FP stack.
  /// Similarly, EVT::v2i64 turns into 4 EVT::i32 values with both PPC and X86.
  ///
  /// This method returns the number of registers needed, and the VT for each
  /// register.  It also returns the VT and quantity of the intermediate values
  /// before they are promoted/expanded.
  ///
  unsigned getVectorTypeBreakdown(LLVMContext &Context, EVT VT,
                                  EVT &IntermediateVT,
                                  unsigned &NumIntermediates,
                                  EVT &RegisterVT) const;

  /// getTgtMemIntrinsic: Given an intrinsic, checks if on the target the
  /// intrinsic will need to map to a MemIntrinsicNode (touches memory). If
  /// this is the case, it returns true and store the intrinsic
  /// information into the IntrinsicInfo that was passed to the function.
  struct IntrinsicInfo {
    unsigned     opc;         // target opcode
    EVT          memVT;       // memory VT
    const Value* ptrVal;      // value representing memory location
    int          offset;      // offset off of ptrVal
    unsigned     align;       // alignment
    bool         vol;         // is volatile?
    bool         readMem;     // reads memory?
    bool         writeMem;    // writes memory?
  };

  virtual bool getTgtMemIntrinsic(IntrinsicInfo &, const CallInst &,
                                  unsigned /*Intrinsic*/) const {
    return false;
  }

  /// isFPImmLegal - Returns true if the target can instruction select the
  /// specified FP immediate natively. If false, the legalizer will materialize
  /// the FP immediate as a load from a constant pool.
  virtual bool isFPImmLegal(const APFloat &/*Imm*/, EVT /*VT*/) const {
    return false;
  }

  /// isShuffleMaskLegal - Targets can use this to indicate that they only
  /// support *some* VECTOR_SHUFFLE operations, those with specific masks.
  /// By default, if a target supports the VECTOR_SHUFFLE node, all mask values
  /// are assumed to be legal.
  virtual bool isShuffleMaskLegal(const SmallVectorImpl<int> &/*Mask*/,
                                  EVT /*VT*/) const {
    return true;
  }

  /// canOpTrap - Returns true if the operation can trap for the value type.
  /// VT must be a legal type. By default, we optimistically assume most
  /// operations don't trap except for divide and remainder.
  virtual bool canOpTrap(unsigned Op, EVT VT) const;

  /// isVectorClearMaskLegal - Similar to isShuffleMaskLegal. This is
  /// used by Targets can use this to indicate if there is a suitable
  /// VECTOR_SHUFFLE that can be used to replace a VAND with a constant
  /// pool entry.
  virtual bool isVectorClearMaskLegal(const SmallVectorImpl<int> &/*Mask*/,
                                      EVT /*VT*/) const {
    return false;
  }

  /// getOperationAction - Return how this operation should be treated: either
  /// it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction getOperationAction(unsigned Op, EVT VT) const {
    if (VT.isExtended()) return Expand;
    assert(Op < array_lengthof(OpActions[0]) && "Table isn't big enough!");
    unsigned I = (unsigned) VT.getSimpleVT().SimpleTy;
    return (LegalizeAction)OpActions[I][Op];
  }

  /// isOperationLegalOrCustom - Return true if the specified operation is
  /// legal on this target or can be made legal with custom lowering. This
  /// is used to help guide high-level lowering decisions.
  bool isOperationLegalOrCustom(unsigned Op, EVT VT) const {
    return (VT == MVT::Other || isTypeLegal(VT)) &&
      (getOperationAction(Op, VT) == Legal ||
       getOperationAction(Op, VT) == Custom);
  }

  /// isOperationLegal - Return true if the specified operation is legal on this
  /// target.
  bool isOperationLegal(unsigned Op, EVT VT) const {
    return (VT == MVT::Other || isTypeLegal(VT)) &&
           getOperationAction(Op, VT) == Legal;
  }

  /// getLoadExtAction - Return how this load with extension should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction getLoadExtAction(unsigned ExtType, EVT VT) const {
    assert(ExtType < ISD::LAST_LOADEXT_TYPE &&
           VT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    return (LegalizeAction)LoadExtActions[VT.getSimpleVT().SimpleTy][ExtType];
  }

  /// isLoadExtLegal - Return true if the specified load with extension is legal
  /// on this target.
  bool isLoadExtLegal(unsigned ExtType, EVT VT) const {
    return VT.isSimple() && getLoadExtAction(ExtType, VT) == Legal;
  }

  /// getTruncStoreAction - Return how this store with truncation should be
  /// treated: either it is legal, needs to be promoted to a larger size, needs
  /// to be expanded to some other code sequence, or the target has a custom
  /// expander for it.
  LegalizeAction getTruncStoreAction(EVT ValVT, EVT MemVT) const {
    assert(ValVT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           MemVT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    return (LegalizeAction)TruncStoreActions[ValVT.getSimpleVT().SimpleTy]
                                            [MemVT.getSimpleVT().SimpleTy];
  }

  /// isTruncStoreLegal - Return true if the specified store with truncation is
  /// legal on this target.
  bool isTruncStoreLegal(EVT ValVT, EVT MemVT) const {
    return isTypeLegal(ValVT) && MemVT.isSimple() &&
           getTruncStoreAction(ValVT, MemVT) == Legal;
  }

  /// getIndexedLoadAction - Return how the indexed load should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction
  getIndexedLoadAction(unsigned IdxMode, EVT VT) const {
    assert(IdxMode < ISD::LAST_INDEXED_MODE &&
           VT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    unsigned Ty = (unsigned)VT.getSimpleVT().SimpleTy;
    return (LegalizeAction)((IndexedModeActions[Ty][IdxMode] & 0xf0) >> 4);
  }

  /// isIndexedLoadLegal - Return true if the specified indexed load is legal
  /// on this target.
  bool isIndexedLoadLegal(unsigned IdxMode, EVT VT) const {
    return VT.isSimple() &&
      (getIndexedLoadAction(IdxMode, VT) == Legal ||
       getIndexedLoadAction(IdxMode, VT) == Custom);
  }

  /// getIndexedStoreAction - Return how the indexed store should be treated:
  /// either it is legal, needs to be promoted to a larger size, needs to be
  /// expanded to some other code sequence, or the target has a custom expander
  /// for it.
  LegalizeAction
  getIndexedStoreAction(unsigned IdxMode, EVT VT) const {
    assert(IdxMode < ISD::LAST_INDEXED_MODE &&
           VT.getSimpleVT() < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    unsigned Ty = (unsigned)VT.getSimpleVT().SimpleTy;
    return (LegalizeAction)(IndexedModeActions[Ty][IdxMode] & 0x0f);
  }

  /// isIndexedStoreLegal - Return true if the specified indexed load is legal
  /// on this target.
  bool isIndexedStoreLegal(unsigned IdxMode, EVT VT) const {
    return VT.isSimple() &&
      (getIndexedStoreAction(IdxMode, VT) == Legal ||
       getIndexedStoreAction(IdxMode, VT) == Custom);
  }

  /// getCondCodeAction - Return how the condition code should be treated:
  /// either it is legal, needs to be expanded to some other code sequence,
  /// or the target has a custom expander for it.
  LegalizeAction
  getCondCodeAction(ISD::CondCode CC, EVT VT) const {
    assert((unsigned)CC < array_lengthof(CondCodeActions) &&
           (unsigned)VT.getSimpleVT().SimpleTy < sizeof(CondCodeActions[0])*4 &&
           "Table isn't big enough!");
    LegalizeAction Action = (LegalizeAction)
      ((CondCodeActions[CC] >> (2*VT.getSimpleVT().SimpleTy)) & 3);
    assert(Action != Promote && "Can't promote condition code!");
    return Action;
  }

  /// isCondCodeLegal - Return true if the specified condition code is legal
  /// on this target.
  bool isCondCodeLegal(ISD::CondCode CC, EVT VT) const {
    return getCondCodeAction(CC, VT) == Legal ||
           getCondCodeAction(CC, VT) == Custom;
  }


  /// getTypeToPromoteTo - If the action for this operation is to promote, this
  /// method returns the ValueType to promote to.
  EVT getTypeToPromoteTo(unsigned Op, EVT VT) const {
    assert(getOperationAction(Op, VT) == Promote &&
           "This operation isn't promoted!");

    // See if this has an explicit type specified.
    std::map<std::pair<unsigned, MVT::SimpleValueType>,
             MVT::SimpleValueType>::const_iterator PTTI =
      PromoteToType.find(std::make_pair(Op, VT.getSimpleVT().SimpleTy));
    if (PTTI != PromoteToType.end()) return PTTI->second;

    assert((VT.isInteger() || VT.isFloatingPoint()) &&
           "Cannot autopromote this type, add it with AddPromotedToType.");

    EVT NVT = VT;
    do {
      NVT = (MVT::SimpleValueType)(NVT.getSimpleVT().SimpleTy+1);
      assert(NVT.isInteger() == VT.isInteger() && NVT != MVT::isVoid &&
             "Didn't find type to promote to!");
    } while (!isTypeLegal(NVT) ||
              getOperationAction(Op, NVT) == Promote);
    return NVT;
  }

  /// getValueType - Return the EVT corresponding to this LLVM type.
  /// This is fixed by the LLVM operations except for the pointer size.  If
  /// AllowUnknown is true, this will return MVT::Other for types with no EVT
  /// counterpart (e.g. structs), otherwise it will assert.
  EVT getValueType(Type *Ty, bool AllowUnknown = false) const {
    EVT VT = EVT::getEVT(Ty, AllowUnknown);
    return VT == MVT::iPTR ? PointerTy : VT;
  }

  /// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
  /// function arguments in the caller parameter area.  This is the actual
  /// alignment, not its logarithm.
  virtual unsigned getByValTypeAlignment(Type *Ty) const;

  /// getRegisterType - Return the type of registers that this ValueType will
  /// eventually require.
  EVT getRegisterType(MVT VT) const {
    assert((unsigned)VT.SimpleTy < array_lengthof(RegisterTypeForVT));
    return RegisterTypeForVT[VT.SimpleTy];
  }

  /// getRegisterType - Return the type of registers that this ValueType will
  /// eventually require.
  EVT getRegisterType(LLVMContext &Context, EVT VT) const {
    if (VT.isSimple()) {
      assert((unsigned)VT.getSimpleVT().SimpleTy <
                array_lengthof(RegisterTypeForVT));
      return RegisterTypeForVT[VT.getSimpleVT().SimpleTy];
    }
    if (VT.isVector()) {
      EVT VT1, RegisterVT;
      unsigned NumIntermediates;
      (void)getVectorTypeBreakdown(Context, VT, VT1,
                                   NumIntermediates, RegisterVT);
      return RegisterVT;
    }
    if (VT.isInteger()) {
      return getRegisterType(Context, getTypeToTransformTo(Context, VT));
    }
    assert(0 && "Unsupported extended type!");
    return EVT(MVT::Other); // Not reached
  }

  /// getNumRegisters - Return the number of registers that this ValueType will
  /// eventually require.  This is one for any types promoted to live in larger
  /// registers, but may be more than one for types (like i64) that are split
  /// into pieces.  For types like i140, which are first promoted then expanded,
  /// it is the number of registers needed to hold all the bits of the original
  /// type.  For an i140 on a 32 bit machine this means 5 registers.
  unsigned getNumRegisters(LLVMContext &Context, EVT VT) const {
    if (VT.isSimple()) {
      assert((unsigned)VT.getSimpleVT().SimpleTy <
                array_lengthof(NumRegistersForVT));
      return NumRegistersForVT[VT.getSimpleVT().SimpleTy];
    }
    if (VT.isVector()) {
      EVT VT1, VT2;
      unsigned NumIntermediates;
      return getVectorTypeBreakdown(Context, VT, VT1, NumIntermediates, VT2);
    }
    if (VT.isInteger()) {
      unsigned BitWidth = VT.getSizeInBits();
      unsigned RegWidth = getRegisterType(Context, VT).getSizeInBits();
      return (BitWidth + RegWidth - 1) / RegWidth;
    }
    assert(0 && "Unsupported extended type!");
    return 0; // Not reached
  }

  /// ShouldShrinkFPConstant - If true, then instruction selection should
  /// seek to shrink the FP constant of the specified type to a smaller type
  /// in order to save space and / or reduce runtime.
  virtual bool ShouldShrinkFPConstant(EVT) const { return true; }

  /// hasTargetDAGCombine - If true, the target has custom DAG combine
  /// transformations that it can perform for the specified node.
  bool hasTargetDAGCombine(ISD::NodeType NT) const {
    assert(unsigned(NT >> 3) < array_lengthof(TargetDAGCombineArray));
    return TargetDAGCombineArray[NT >> 3] & (1 << (NT&7));
  }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memset. The value is set by the target at the
  /// performance threshold for such a replacement. If OptSize is true,
  /// return the limit for functions that have OptSize attribute.
  /// @brief Get maximum # of store operations permitted for llvm.memset
  unsigned getMaxStoresPerMemset(bool OptSize) const {
    return OptSize ? maxStoresPerMemsetOptSize : maxStoresPerMemset;
  }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memcpy. The value is set by the target at the
  /// performance threshold for such a replacement. If OptSize is true,
  /// return the limit for functions that have OptSize attribute.
  /// @brief Get maximum # of store operations permitted for llvm.memcpy
  unsigned getMaxStoresPerMemcpy(bool OptSize) const {
    return OptSize ? maxStoresPerMemcpyOptSize : maxStoresPerMemcpy;
  }

  /// This function returns the maximum number of store operations permitted
  /// to replace a call to llvm.memmove. The value is set by the target at the
  /// performance threshold for such a replacement. If OptSize is true,
  /// return the limit for functions that have OptSize attribute.
  /// @brief Get maximum # of store operations permitted for llvm.memmove
  unsigned getMaxStoresPerMemmove(bool OptSize) const {
    return OptSize ? maxStoresPerMemmoveOptSize : maxStoresPerMemmove;
  }

  /// This function returns true if the target allows unaligned memory accesses.
  /// of the specified type. This is used, for example, in situations where an
  /// array copy/move/set is  converted to a sequence of store operations. It's
  /// use helps to ensure that such replacements don't generate code that causes
  /// an alignment error  (trap) on the target machine.
  /// @brief Determine if the target supports unaligned memory accesses.
  virtual bool allowsUnalignedMemoryAccesses(EVT) const {
    return false;
  }

  /// This function returns true if the target would benefit from code placement
  /// optimization.
  /// @brief Determine if the target should perform code placement optimization.
  bool shouldOptimizeCodePlacement() const {
    return benefitFromCodePlacementOpt;
  }

  /// getOptimalMemOpType - Returns the target specific optimal type for load
  /// and store operations as a result of memset, memcpy, and memmove
  /// lowering. If DstAlign is zero that means it's safe to destination
  /// alignment can satisfy any constraint. Similarly if SrcAlign is zero it
  /// means there isn't a need to check it against alignment requirement,
  /// probably because the source does not need to be loaded. If
  /// 'NonScalarIntSafe' is true, that means it's safe to return a
  /// non-scalar-integer type, e.g. empty string source, constant, or loaded
  /// from memory. 'MemcpyStrSrc' indicates whether the memcpy source is
  /// constant so it does not need to be loaded.
  /// It returns EVT::Other if the type should be determined using generic
  /// target-independent logic.
  virtual EVT getOptimalMemOpType(uint64_t /*Size*/,
                                  unsigned /*DstAlign*/, unsigned /*SrcAlign*/,
                                  bool /*NonScalarIntSafe*/,
                                  bool /*MemcpyStrSrc*/,
                                  MachineFunction &/*MF*/) const {
    return MVT::Other;
  }

  /// usesUnderscoreSetJmp - Determine if we should use _setjmp or setjmp
  /// to implement llvm.setjmp.
  bool usesUnderscoreSetJmp() const {
    return UseUnderscoreSetJmp;
  }

  /// usesUnderscoreLongJmp - Determine if we should use _longjmp or longjmp
  /// to implement llvm.longjmp.
  bool usesUnderscoreLongJmp() const {
    return UseUnderscoreLongJmp;
  }

  /// getStackPointerRegisterToSaveRestore - If a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  unsigned getStackPointerRegisterToSaveRestore() const {
    return StackPointerRegisterToSaveRestore;
  }

  /// getExceptionAddressRegister - If a physical register, this returns
  /// the register that receives the exception address on entry to a landing
  /// pad.
  unsigned getExceptionAddressRegister() const {
    return ExceptionPointerRegister;
  }

  /// getExceptionSelectorRegister - If a physical register, this returns
  /// the register that receives the exception typeid on entry to a landing
  /// pad.
  unsigned getExceptionSelectorRegister() const {
    return ExceptionSelectorRegister;
  }

  /// getJumpBufSize - returns the target's jmp_buf size in bytes (if never
  /// set, the default is 200)
  unsigned getJumpBufSize() const {
    return JumpBufSize;
  }

  /// getJumpBufAlignment - returns the target's jmp_buf alignment in bytes
  /// (if never set, the default is 0)
  unsigned getJumpBufAlignment() const {
    return JumpBufAlignment;
  }

  /// getMinStackArgumentAlignment - return the minimum stack alignment of an
  /// argument.
  unsigned getMinStackArgumentAlignment() const {
    return MinStackArgumentAlignment;
  }

  /// getMinFunctionAlignment - return the minimum function alignment.
  ///
  unsigned getMinFunctionAlignment() const {
    return MinFunctionAlignment;
  }

  /// getPrefFunctionAlignment - return the preferred function alignment.
  ///
  unsigned getPrefFunctionAlignment() const {
    return PrefFunctionAlignment;
  }

  /// getPrefLoopAlignment - return the preferred loop alignment.
  ///
  unsigned getPrefLoopAlignment() const {
    return PrefLoopAlignment;
  }

  /// getShouldFoldAtomicFences - return whether the combiner should fold
  /// fence MEMBARRIER instructions into the atomic intrinsic instructions.
  ///
  bool getShouldFoldAtomicFences() const {
    return ShouldFoldAtomicFences;
  }

  /// getInsertFencesFor - return whether the DAG builder should automatically
  /// insert fences and reduce ordering for atomics.
  ///
  bool getInsertFencesForAtomic() const {
    return InsertFencesForAtomic;
  }

  /// getPreIndexedAddressParts - returns true by value, base pointer and
  /// offset pointer and addressing mode by reference if the node's address
  /// can be legally represented as pre-indexed load / store address.
  virtual bool getPreIndexedAddressParts(SDNode * /*N*/, SDValue &/*Base*/,
                                         SDValue &/*Offset*/,
                                         ISD::MemIndexedMode &/*AM*/,
                                         SelectionDAG &/*DAG*/) const {
    return false;
  }

  /// getPostIndexedAddressParts - returns true by value, base pointer and
  /// offset pointer and addressing mode by reference if this node can be
  /// combined with a load / store to form a post-indexed load / store.
  virtual bool getPostIndexedAddressParts(SDNode * /*N*/, SDNode * /*Op*/,
                                          SDValue &/*Base*/, SDValue &/*Offset*/,
                                          ISD::MemIndexedMode &/*AM*/,
                                          SelectionDAG &/*DAG*/) const {
    return false;
  }

  /// getJumpTableEncoding - Return the entry encoding for a jump table in the
  /// current function.  The returned value is a member of the
  /// MachineJumpTableInfo::JTEntryKind enum.
  virtual unsigned getJumpTableEncoding() const;

  virtual const MCExpr *
  LowerCustomJumpTableEntry(const MachineJumpTableInfo * /*MJTI*/,
                            const MachineBasicBlock * /*MBB*/, unsigned /*uid*/,
                            MCContext &/*Ctx*/) const {
    assert(0 && "Need to implement this hook if target has custom JTIs");
    return 0;
  }

  /// getPICJumpTableRelocaBase - Returns relocation base for the given PIC
  /// jumptable.
  virtual SDValue getPICJumpTableRelocBase(SDValue Table,
                                           SelectionDAG &DAG) const;

  /// getPICJumpTableRelocBaseExpr - This returns the relocation base for the
  /// given PIC jumptable, the same as getPICJumpTableRelocBase, but as an
  /// MCExpr.
  virtual const MCExpr *
  getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                               unsigned JTI, MCContext &Ctx) const;

  /// isOffsetFoldingLegal - Return true if folding a constant offset
  /// with the given GlobalAddress is legal.  It is frequently not legal in
  /// PIC relocation models.
  virtual bool isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const;

  /// getStackCookieLocation - Return true if the target stores stack
  /// protector cookies at a fixed offset in some non-standard address
  /// space, and populates the address space and offset as
  /// appropriate.
  virtual bool getStackCookieLocation(unsigned &/*AddressSpace*/,
                                      unsigned &/*Offset*/) const {
    return false;
  }

  /// getMaximalGlobalOffset - Returns the maximal possible offset which can be
  /// used for loads / stores from the global.
  virtual unsigned getMaximalGlobalOffset() const {
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Optimization Methods
  //

  /// TargetLoweringOpt - A convenience struct that encapsulates a DAG, and two
  /// SDValues for returning information from TargetLowering to its clients
  /// that want to combine
  struct TargetLoweringOpt {
    SelectionDAG &DAG;
    bool LegalTys;
    bool LegalOps;
    SDValue Old;
    SDValue New;

    explicit TargetLoweringOpt(SelectionDAG &InDAG,
                               bool LT, bool LO) :
      DAG(InDAG), LegalTys(LT), LegalOps(LO) {}

    bool LegalTypes() const { return LegalTys; }
    bool LegalOperations() const { return LegalOps; }

    bool CombineTo(SDValue O, SDValue N) {
      Old = O;
      New = N;
      return true;
    }

    /// ShrinkDemandedConstant - Check to see if the specified operand of the
    /// specified instruction is a constant integer.  If so, check to see if
    /// there are any bits set in the constant that are not demanded.  If so,
    /// shrink the constant and return true.
    bool ShrinkDemandedConstant(SDValue Op, const APInt &Demanded);

    /// ShrinkDemandedOp - Convert x+y to (VT)((SmallVT)x+(SmallVT)y) if the
    /// casts are free.  This uses isZExtFree and ZERO_EXTEND for the widening
    /// cast, but it could be generalized for targets with other types of
    /// implicit widening casts.
    bool ShrinkDemandedOp(SDValue Op, unsigned BitWidth, const APInt &Demanded,
                          DebugLoc dl);
  };

  /// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
  /// DemandedMask bits of the result of Op are ever used downstream.  If we can
  /// use this information to simplify Op, create a new simplified DAG node and
  /// return true, returning the original and new nodes in Old and New.
  /// Otherwise, analyze the expression and return a mask of KnownOne and
  /// KnownZero bits for the expression (used to simplify the caller).
  /// The KnownZero/One bits may only be accurate for those bits in the
  /// DemandedMask.
  bool SimplifyDemandedBits(SDValue Op, const APInt &DemandedMask,
                            APInt &KnownZero, APInt &KnownOne,
                            TargetLoweringOpt &TLO, unsigned Depth = 0) const;

  /// computeMaskedBitsForTargetNode - Determine which of the bits specified in
  /// Mask are known to be either zero or one and return them in the
  /// KnownZero/KnownOne bitsets.
  virtual void computeMaskedBitsForTargetNode(const SDValue Op,
                                              const APInt &Mask,
                                              APInt &KnownZero,
                                              APInt &KnownOne,
                                              const SelectionDAG &DAG,
                                              unsigned Depth = 0) const;

  /// ComputeNumSignBitsForTargetNode - This method can be implemented by
  /// targets that want to expose additional information about sign bits to the
  /// DAG Combiner.
  virtual unsigned ComputeNumSignBitsForTargetNode(SDValue Op,
                                                   unsigned Depth = 0) const;

  struct DAGCombinerInfo {
    void *DC;  // The DAG Combiner object.
    bool BeforeLegalize;
    bool BeforeLegalizeOps;
    bool CalledByLegalizer;
  public:
    SelectionDAG &DAG;

    DAGCombinerInfo(SelectionDAG &dag, bool bl, bool blo, bool cl, void *dc)
      : DC(dc), BeforeLegalize(bl), BeforeLegalizeOps(blo),
        CalledByLegalizer(cl), DAG(dag) {}

    bool isBeforeLegalize() const { return BeforeLegalize; }
    bool isBeforeLegalizeOps() const { return BeforeLegalizeOps; }
    bool isCalledByLegalizer() const { return CalledByLegalizer; }

    void AddToWorklist(SDNode *N);
    void RemoveFromWorklist(SDNode *N);
    SDValue CombineTo(SDNode *N, const std::vector<SDValue> &To,
                      bool AddTo = true);
    SDValue CombineTo(SDNode *N, SDValue Res, bool AddTo = true);
    SDValue CombineTo(SDNode *N, SDValue Res0, SDValue Res1, bool AddTo = true);

    void CommitTargetLoweringOpt(const TargetLoweringOpt &TLO);
  };

  /// SimplifySetCC - Try to simplify a setcc built with the specified operands
  /// and cc. If it is unable to simplify it, return a null SDValue.
  SDValue SimplifySetCC(EVT VT, SDValue N0, SDValue N1,
                          ISD::CondCode Cond, bool foldBooleans,
                          DAGCombinerInfo &DCI, DebugLoc dl) const;

  /// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
  /// node is a GlobalAddress + offset.
  virtual bool
  isGAPlusOffset(SDNode *N, const GlobalValue* &GA, int64_t &Offset) const;

  /// PerformDAGCombine - This method will be invoked for all target nodes and
  /// for any target-independent nodes that the target has registered with
  /// invoke it for.
  ///
  /// The semantics are as follows:
  /// Return Value:
  ///   SDValue.Val == 0   - No change was made
  ///   SDValue.Val == N   - N was replaced, is dead, and is already handled.
  ///   otherwise          - N should be replaced by the returned Operand.
  ///
  /// In addition, methods provided by DAGCombinerInfo may be used to perform
  /// more complex transformations.
  ///
  virtual SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  /// isTypeDesirableForOp - Return true if the target has native support for
  /// the specified value type and it is 'desirable' to use the type for the
  /// given node type. e.g. On x86 i16 is legal, but undesirable since i16
  /// instruction encodings are longer and some i16 instructions are slow.
  virtual bool isTypeDesirableForOp(unsigned /*Opc*/, EVT VT) const {
    // By default, assume all legal types are desirable.
    return isTypeLegal(VT);
  }

  /// isDesirableToPromoteOp - Return true if it is profitable for dag combiner
  /// to transform a floating point op of specified opcode to a equivalent op of
  /// an integer type. e.g. f32 load -> i32 load can be profitable on ARM.
  virtual bool isDesirableToTransformToIntegerOp(unsigned /*Opc*/,
                                                 EVT /*VT*/) const {
    return false;
  }

  /// IsDesirableToPromoteOp - This method query the target whether it is
  /// beneficial for dag combiner to promote the specified node. If true, it
  /// should return the desired promotion type by reference.
  virtual bool IsDesirableToPromoteOp(SDValue /*Op*/, EVT &/*PVT*/) const {
    return false;
  }

  //===--------------------------------------------------------------------===//
  // TargetLowering Configuration Methods - These methods should be invoked by
  // the derived class constructor to configure this object for the target.
  //

protected:
  /// setBooleanContents - Specify how the target extends the result of a
  /// boolean value from i1 to a wider type.  See getBooleanContents.
  void setBooleanContents(BooleanContent Ty) { BooleanContents = Ty; }
  /// setBooleanVectorContents - Specify how the target extends the result
  /// of a vector boolean value from a vector of i1 to a wider type.  See
  /// getBooleanContents.
  void setBooleanVectorContents(BooleanContent Ty) {
    BooleanVectorContents = Ty;
  }

  /// setSchedulingPreference - Specify the target scheduling preference.
  void setSchedulingPreference(Sched::Preference Pref) {
    SchedPreferenceInfo = Pref;
  }

  /// setUseUnderscoreSetJmp - Indicate whether this target prefers to
  /// use _setjmp to implement llvm.setjmp or the non _ version.
  /// Defaults to false.
  void setUseUnderscoreSetJmp(bool Val) {
    UseUnderscoreSetJmp = Val;
  }

  /// setUseUnderscoreLongJmp - Indicate whether this target prefers to
  /// use _longjmp to implement llvm.longjmp or the non _ version.
  /// Defaults to false.
  void setUseUnderscoreLongJmp(bool Val) {
    UseUnderscoreLongJmp = Val;
  }

  /// setStackPointerRegisterToSaveRestore - If set to a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  void setStackPointerRegisterToSaveRestore(unsigned R) {
    StackPointerRegisterToSaveRestore = R;
  }

  /// setExceptionPointerRegister - If set to a physical register, this sets
  /// the register that receives the exception address on entry to a landing
  /// pad.
  void setExceptionPointerRegister(unsigned R) {
    ExceptionPointerRegister = R;
  }

  /// setExceptionSelectorRegister - If set to a physical register, this sets
  /// the register that receives the exception typeid on entry to a landing
  /// pad.
  void setExceptionSelectorRegister(unsigned R) {
    ExceptionSelectorRegister = R;
  }

  /// SelectIsExpensive - Tells the code generator not to expand operations
  /// into sequences that use the select operations if possible.
  void setSelectIsExpensive(bool isExpensive = true) {
    SelectIsExpensive = isExpensive;
  }

  /// JumpIsExpensive - Tells the code generator not to expand sequence of
  /// operations into a separate sequences that increases the amount of
  /// flow control.
  void setJumpIsExpensive(bool isExpensive = true) {
    JumpIsExpensive = isExpensive;
  }

  /// setIntDivIsCheap - Tells the code generator that integer divide is
  /// expensive, and if possible, should be replaced by an alternate sequence
  /// of instructions not containing an integer divide.
  void setIntDivIsCheap(bool isCheap = true) { IntDivIsCheap = isCheap; }

  /// setPow2DivIsCheap - Tells the code generator that it shouldn't generate
  /// srl/add/sra for a signed divide by power of two, and let the target handle
  /// it.
  void setPow2DivIsCheap(bool isCheap = true) { Pow2DivIsCheap = isCheap; }

  /// addRegisterClass - Add the specified register class as an available
  /// regclass for the specified value type.  This indicates the selector can
  /// handle values of that class natively.
  void addRegisterClass(EVT VT, TargetRegisterClass *RC) {
    assert((unsigned)VT.getSimpleVT().SimpleTy < array_lengthof(RegClassForVT));
    AvailableRegClasses.push_back(std::make_pair(VT, RC));
    RegClassForVT[VT.getSimpleVT().SimpleTy] = RC;
  }

  /// findRepresentativeClass - Return the largest legal super-reg register class
  /// of the register class for the specified type and its associated "cost".
  virtual std::pair<const TargetRegisterClass*, uint8_t>
  findRepresentativeClass(EVT VT) const;

  /// computeRegisterProperties - Once all of the register classes are added,
  /// this allows us to compute derived properties we expose.
  void computeRegisterProperties();

  /// setOperationAction - Indicate that the specified operation does not work
  /// with the specified type and indicate what to do about it.
  void setOperationAction(unsigned Op, MVT VT,
                          LegalizeAction Action) {
    assert(Op < array_lengthof(OpActions[0]) && "Table isn't big enough!");
    OpActions[(unsigned)VT.SimpleTy][Op] = (uint8_t)Action;
  }

  /// setLoadExtAction - Indicate that the specified load with extension does
  /// not work with the specified type and indicate what to do about it.
  void setLoadExtAction(unsigned ExtType, MVT VT,
                        LegalizeAction Action) {
    assert(ExtType < ISD::LAST_LOADEXT_TYPE && VT < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    LoadExtActions[VT.SimpleTy][ExtType] = (uint8_t)Action;
  }

  /// setTruncStoreAction - Indicate that the specified truncating store does
  /// not work with the specified type and indicate what to do about it.
  void setTruncStoreAction(MVT ValVT, MVT MemVT,
                           LegalizeAction Action) {
    assert(ValVT < MVT::LAST_VALUETYPE && MemVT < MVT::LAST_VALUETYPE &&
           "Table isn't big enough!");
    TruncStoreActions[ValVT.SimpleTy][MemVT.SimpleTy] = (uint8_t)Action;
  }

  /// setIndexedLoadAction - Indicate that the specified indexed load does or
  /// does not work with the specified type and indicate what to do abort
  /// it. NOTE: All indexed mode loads are initialized to Expand in
  /// TargetLowering.cpp
  void setIndexedLoadAction(unsigned IdxMode, MVT VT,
                            LegalizeAction Action) {
    assert(VT < MVT::LAST_VALUETYPE && IdxMode < ISD::LAST_INDEXED_MODE &&
           (unsigned)Action < 0xf && "Table isn't big enough!");
    // Load action are kept in the upper half.
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] &= ~0xf0;
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] |= ((uint8_t)Action) <<4;
  }

  /// setIndexedStoreAction - Indicate that the specified indexed store does or
  /// does not work with the specified type and indicate what to do about
  /// it. NOTE: All indexed mode stores are initialized to Expand in
  /// TargetLowering.cpp
  void setIndexedStoreAction(unsigned IdxMode, MVT VT,
                             LegalizeAction Action) {
    assert(VT < MVT::LAST_VALUETYPE && IdxMode < ISD::LAST_INDEXED_MODE &&
           (unsigned)Action < 0xf && "Table isn't big enough!");
    // Store action are kept in the lower half.
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] &= ~0x0f;
    IndexedModeActions[(unsigned)VT.SimpleTy][IdxMode] |= ((uint8_t)Action);
  }

  /// setCondCodeAction - Indicate that the specified condition code is or isn't
  /// supported on the target and indicate what to do about it.
  void setCondCodeAction(ISD::CondCode CC, MVT VT,
                         LegalizeAction Action) {
    assert(VT < MVT::LAST_VALUETYPE &&
           (unsigned)CC < array_lengthof(CondCodeActions) &&
           "Table isn't big enough!");
    CondCodeActions[(unsigned)CC] &= ~(uint64_t(3UL)  << VT.SimpleTy*2);
    CondCodeActions[(unsigned)CC] |= (uint64_t)Action << VT.SimpleTy*2;
  }

  /// AddPromotedToType - If Opc/OrigVT is specified as being promoted, the
  /// promotion code defaults to trying a larger integer/fp until it can find
  /// one that works.  If that default is insufficient, this method can be used
  /// by the target to override the default.
  void AddPromotedToType(unsigned Opc, MVT OrigVT, MVT DestVT) {
    PromoteToType[std::make_pair(Opc, OrigVT.SimpleTy)] = DestVT.SimpleTy;
  }

  /// setTargetDAGCombine - Targets should invoke this method for each target
  /// independent node that they want to provide a custom DAG combiner for by
  /// implementing the PerformDAGCombine virtual method.
  void setTargetDAGCombine(ISD::NodeType NT) {
    assert(unsigned(NT >> 3) < array_lengthof(TargetDAGCombineArray));
    TargetDAGCombineArray[NT >> 3] |= 1 << (NT&7);
  }

  /// setJumpBufSize - Set the target's required jmp_buf buffer size (in
  /// bytes); default is 200
  void setJumpBufSize(unsigned Size) {
    JumpBufSize = Size;
  }

  /// setJumpBufAlignment - Set the target's required jmp_buf buffer
  /// alignment (in bytes); default is 0
  void setJumpBufAlignment(unsigned Align) {
    JumpBufAlignment = Align;
  }

  /// setMinFunctionAlignment - Set the target's minimum function alignment.
  void setMinFunctionAlignment(unsigned Align) {
    MinFunctionAlignment = Align;
  }

  /// setPrefFunctionAlignment - Set the target's preferred function alignment.
  /// This should be set if there is a performance benefit to
  /// higher-than-minimum alignment
  void setPrefFunctionAlignment(unsigned Align) {
    PrefFunctionAlignment = Align;
  }

  /// setPrefLoopAlignment - Set the target's preferred loop alignment. Default
  /// alignment is zero, it means the target does not care about loop alignment.
  void setPrefLoopAlignment(unsigned Align) {
    PrefLoopAlignment = Align;
  }

  /// setMinStackArgumentAlignment - Set the minimum stack alignment of an
  /// argument.
  void setMinStackArgumentAlignment(unsigned Align) {
    MinStackArgumentAlignment = Align;
  }

  /// setShouldFoldAtomicFences - Set if the target's implementation of the
  /// atomic operation intrinsics includes locking. Default is false.
  void setShouldFoldAtomicFences(bool fold) {
    ShouldFoldAtomicFences = fold;
  }

  /// setInsertFencesForAtomic - Set if the the DAG builder should
  /// automatically insert fences and reduce the order of atomic memory
  /// operations to Monotonic.
  void setInsertFencesForAtomic(bool fence) {
    InsertFencesForAtomic = fence;
  }

public:
  //===--------------------------------------------------------------------===//
  // Lowering methods - These methods must be implemented by targets so that
  // the SelectionDAGLowering code knows how to lower these.
  //

  /// LowerFormalArguments - This hook must be implemented to lower the
  /// incoming (formal) arguments, described by the Ins array, into the
  /// specified DAG. The implementation should fill in the InVals array
  /// with legal-type argument values, and return the resulting token
  /// chain value.
  ///
  virtual SDValue
    LowerFormalArguments(SDValue /*Chain*/, CallingConv::ID /*CallConv*/,
                         bool /*isVarArg*/,
                         const SmallVectorImpl<ISD::InputArg> &/*Ins*/,
                         DebugLoc /*dl*/, SelectionDAG &/*DAG*/,
                         SmallVectorImpl<SDValue> &/*InVals*/) const {
    assert(0 && "Not Implemented");
    return SDValue();    // this is here to silence compiler errors
  }

  /// LowerCallTo - This function lowers an abstract call to a function into an
  /// actual call.  This returns a pair of operands.  The first element is the
  /// return value for the function (if RetTy is not VoidTy).  The second
  /// element is the outgoing token chain. It calls LowerCall to do the actual
  /// lowering.
  struct ArgListEntry {
    SDValue Node;
    Type* Ty;
    bool isSExt  : 1;
    bool isZExt  : 1;
    bool isInReg : 1;
    bool isSRet  : 1;
    bool isNest  : 1;
    bool isByVal : 1;
    uint16_t Alignment;

    ArgListEntry() : isSExt(false), isZExt(false), isInReg(false),
      isSRet(false), isNest(false), isByVal(false), Alignment(0) { }
  };
  typedef std::vector<ArgListEntry> ArgListTy;
  std::pair<SDValue, SDValue>
  LowerCallTo(SDValue Chain, Type *RetTy, bool RetSExt, bool RetZExt,
              bool isVarArg, bool isInreg, unsigned NumFixedArgs,
              CallingConv::ID CallConv, bool isTailCall,
              bool isReturnValueUsed, SDValue Callee, ArgListTy &Args,
              SelectionDAG &DAG, DebugLoc dl) const;

  /// LowerCall - This hook must be implemented to lower calls into the
  /// the specified DAG. The outgoing arguments to the call are described
  /// by the Outs array, and the values to be returned by the call are
  /// described by the Ins array. The implementation should fill in the
  /// InVals array with legal-type return values from the call, and return
  /// the resulting token chain value.
  virtual SDValue
    LowerCall(SDValue /*Chain*/, SDValue /*Callee*/,
              CallingConv::ID /*CallConv*/, bool /*isVarArg*/,
              bool &/*isTailCall*/,
              const SmallVectorImpl<ISD::OutputArg> &/*Outs*/,
              const SmallVectorImpl<SDValue> &/*OutVals*/,
              const SmallVectorImpl<ISD::InputArg> &/*Ins*/,
              DebugLoc /*dl*/, SelectionDAG &/*DAG*/,
              SmallVectorImpl<SDValue> &/*InVals*/) const {
    assert(0 && "Not Implemented");
    return SDValue();    // this is here to silence compiler errors
  }

  /// HandleByVal - Target-specific cleanup for formal ByVal parameters.
  virtual void HandleByVal(CCState *, unsigned &) const {}

  /// CanLowerReturn - This hook should be implemented to check whether the
  /// return values described by the Outs array can fit into the return
  /// registers.  If false is returned, an sret-demotion is performed.
  ///
  virtual bool CanLowerReturn(CallingConv::ID /*CallConv*/,
			      MachineFunction &/*MF*/, bool /*isVarArg*/,
               const SmallVectorImpl<ISD::OutputArg> &/*Outs*/,
               LLVMContext &/*Context*/) const
  {
    // Return true by default to get preexisting behavior.
    return true;
  }

  /// LowerReturn - This hook must be implemented to lower outgoing
  /// return values, described by the Outs array, into the specified
  /// DAG. The implementation should return the resulting token chain
  /// value.
  ///
  virtual SDValue
    LowerReturn(SDValue /*Chain*/, CallingConv::ID /*CallConv*/,
                bool /*isVarArg*/,
                const SmallVectorImpl<ISD::OutputArg> &/*Outs*/,
                const SmallVectorImpl<SDValue> &/*OutVals*/,
                DebugLoc /*dl*/, SelectionDAG &/*DAG*/) const {
    assert(0 && "Not Implemented");
    return SDValue();    // this is here to silence compiler errors
  }

  /// isUsedByReturnOnly - Return true if result of the specified node is used
  /// by a return node only. This is used to determine whether it is possible
  /// to codegen a libcall as tail call at legalization time.
  virtual bool isUsedByReturnOnly(SDNode *) const {
    return false;
  }

  /// mayBeEmittedAsTailCall - Return true if the target may be able emit the
  /// call instruction as a tail call. This is used by optimization passes to
  /// determine if it's profitable to duplicate return instructions to enable
  /// tailcall optimization.
  virtual bool mayBeEmittedAsTailCall(CallInst *) const {
    return false;
  }

  /// getTypeForExtArgOrReturn - Return the type that should be used to zero or
  /// sign extend a zeroext/signext integer argument or return value.
  /// FIXME: Most C calling convention requires the return type to be promoted,
  /// but this is not true all the time, e.g. i1 on x86-64. It is also not
  /// necessary for non-C calling conventions. The frontend should handle this
  /// and include all of the necessary information.
  virtual EVT getTypeForExtArgOrReturn(LLVMContext &Context, EVT VT,
                                       ISD::NodeType /*ExtendKind*/) const {
    EVT MinVT = getRegisterType(Context, MVT::i32);
    return VT.bitsLT(MinVT) ? MinVT : VT;
  }

  /// LowerOperationWrapper - This callback is invoked by the type legalizer
  /// to legalize nodes with an illegal operand type but legal result types.
  /// It replaces the LowerOperation callback in the type Legalizer.
  /// The reason we can not do away with LowerOperation entirely is that
  /// LegalizeDAG isn't yet ready to use this callback.
  /// TODO: Consider merging with ReplaceNodeResults.

  /// The target places new result values for the node in Results (their number
  /// and types must exactly match those of the original return values of
  /// the node), or leaves Results empty, which indicates that the node is not
  /// to be custom lowered after all.
  /// The default implementation calls LowerOperation.
  virtual void LowerOperationWrapper(SDNode *N,
                                     SmallVectorImpl<SDValue> &Results,
                                     SelectionDAG &DAG) const;

  /// LowerOperation - This callback is invoked for operations that are
  /// unsupported by the target, which are registered to use 'custom' lowering,
  /// and whose defined values are all legal.
  /// If the target has no operations that require custom lowering, it need not
  /// implement this.  The default implementation of this aborts.
  virtual SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const;

  /// ReplaceNodeResults - This callback is invoked when a node result type is
  /// illegal for the target, and the operation was registered to use 'custom'
  /// lowering for that result type.  The target places new result values for
  /// the node in Results (their number and types must exactly match those of
  /// the original return values of the node), or leaves Results empty, which
  /// indicates that the node is not to be custom lowered after all.
  ///
  /// If the target has no operations that require custom lowering, it need not
  /// implement this.  The default implementation aborts.
  virtual void ReplaceNodeResults(SDNode * /*N*/,
                                  SmallVectorImpl<SDValue> &/*Results*/,
                                  SelectionDAG &/*DAG*/) const {
    assert(0 && "ReplaceNodeResults not implemented for this target!");
  }

  /// getTargetNodeName() - This method returns the name of a target specific
  /// DAG node.
  virtual const char *getTargetNodeName(unsigned Opcode) const;

  /// createFastISel - This method returns a target specific FastISel object,
  /// or null if the target does not support "fast" ISel.
  virtual FastISel *createFastISel(FunctionLoweringInfo &) const {
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Inline Asm Support hooks
  //

  /// ExpandInlineAsm - This hook allows the target to expand an inline asm
  /// call to be explicit llvm code if it wants to.  This is useful for
  /// turning simple inline asms into LLVM intrinsics, which gives the
  /// compiler more information about the behavior of the code.
  virtual bool ExpandInlineAsm(CallInst *) const {
    return false;
  }

  enum ConstraintType {
    C_Register,            // Constraint represents specific register(s).
    C_RegisterClass,       // Constraint represents any of register(s) in class.
    C_Memory,              // Memory constraint.
    C_Other,               // Something else.
    C_Unknown              // Unsupported constraint.
  };

  enum ConstraintWeight {
    // Generic weights.
    CW_Invalid  = -1,     // No match.
    CW_Okay     = 0,      // Acceptable.
    CW_Good     = 1,      // Good weight.
    CW_Better   = 2,      // Better weight.
    CW_Best     = 3,      // Best weight.

    // Well-known weights.
    CW_SpecificReg  = CW_Okay,    // Specific register operands.
    CW_Register     = CW_Good,    // Register operands.
    CW_Memory       = CW_Better,  // Memory operands.
    CW_Constant     = CW_Best,    // Constant operand.
    CW_Default      = CW_Okay     // Default or don't know type.
  };

  /// AsmOperandInfo - This contains information for each constraint that we are
  /// lowering.
  struct AsmOperandInfo : public InlineAsm::ConstraintInfo {
    /// ConstraintCode - This contains the actual string for the code, like "m".
    /// TargetLowering picks the 'best' code from ConstraintInfo::Codes that
    /// most closely matches the operand.
    std::string ConstraintCode;

    /// ConstraintType - Information about the constraint code, e.g. Register,
    /// RegisterClass, Memory, Other, Unknown.
    TargetLowering::ConstraintType ConstraintType;

    /// CallOperandval - If this is the result output operand or a
    /// clobber, this is null, otherwise it is the incoming operand to the
    /// CallInst.  This gets modified as the asm is processed.
    Value *CallOperandVal;

    /// ConstraintVT - The ValueType for the operand value.
    EVT ConstraintVT;

    /// isMatchingInputConstraint - Return true of this is an input operand that
    /// is a matching constraint like "4".
    bool isMatchingInputConstraint() const;

    /// getMatchedOperand - If this is an input matching constraint, this method
    /// returns the output operand it matches.
    unsigned getMatchedOperand() const;

    /// Copy constructor for copying from an AsmOperandInfo.
    AsmOperandInfo(const AsmOperandInfo &info)
      : InlineAsm::ConstraintInfo(info),
        ConstraintCode(info.ConstraintCode),
        ConstraintType(info.ConstraintType),
        CallOperandVal(info.CallOperandVal),
        ConstraintVT(info.ConstraintVT) {
    }

    /// Copy constructor for copying from a ConstraintInfo.
    AsmOperandInfo(const InlineAsm::ConstraintInfo &info)
      : InlineAsm::ConstraintInfo(info),
        ConstraintType(TargetLowering::C_Unknown),
        CallOperandVal(0), ConstraintVT(MVT::Other) {
    }
  };

  typedef std::vector<AsmOperandInfo> AsmOperandInfoVector;

  /// ParseConstraints - Split up the constraint string from the inline
  /// assembly value into the specific constraints and their prefixes,
  /// and also tie in the associated operand values.
  /// If this returns an empty vector, and if the constraint string itself
  /// isn't empty, there was an error parsing.
  virtual AsmOperandInfoVector ParseConstraints(ImmutableCallSite CS) const;

  /// Examine constraint type and operand type and determine a weight value.
  /// The operand object must already have been set up with the operand type.
  virtual ConstraintWeight getMultipleConstraintMatchWeight(
      AsmOperandInfo &info, int maIndex) const;

  /// Examine constraint string and operand type and determine a weight value.
  /// The operand object must already have been set up with the operand type.
  virtual ConstraintWeight getSingleConstraintMatchWeight(
      AsmOperandInfo &info, const char *constraint) const;

  /// ComputeConstraintToUse - Determines the constraint code and constraint
  /// type to use for the specific AsmOperandInfo, setting
  /// OpInfo.ConstraintCode and OpInfo.ConstraintType.  If the actual operand
  /// being passed in is available, it can be passed in as Op, otherwise an
  /// empty SDValue can be passed.
  virtual void ComputeConstraintToUse(AsmOperandInfo &OpInfo,
                                      SDValue Op,
                                      SelectionDAG *DAG = 0) const;

  /// getConstraintType - Given a constraint, return the type of constraint it
  /// is for this target.
  virtual ConstraintType getConstraintType(const std::string &Constraint) const;

  /// getRegForInlineAsmConstraint - Given a physical register constraint (e.g.
  /// {edx}), return the register number and the register class for the
  /// register.
  ///
  /// Given a register class constraint, like 'r', if this corresponds directly
  /// to an LLVM register class, return a register of 0 and the register class
  /// pointer.
  ///
  /// This should only be used for C_Register constraints.  On error,
  /// this returns a register number of 0 and a null register class pointer..
  virtual std::pair<unsigned, const TargetRegisterClass*>
    getRegForInlineAsmConstraint(const std::string &Constraint,
                                 EVT VT) const;

  /// LowerXConstraint - try to replace an X constraint, which matches anything,
  /// with another that has more specific requirements based on the type of the
  /// corresponding operand.  This returns null if there is no replacement to
  /// make.
  virtual const char *LowerXConstraint(EVT ConstraintVT) const;

  /// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
  /// vector.  If it is invalid, don't add anything to Ops.
  virtual void LowerAsmOperandForConstraint(SDValue Op, std::string &Constraint,
                                            std::vector<SDValue> &Ops,
                                            SelectionDAG &DAG) const;

  //===--------------------------------------------------------------------===//
  // Instruction Emitting Hooks
  //

  // EmitInstrWithCustomInserter - This method should be implemented by targets
  // that mark instructions with the 'usesCustomInserter' flag.  These
  // instructions are special in various ways, which require special support to
  // insert.  The specified MachineInstr is created but not inserted into any
  // basic blocks, and this method is called to expand it into a sequence of
  // instructions, potentially also creating new basic blocks and control flow.
  virtual MachineBasicBlock *
    EmitInstrWithCustomInserter(MachineInstr *MI, MachineBasicBlock *MBB) const;

  /// AdjustInstrPostInstrSelection - This method should be implemented by
  /// targets that mark instructions with the 'hasPostISelHook' flag. These
  /// instructions must be adjusted after instruction selection by target hooks.
  /// e.g. To fill in optional defs for ARM 's' setting instructions.
  virtual void
  AdjustInstrPostInstrSelection(MachineInstr *MI, SDNode *Node) const;

  //===--------------------------------------------------------------------===//
  // Addressing mode description hooks (used by LSR etc).
  //

  /// AddrMode - This represents an addressing mode of:
  ///    BaseGV + BaseOffs + BaseReg + Scale*ScaleReg
  /// If BaseGV is null,  there is no BaseGV.
  /// If BaseOffs is zero, there is no base offset.
  /// If HasBaseReg is false, there is no base register.
  /// If Scale is zero, there is no ScaleReg.  Scale of 1 indicates a reg with
  /// no scale.
  ///
  struct AddrMode {
    GlobalValue *BaseGV;
    int64_t      BaseOffs;
    bool         HasBaseReg;
    int64_t      Scale;
    AddrMode() : BaseGV(0), BaseOffs(0), HasBaseReg(false), Scale(0) {}
  };

  /// isLegalAddressingMode - Return true if the addressing mode represented by
  /// AM is legal for this target, for a load/store of the specified type.
  /// The type may be VoidTy, in which case only return true if the addressing
  /// mode is legal for a load/store of any legal type.
  /// TODO: Handle pre/postinc as well.
  virtual bool isLegalAddressingMode(const AddrMode &AM, Type *Ty) const;

  /// isLegalICmpImmediate - Return true if the specified immediate is legal
  /// icmp immediate, that is the target has icmp instructions which can compare
  /// a register against the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalICmpImmediate(int64_t) const {
    return true;
  }

  /// isLegalAddImmediate - Return true if the specified immediate is legal
  /// add immediate, that is the target has add instructions which can add
  /// a register with the immediate without having to materialize the
  /// immediate into a register.
  virtual bool isLegalAddImmediate(int64_t) const {
    return true;
  }

  /// isTruncateFree - Return true if it's free to truncate a value of
  /// type Ty1 to type Ty2. e.g. On x86 it's free to truncate a i32 value in
  /// register EAX to i16 by referencing its sub-register AX.
  virtual bool isTruncateFree(Type * /*Ty1*/, Type * /*Ty2*/) const {
    return false;
  }

  virtual bool isTruncateFree(EVT /*VT1*/, EVT /*VT2*/) const {
    return false;
  }

  /// isZExtFree - Return true if any actual instruction that defines a
  /// value of type Ty1 implicitly zero-extends the value to Ty2 in the result
  /// register. This does not necessarily include registers defined in
  /// unknown ways, such as incoming arguments, or copies from unknown
  /// virtual registers. Also, if isTruncateFree(Ty2, Ty1) is true, this
  /// does not necessarily apply to truncate instructions. e.g. on x86-64,
  /// all instructions that define 32-bit values implicit zero-extend the
  /// result out to 64 bits.
  virtual bool isZExtFree(Type * /*Ty1*/, Type * /*Ty2*/) const {
    return false;
  }

  virtual bool isZExtFree(EVT /*VT1*/, EVT /*VT2*/) const {
    return false;
  }

  /// isNarrowingProfitable - Return true if it's profitable to narrow
  /// operations of type VT1 to VT2. e.g. on x86, it's profitable to narrow
  /// from i32 to i8 but not from i32 to i16.
  virtual bool isNarrowingProfitable(EVT /*VT1*/, EVT /*VT2*/) const {
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Div utility functions
  //
  SDValue BuildExactSDIV(SDValue Op1, SDValue Op2, DebugLoc dl,
                         SelectionDAG &DAG) const;
  SDValue BuildSDIV(SDNode *N, SelectionDAG &DAG,
                      std::vector<SDNode*>* Created) const;
  SDValue BuildUDIV(SDNode *N, SelectionDAG &DAG,
                      std::vector<SDNode*>* Created) const;


  //===--------------------------------------------------------------------===//
  // Runtime Library hooks
  //

  /// setLibcallName - Rename the default libcall routine name for the specified
  /// libcall.
  void setLibcallName(RTLIB::Libcall Call, const char *Name) {
    LibcallRoutineNames[Call] = Name;
  }

  /// getLibcallName - Get the libcall routine name for the specified libcall.
  ///
  const char *getLibcallName(RTLIB::Libcall Call) const {
    return LibcallRoutineNames[Call];
  }

  /// setCmpLibcallCC - Override the default CondCode to be used to test the
  /// result of the comparison libcall against zero.
  void setCmpLibcallCC(RTLIB::Libcall Call, ISD::CondCode CC) {
    CmpLibcallCCs[Call] = CC;
  }

  /// getCmpLibcallCC - Get the CondCode that's to be used to test the result of
  /// the comparison libcall against zero.
  ISD::CondCode getCmpLibcallCC(RTLIB::Libcall Call) const {
    return CmpLibcallCCs[Call];
  }

  /// setLibcallCallingConv - Set the CallingConv that should be used for the
  /// specified libcall.
  void setLibcallCallingConv(RTLIB::Libcall Call, CallingConv::ID CC) {
    LibcallCallingConvs[Call] = CC;
  }

  /// getLibcallCallingConv - Get the CallingConv that should be used for the
  /// specified libcall.
  CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return LibcallCallingConvs[Call];
  }

private:
  const TargetMachine &TM;
  const TargetData *TD;
  const TargetLoweringObjectFile &TLOF;

  /// We are in the process of implementing a new TypeLegalization action
  /// which is the promotion of vector elements. This feature is under
  /// development. Until this feature is complete, it is only enabled using a
  /// flag. We pass this flag using a member because of circular dep issues.
  /// This member will be removed with the flag once we complete the transition.
  bool mayPromoteElements;

  /// PointerTy - The type to use for pointers, usually i32 or i64.
  ///
  MVT PointerTy;

  /// IsLittleEndian - True if this is a little endian target.
  ///
  bool IsLittleEndian;

  /// SelectIsExpensive - Tells the code generator not to expand operations
  /// into sequences that use the select operations if possible.
  bool SelectIsExpensive;

  /// IntDivIsCheap - Tells the code generator not to expand integer divides by
  /// constants into a sequence of muls, adds, and shifts.  This is a hack until
  /// a real cost model is in place.  If we ever optimize for size, this will be
  /// set to true unconditionally.
  bool IntDivIsCheap;

  /// Pow2DivIsCheap - Tells the code generator that it shouldn't generate
  /// srl/add/sra for a signed divide by power of two, and let the target handle
  /// it.
  bool Pow2DivIsCheap;

  /// JumpIsExpensive - Tells the code generator that it shouldn't generate
  /// extra flow control instructions and should attempt to combine flow
  /// control instructions via predication.
  bool JumpIsExpensive;

  /// UseUnderscoreSetJmp - This target prefers to use _setjmp to implement
  /// llvm.setjmp.  Defaults to false.
  bool UseUnderscoreSetJmp;

  /// UseUnderscoreLongJmp - This target prefers to use _longjmp to implement
  /// llvm.longjmp.  Defaults to false.
  bool UseUnderscoreLongJmp;

  /// BooleanContents - Information about the contents of the high-bits in
  /// boolean values held in a type wider than i1.  See getBooleanContents.
  BooleanContent BooleanContents;
  /// BooleanVectorContents - Information about the contents of the high-bits
  /// in boolean vector values when the element type is wider than i1.  See
  /// getBooleanContents.
  BooleanContent BooleanVectorContents;

  /// SchedPreferenceInfo - The target scheduling preference: shortest possible
  /// total cycles or lowest register usage.
  Sched::Preference SchedPreferenceInfo;

  /// JumpBufSize - The size, in bytes, of the target's jmp_buf buffers
  unsigned JumpBufSize;

  /// JumpBufAlignment - The alignment, in bytes, of the target's jmp_buf
  /// buffers
  unsigned JumpBufAlignment;

  /// MinStackArgumentAlignment - The minimum alignment that any argument
  /// on the stack needs to have.
  ///
  unsigned MinStackArgumentAlignment;

  /// MinFunctionAlignment - The minimum function alignment (used when
  /// optimizing for size, and to prevent explicitly provided alignment
  /// from leading to incorrect code).
  ///
  unsigned MinFunctionAlignment;

  /// PrefFunctionAlignment - The preferred function alignment (used when
  /// alignment unspecified and optimizing for speed).
  ///
  unsigned PrefFunctionAlignment;

  /// PrefLoopAlignment - The preferred loop alignment.
  ///
  unsigned PrefLoopAlignment;

  /// ShouldFoldAtomicFences - Whether fencing MEMBARRIER instructions should
  /// be folded into the enclosed atomic intrinsic instruction by the
  /// combiner.
  bool ShouldFoldAtomicFences;

  /// InsertFencesForAtomic - Whether the DAG builder should automatically
  /// insert fences and reduce ordering for atomics.  (This will be set for
  /// for most architectures with weak memory ordering.)
  bool InsertFencesForAtomic;

  /// StackPointerRegisterToSaveRestore - If set to a physical register, this
  /// specifies the register that llvm.savestack/llvm.restorestack should save
  /// and restore.
  unsigned StackPointerRegisterToSaveRestore;

  /// ExceptionPointerRegister - If set to a physical register, this specifies
  /// the register that receives the exception address on entry to a landing
  /// pad.
  unsigned ExceptionPointerRegister;

  /// ExceptionSelectorRegister - If set to a physical register, this specifies
  /// the register that receives the exception typeid on entry to a landing
  /// pad.
  unsigned ExceptionSelectorRegister;

  /// RegClassForVT - This indicates the default register class to use for
  /// each ValueType the target supports natively.
  TargetRegisterClass *RegClassForVT[MVT::LAST_VALUETYPE];
  unsigned char NumRegistersForVT[MVT::LAST_VALUETYPE];
  EVT RegisterTypeForVT[MVT::LAST_VALUETYPE];

  /// RepRegClassForVT - This indicates the "representative" register class to
  /// use for each ValueType the target supports natively. This information is
  /// used by the scheduler to track register pressure. By default, the
  /// representative register class is the largest legal super-reg register
  /// class of the register class of the specified type. e.g. On x86, i8, i16,
  /// and i32's representative class would be GR32.
  const TargetRegisterClass *RepRegClassForVT[MVT::LAST_VALUETYPE];

  /// RepRegClassCostForVT - This indicates the "cost" of the "representative"
  /// register class for each ValueType. The cost is used by the scheduler to
  /// approximate register pressure.
  uint8_t RepRegClassCostForVT[MVT::LAST_VALUETYPE];

  /// TransformToType - For any value types we are promoting or expanding, this
  /// contains the value type that we are changing to.  For Expanded types, this
  /// contains one step of the expand (e.g. i64 -> i32), even if there are
  /// multiple steps required (e.g. i64 -> i16).  For types natively supported
  /// by the system, this holds the same type (e.g. i32 -> i32).
  EVT TransformToType[MVT::LAST_VALUETYPE];

  /// OpActions - For each operation and each value type, keep a LegalizeAction
  /// that indicates how instruction selection should deal with the operation.
  /// Most operations are Legal (aka, supported natively by the target), but
  /// operations that are not should be described.  Note that operations on
  /// non-legal value types are not described here.
  uint8_t OpActions[MVT::LAST_VALUETYPE][ISD::BUILTIN_OP_END];

  /// LoadExtActions - For each load extension type and each value type,
  /// keep a LegalizeAction that indicates how instruction selection should deal
  /// with a load of a specific value type and extension type.
  uint8_t LoadExtActions[MVT::LAST_VALUETYPE][ISD::LAST_LOADEXT_TYPE];

  /// TruncStoreActions - For each value type pair keep a LegalizeAction that
  /// indicates whether a truncating store of a specific value type and
  /// truncating type is legal.
  uint8_t TruncStoreActions[MVT::LAST_VALUETYPE][MVT::LAST_VALUETYPE];

  /// IndexedModeActions - For each indexed mode and each value type,
  /// keep a pair of LegalizeAction that indicates how instruction
  /// selection should deal with the load / store.  The first dimension is the
  /// value_type for the reference. The second dimension represents the various
  /// modes for load store.
  uint8_t IndexedModeActions[MVT::LAST_VALUETYPE][ISD::LAST_INDEXED_MODE];

  /// CondCodeActions - For each condition code (ISD::CondCode) keep a
  /// LegalizeAction that indicates how instruction selection should
  /// deal with the condition code.
  uint64_t CondCodeActions[ISD::SETCC_INVALID];

  ValueTypeActionImpl ValueTypeActions;

  typedef std::pair<LegalizeTypeAction, EVT> LegalizeKind;

  LegalizeKind
  getTypeConversion(LLVMContext &Context, EVT VT) const {
    // If this is a simple type, use the ComputeRegisterProp mechanism.
    if (VT.isSimple()) {
      assert((unsigned)VT.getSimpleVT().SimpleTy <
             array_lengthof(TransformToType));
      EVT NVT = TransformToType[VT.getSimpleVT().SimpleTy];
      LegalizeTypeAction LA = ValueTypeActions.getTypeAction(VT.getSimpleVT());

      assert(
        (!(NVT.isSimple() && LA != TypeLegal) ||
         ValueTypeActions.getTypeAction(NVT.getSimpleVT()) != TypePromoteInteger)
         && "Promote may not follow Expand or Promote");

      return LegalizeKind(LA, NVT);
    }

    // Handle Extended Scalar Types.
    if (!VT.isVector()) {
      assert(VT.isInteger() && "Float types must be simple");
      unsigned BitSize = VT.getSizeInBits();
      // First promote to a power-of-two size, then expand if necessary.
      if (BitSize < 8 || !isPowerOf2_32(BitSize)) {
        EVT NVT = VT.getRoundIntegerType(Context);
        assert(NVT != VT && "Unable to round integer VT");
        LegalizeKind NextStep = getTypeConversion(Context, NVT);
        // Avoid multi-step promotion.
        if (NextStep.first == TypePromoteInteger) return NextStep;
        // Return rounded integer type.
        return LegalizeKind(TypePromoteInteger, NVT);
      }

      return LegalizeKind(TypeExpandInteger,
                          EVT::getIntegerVT(Context, VT.getSizeInBits()/2));
    }

    // Handle vector types.
    unsigned NumElts = VT.getVectorNumElements();
    EVT EltVT = VT.getVectorElementType();

    // Vectors with only one element are always scalarized.
    if (NumElts == 1)
      return LegalizeKind(TypeScalarizeVector, EltVT);

    // If we allow the promotion of vector elements using a flag,
    // then try to widen vector elements until a legal type is found.
    if (mayPromoteElements && EltVT.isInteger()) {
      // Vectors with a number of elements that is not a power of two are always
      // widened, for example <3 x float> -> <4 x float>.
      if (!VT.isPow2VectorType()) {
        NumElts = (unsigned)NextPowerOf2(NumElts);
        EVT NVT = EVT::getVectorVT(Context, EltVT, NumElts);
        return LegalizeKind(TypeWidenVector, NVT);
      }

      // Examine the element type.
      LegalizeKind LK = getTypeConversion(Context, EltVT);

      // If type is to be expanded, split the vector.
      //  <4 x i140> -> <2 x i140>
      if (LK.first == TypeExpandInteger)
        return LegalizeKind(TypeSplitVector,
                            EVT::getVectorVT(Context, EltVT, NumElts / 2));

      // Promote the integer element types until a legal vector type is found
      // or until the element integer type is too big. If a legal type was not
      // found, fallback to the usual mechanism of widening/splitting the
      // vector.
      while (1) {
        // Increase the bitwidth of the element to the next pow-of-two
        // (which is greater than 8 bits).
        EltVT = EVT::getIntegerVT(Context, 1 + EltVT.getSizeInBits()
                                 ).getRoundIntegerType(Context);

        // Stop trying when getting a non-simple element type.
        // Note that vector elements may be greater than legal vector element
        // types. Example: X86 XMM registers hold 64bit element on 32bit systems.
        if (!EltVT.isSimple()) break;

        // Build a new vector type and check if it is legal.
        MVT NVT = MVT::getVectorVT(EltVT.getSimpleVT(), NumElts);
        // Found a legal promoted vector type.
        if (NVT != MVT() && ValueTypeActions.getTypeAction(NVT) == TypeLegal)
          return LegalizeKind(TypePromoteInteger,
                              EVT::getVectorVT(Context, EltVT, NumElts));
      }
    }

    // Try to widen the vector until a legal type is found.
    // If there is no wider legal type, split the vector.
    while (1) {
      // Round up to the next power of 2.
      NumElts = (unsigned)NextPowerOf2(NumElts);

      // If there is no simple vector type with this many elements then there
      // cannot be a larger legal vector type.  Note that this assumes that
      // there are no skipped intermediate vector types in the simple types.
      if (!EltVT.isSimple()) break;
      MVT LargerVector = MVT::getVectorVT(EltVT.getSimpleVT(), NumElts);
      if (LargerVector == MVT()) break;

      // If this type is legal then widen the vector.
      if (ValueTypeActions.getTypeAction(LargerVector) == TypeLegal)
        return LegalizeKind(TypeWidenVector, LargerVector);
    }

    // Widen odd vectors to next power of two.
    if (!VT.isPow2VectorType()) {
      EVT NVT = VT.getPow2VectorType(Context);
      return LegalizeKind(TypeWidenVector, NVT);
    }

    // Vectors with illegal element types are expanded.
    EVT NVT = EVT::getVectorVT(Context, EltVT, VT.getVectorNumElements() / 2);
    return LegalizeKind(TypeSplitVector, NVT);

    assert(false && "Unable to handle this kind of vector type");
    return LegalizeKind(TypeLegal, VT);
  }

  std::vector<std::pair<EVT, TargetRegisterClass*> > AvailableRegClasses;

  /// TargetDAGCombineArray - Targets can specify ISD nodes that they would
  /// like PerformDAGCombine callbacks for by calling setTargetDAGCombine(),
  /// which sets a bit in this array.
  unsigned char
  TargetDAGCombineArray[(ISD::BUILTIN_OP_END+CHAR_BIT-1)/CHAR_BIT];

  /// PromoteToType - For operations that must be promoted to a specific type,
  /// this holds the destination type.  This map should be sparse, so don't hold
  /// it as an array.
  ///
  /// Targets add entries to this map with AddPromotedToType(..), clients access
  /// this with getTypeToPromoteTo(..).
  std::map<std::pair<unsigned, MVT::SimpleValueType>, MVT::SimpleValueType>
    PromoteToType;

  /// LibcallRoutineNames - Stores the name each libcall.
  ///
  const char *LibcallRoutineNames[RTLIB::UNKNOWN_LIBCALL];

  /// CmpLibcallCCs - The ISD::CondCode that should be used to test the result
  /// of each of the comparison libcall against zero.
  ISD::CondCode CmpLibcallCCs[RTLIB::UNKNOWN_LIBCALL];

  /// LibcallCallingConvs - Stores the CallingConv that should be used for each
  /// libcall.
  CallingConv::ID LibcallCallingConvs[RTLIB::UNKNOWN_LIBCALL];

protected:
  /// When lowering \@llvm.memset this field specifies the maximum number of
  /// store operations that may be substituted for the call to memset. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memset will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, storing 9 bytes on a 32-bit machine
  /// with 16-bit alignment would result in four 2-byte stores and one 1-byte
  /// store.  This only applies to setting a constant array of a constant size.
  /// @brief Specify maximum number of store instructions per memset call.
  unsigned maxStoresPerMemset;

  /// Maximum number of stores operations that may be substituted for the call
  /// to memset, used for functions with OptSize attribute.
  unsigned maxStoresPerMemsetOptSize;

  /// When lowering \@llvm.memcpy this field specifies the maximum number of
  /// store operations that may be substituted for a call to memcpy. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memcpy will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, storing 7 bytes on a 32-bit machine
  /// with 32-bit alignment would result in one 4-byte store, a one 2-byte store
  /// and one 1-byte store. This only applies to copying a constant array of
  /// constant size.
  /// @brief Specify maximum bytes of store instructions per memcpy call.
  unsigned maxStoresPerMemcpy;

  /// Maximum number of store operations that may be substituted for a call
  /// to memcpy, used for functions with OptSize attribute.
  unsigned maxStoresPerMemcpyOptSize;

  /// When lowering \@llvm.memmove this field specifies the maximum number of
  /// store instructions that may be substituted for a call to memmove. Targets
  /// must set this value based on the cost threshold for that target. Targets
  /// should assume that the memmove will be done using as many of the largest
  /// store operations first, followed by smaller ones, if necessary, per
  /// alignment restrictions. For example, moving 9 bytes on a 32-bit machine
  /// with 8-bit alignment would result in nine 1-byte stores.  This only
  /// applies to copying a constant array of constant size.
  /// @brief Specify maximum bytes of store instructions per memmove call.
  unsigned maxStoresPerMemmove;

  /// Maximum number of store instructions that may be substituted for a call
  /// to memmove, used for functions with OpSize attribute.
  unsigned maxStoresPerMemmoveOptSize;

  /// This field specifies whether the target can benefit from code placement
  /// optimization.
  bool benefitFromCodePlacementOpt;

private:
  /// isLegalRC - Return true if the value types that can be represented by the
  /// specified register class are all legal.
  bool isLegalRC(const TargetRegisterClass *RC) const;

  /// hasLegalSuperRegRegClasses - Return true if the specified register class
  /// has one or more super-reg register classes that are legal.
  bool hasLegalSuperRegRegClasses(const TargetRegisterClass *RC) const;
};

/// GetReturnInfo - Given an LLVM IR type and return type attributes,
/// compute the return value EVTs and flags, and optionally also
/// the offsets, if the return value is being lowered to memory.
void GetReturnInfo(Type* ReturnType, Attributes attr,
                   SmallVectorImpl<ISD::OutputArg> &Outs,
                   const TargetLowering &TLI,
                   SmallVectorImpl<uint64_t> *Offsets = 0);

} // end llvm namespace

#endif
