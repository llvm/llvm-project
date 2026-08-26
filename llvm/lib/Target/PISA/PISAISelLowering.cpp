//===-- PISAISelLowering.cpp - PISA DAG Lowering Impl ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAISelLowering.h"
#include "PISA.h"
#include "PISACacheCtrlMMRA.h"
#include "PISASubtarget.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsPISA.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/PISAAddrSpace.h"

#include <type_traits>

#define DEBUG_TYPE "pisa-lower"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Atomic legalization
//
// Atomics are legalized by LLVM's AtomicExpandPass, driven entirely through the
// TargetLowering hooks below. These tables and helpers describe which native
// atomic operations PISA supports and which operations need IR expansion.
//===----------------------------------------------------------------------===//
namespace {
using namespace llvm::PISAAS;

constexpr unsigned PrivateAS = static_cast<unsigned>(AddressSpace::PRIVATE);
constexpr unsigned GenericAS = static_cast<unsigned>(AddressSpace::GENERIC);
constexpr unsigned SharedAS = static_cast<unsigned>(AddressSpace::SHARED);
constexpr unsigned GlobalAS = static_cast<unsigned>(AddressSpace::GLOBAL);

// clang-format off
const bool SupportedGlobal[AtomicRMWInst::LAST_BINOP + 1][4] = {
  // 16b    32b    64b   128b
  { true,  true,  true,  true}, // Xchg
  { true,  true,  true, false}, // Add
  { true,  true,  true, false}, // Sub
  { true,  true,  true, false}, // And
  { true,  true,  true, false}, // Nand
  { true,  true,  true, false}, // Or
  { true,  true,  true, false}, // Xor
  { true,  true,  true, false}, // Max
  { true,  true,  true, false}, // Min
  { true,  true,  true, false}, // Umax
  { true,  true,  true, false}, // Umin
  { true,  true,  true, false}, // FAdd
  { true,  true,  true, false}, // FSub
  { true,  true, false, false}, // FMax
  { true,  true, false, false}, // FMin
  {false, false, false, false}, // FMaximum
  {false, false, false, false}, // FMinimum
  {false, false, false, false}, // FMaximumNum
  {false, false, false, false}, // FMinimumNum
  {false,  true,  true, false}, // Uinc_wrap
  {false,  true,  true, false}, // Udec_wrap
  {false, false, false, false}, // Usub_cond
  {false, false, false, false}, // Usub_sat
};
// clang-format on

constexpr uint8_t SupportedGlobalCols =
    std::extent<decltype(SupportedGlobal), 1>::value;
constexpr uint8_t SupportedGlobalRows =
    std::extent<decltype(SupportedGlobal), 0>::value;

// clang-format off
const bool SupportedShared[AtomicRMWInst::LAST_BINOP + 1][4] = {
  // 16b    32b    64b    128b
  { true,  true, false,  true}, // Xchg
  { true,  true, false, false}, // Add
  { true,  true, false, false}, // Sub
  { true,  true, false, false}, // And
  { true,  true, false, false}, // Nand
  { true,  true, false, false}, // Or
  { true,  true, false, false}, // Xor
  { true,  true, false, false}, // Max
  { true,  true, false, false}, // Min
  { true,  true, false, false}, // Umax
  { true,  true, false, false}, // Umin
  { true,  true, false, false}, // FAdd
  { true,  true, false, false}, // FSub
  { true,  true, false, false}, // FMax
  { true,  true, false, false}, // FMin
  {false, false, false, false}, // FMaximum
  {false, false, false, false}, // FMinimum
  {false, false, false, false}, // FMaximumNum
  {false, false, false, false}, // FMinimumNum
  {false,  true, false, false}, // Uinc_wrap
  {false,  true, false, false}, // Udec_wrap
  {false, false, false, false}, // Usub_cond
  {false, false, false, false}, // Usub_sat
};
// clang-format on

constexpr uint8_t SupportedSharedCols =
    std::extent<decltype(SupportedShared), 1>::value;
constexpr uint8_t SupportedSharedRows =
    std::extent<decltype(SupportedShared), 0>::value;

const bool SupportedLoadStoreGlobal[4] = {
    true, // 16b
    true, // 32b
    true, // 64b
    true, // 128b
};

const bool SupportedLoadStoreShared[4] = {
    true,  // 16b
    true,  // 32b
    false, // 64b
    true,  // 128b
};

bool isLegalLoadStore(unsigned BitWidth, unsigned AddrSpace) {
  // Don't legalize loads/stores of unsupported bitwidths
  if (BitWidth < 16 || BitWidth > 128)
    return true;

  unsigned TableCol = Log2_32(BitWidth / 16);
  switch (AddrSpace) {
  case GlobalAS:
    return SupportedLoadStoreGlobal[TableCol];
  case SharedAS:
    return SupportedLoadStoreShared[TableCol];
  case GenericAS:
    return SupportedLoadStoreGlobal[TableCol] &&
           SupportedLoadStoreShared[TableCol];
  }

  // Don't legalize loads/stores in other address spaces
  return false;
}

bool isAtomicRMWLegal(const AtomicRMWInst *A) {
  Type *Ty = A->getType();
  const DataLayout &DL = A->getDataLayout();
  unsigned BitWidth = DL.getTypeSizeInBits(Ty);
  if (BitWidth < 16 || BitWidth > 128)
    return false;
  unsigned TableCol = Log2_32(BitWidth / 16);
  unsigned Op = A->getOperation();

  unsigned AS = A->getPointerAddressSpace();
  if (((AS != SharedAS) &&
       (TableCol >= SupportedGlobalCols || Op >= SupportedGlobalRows)) ||
      ((AS != GlobalAS) &&
       (TableCol >= SupportedSharedCols || Op >= SupportedSharedRows)))
    return false;
  switch (AS) {
  default:
    return false;
  case GlobalAS:
    return SupportedGlobal[Op][TableCol];
  case SharedAS:
    return SupportedShared[Op][TableCol];
  case GenericAS:
    return SupportedGlobal[Op][TableCol] && SupportedShared[Op][TableCol];
  }
}

TargetLowering::AtomicExpansionKind
computeRMWExpansion(const AtomicRMWInst *A) {
  using Kind = TargetLowering::AtomicExpansionKind;
  switch (A->getPointerAddressSpace()) {
  case PrivateAS:
    return Kind::NotAtomic; // load-op-store, single work-item
  case GenericAS:
    // Legal in BOTH global and shared -> native generic; else runtime dispatch.
    return isAtomicRMWLegal(A) ? Kind::None : Kind::CustomExpand;
  case GlobalAS:
  case SharedAS:
    return isAtomicRMWLegal(A) ? Kind::None : Kind::CmpXChg;
  default:
    return Kind::None;
  }
}

SyncScope::ID hoistedFenceScope(Instruction *Inst) {
  // An AtomicRMWInst (the CAS-loop op, incl. the xchg from an expanded store)
  // is what reaches here today, but stay robust to other atomic instructions.
  SyncScope::ID SSID = getAtomicSyncScopeID(Inst).value_or(SyncScope::System);
  // Shared memory has no meaningful system scope, so narrow the scope used for
  // hoisted fences to the widest valid shared-memory scope.
  if (const auto *RMW = dyn_cast<AtomicRMWInst>(Inst))
    if (RMW->getPointerAddressSpace() == SharedAS && SSID == SyncScope::System)
      return Inst->getContext().getOrInsertSyncScopeID("gpu-shared");
  return SSID;
}
} // namespace

PISATargetLowering::PISATargetLowering(const TargetMachine &TM,
                                       const PISASubtarget &STI)
    : TargetLowering(TM, STI) {
  // Route atomics through AtomicExpandPass. PISA supports up to
  // 128-bit atomics; without this every atomic wider than the default falls
  // back to an unsupported __atomic_* libcall.
  setMaxAtomicSizeInBitsSupported(128);

  // these numbers need to be large enough to cover cases that are not
  // expanded by PISAExpandIntrinsics into a ld-st loop.
  MaxStoresPerMemcpy = 64;
  MaxStoresPerMemmove = 64;
  MaxStoresPerMemset = 64;

  // map int types to registers classes
  addRegisterClass(MVT::i8, &PISA::Reg8bRegClass);
  addRegisterClass(MVT::i16, &PISA::Reg16bRegClass);
  addRegisterClass(MVT::i32, &PISA::Reg32bRegClass);
  addRegisterClass(MVT::i64, &PISA::Reg64bRegClass);
  addRegisterClass(MVT::bf16, &PISA::Reg16bRegClass);
  addRegisterClass(MVT::f16, &PISA::Reg16bRegClass);
  addRegisterClass(MVT::f32, &PISA::Reg32bRegClass);
  addRegisterClass(MVT::f64, &PISA::Reg64bRegClass);
  addRegisterClass(MVT::v2i8, &PISA::RegV2_8bRegClass);
  addRegisterClass(MVT::v3i8, &PISA::RegV3_8bRegClass);
  addRegisterClass(MVT::v4i8, &PISA::RegV4_8bRegClass);
  addRegisterClass(MVT::v2i16, &PISA::RegV2_16bRegClass);
  addRegisterClass(MVT::v3i16, &PISA::RegV3_16bRegClass);
  addRegisterClass(MVT::v4i16, &PISA::RegV4_16bRegClass);
  addRegisterClass(MVT::v2i32, &PISA::RegV2_32bRegClass);
  addRegisterClass(MVT::v3i32, &PISA::RegV3_32bRegClass);
  addRegisterClass(MVT::v4i32, &PISA::RegV4_32bRegClass);
  addRegisterClass(MVT::v5i32, &PISA::RegV5_32bRegClass);
  addRegisterClass(MVT::v6i32, &PISA::RegV6_32bRegClass);
  addRegisterClass(MVT::v7i32, &PISA::RegV7_32bRegClass);
  addRegisterClass(MVT::v8i32, &PISA::RegV8_32bRegClass);
  addRegisterClass(MVT::v16i32, &PISA::RegV16_32bRegClass);
  addRegisterClass(MVT::v32i32, &PISA::RegV32_32bRegClass);
  addRegisterClass(MVT::v64i32, &PISA::RegV64_32bRegClass);
  addRegisterClass(MVT::v2i64, &PISA::RegV2_64bRegClass);
  addRegisterClass(MVT::v3i64, &PISA::RegV3_64bRegClass);
  addRegisterClass(MVT::v4i64, &PISA::RegV4_64bRegClass);
  // map floating-point types to registers classes
  addRegisterClass(MVT::v2f16, &PISA::RegV2_16bRegClass);
  addRegisterClass(MVT::v3f16, &PISA::RegV3_16bRegClass);
  addRegisterClass(MVT::v4f16, &PISA::RegV4_16bRegClass);
  addRegisterClass(MVT::v2bf16, &PISA::RegV2_16bRegClass);
  addRegisterClass(MVT::v3bf16, &PISA::RegV3_16bRegClass);
  addRegisterClass(MVT::v4bf16, &PISA::RegV4_16bRegClass);
  addRegisterClass(MVT::v2f32, &PISA::RegV2_32bRegClass);
  addRegisterClass(MVT::v3f32, &PISA::RegV3_32bRegClass);
  addRegisterClass(MVT::v4f32, &PISA::RegV4_32bRegClass);
  addRegisterClass(MVT::v5f32, &PISA::RegV5_32bRegClass);
  addRegisterClass(MVT::v6f32, &PISA::RegV6_32bRegClass);
  addRegisterClass(MVT::v7f32, &PISA::RegV7_32bRegClass);
  addRegisterClass(MVT::v8f32, &PISA::RegV8_32bRegClass);
  addRegisterClass(MVT::v2f64, &PISA::RegV2_64bRegClass);
  addRegisterClass(MVT::v3f64, &PISA::RegV3_64bRegClass);
  addRegisterClass(MVT::v4f64, &PISA::RegV4_64bRegClass);
  // must be done after all classes are added
  computeRegisterProperties(STI.getRegisterInfo());

  // Jump is Expensive. Don't create extra control flow for 'and', 'or'
  // condition branches.
  setJumpIsExpensive(true);
  setMaxDivRemBitWidthSupported(64);
}

unsigned PISATargetLowering::getNumRegistersForCallingConv(LLVMContext &Context,
                                                           CallingConv::ID CC,
                                                           EVT VT) const {
  // This code avoids CallLowering fail inside getVectorTypeBreakdown
  // on v3i1 arguments. Maybe we need to return 1 for all types.
  // TODO: remove it once this case is supported by the default implementation.
  if (VT.isVector() && VT.getVectorNumElements() == 3 &&
      (VT.getVectorElementType() == MVT::i1 ||
       VT.getVectorElementType() == MVT::i8))
    return 1;
  return getNumRegisters(Context, VT);
}

bool PISATargetLowering::isCheapToSpeculateCttz(Type *Ty) const { return true; }

bool PISATargetLowering::isCheapToSpeculateCtlz(Type *Ty) const { return true; }

bool PISATargetLowering::isReassocProfitable(MachineRegisterInfo &MRI,
                                             Register N0, Register N1) const {
  auto GetOneDefInst = [&MRI](Register N) -> MachineInstr * {
    auto *Def = MRI.getOneDef(N);
    if (Def)
      return Def->getParent();
    return nullptr;
  };

  auto *I0 = GetOneDefInst(N0);
  auto *I1 = GetOneDefInst(N1);
  if (I0 && I1) {
    // Prevent reassociating the following pattern
    //  (add (mul a, b), (add (mul c, d), e))
    // into
    //  (add (add (mul a, b), (mul c, d)), e)
    // so that more 'mad's could be selected.
    if (I0->getOpcode() != TargetOpcode::G_MUL)
      std::swap(I0, I1);
    if (I0->getOpcode() == TargetOpcode::G_MUL &&
        I1->getOpcode() == TargetOpcode::G_ADD) {
      auto *NI0 = GetOneDefInst(I1->getOperand(1).getReg());
      auto *NI1 = GetOneDefInst(I1->getOperand(2).getReg());
      if (NI0 && NI1 &&
          (NI0->getOpcode() == TargetOpcode::G_MUL ||
           NI1->getOpcode() == TargetOpcode::G_MUL)) {
        // Don't reassociate to break the selection of MAD.
        return false;
      }
    }
  }

  return TargetLowering::isReassocProfitable(MRI, N0, N1);
}

MVT PISATargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                      CallingConv::ID CC,
                                                      EVT VT) const {
  // This code avoids CallLowering fail inside getVectorTypeBreakdown
  // on v3i1 arguments. Maybe we need to return i32 for all types.
  // TODO: remove it once this case is supported by the default implementation.
  if (VT.isVector() && VT.getVectorNumElements() == 3) {
    if (VT.getVectorElementType() == MVT::i1)
      return MVT::v4i1;
    if (VT.getVectorElementType() == MVT::i8)
      return MVT::v4i8;
  }
  return getRegisterType(Context, VT);
}

void PISATargetLowering::getTgtMemIntrinsic(
    SmallVectorImpl<IntrinsicInfo> &Infos, const CallBase &I,
    MachineFunction &MF, unsigned Intrinsic) const {
  IntrinsicInfo Info;
  Info.flags = MachineMemOperand::MONone;
  switch (Intrinsic) {
  case Intrinsic::pisa_cas_fatom:
    Info.memVT = MVT::getVT(I.getType());
    Info.ptrVal = I.getArgOperand(0);
    Info.align.reset();
    Info.flags |= MachineMemOperand::MOLoad | MachineMemOperand::MOStore;
    Info.flags |= getTargetMMOFlags(I);
    // syncscope("<target-scope>") support for atomics
    if (auto *ConstInt = dyn_cast<ConstantInt>(I.getArgOperand(3)))
      Info.order = static_cast<llvm::AtomicOrdering>(ConstInt->getZExtValue());
    Infos.push_back(Info);
    return;
  default:
    break;
  }
  return;
}

LLT PISATargetLowering::getOptimalMemOpLLT(
    const MemOp &Op, const AttributeList &FuncAttributes) const {
  auto I8 = LLT::integer(8);
  auto I16 = LLT::integer(16);
  auto I32 = LLT::integer(32);
  auto I64 = LLT::integer(64);

  if (Op.size() >= 16 && Op.isAligned(Align(8)))
    return LLT::fixed_vector(2, I64);
  if (Op.size() >= 16 && Op.isAligned(Align(4)))
    return LLT::fixed_vector(4, I32);
  if (Op.size() >= 12 && Op.isAligned(Align(4)))
    return LLT::fixed_vector(3, I32);
  if (Op.size() >= 8 && Op.isAligned(Align(4)))
    return LLT::fixed_vector(2, I32);
  if (Op.size() >= 8 && Op.isAligned(Align(2)))
    return LLT::fixed_vector(4, I16);
  if (Op.size() >= 4 && Op.isAligned(Align(4)))
    return I32;
  if (Op.size() >= 4 && Op.isAligned(Align(2)))
    return LLT::fixed_vector(2, I16);
  if (Op.size() >= 4 && Op.isAligned(Align(1)))
    return LLT::fixed_vector(4, I8);
  if (Op.size() >= 2 && Op.isAligned(Align(2)))
    return I16;
  if (Op.size() >= 2 && Op.isAligned(Align(1)))
    return LLT::fixed_vector(2, I8);
  if (Op.size() >= 1 && Op.isAligned(Align(1)))
    return I8;
  return LLT();
}

bool PISATargetLowering::useFTZ(const MachineFunction &MF) const {
  return MF.getDenormalMode(APFloat::IEEEsingle()).Output ==
         DenormalMode::PreserveSign;
}

TargetLowering::ConstraintType
PISATargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      break;
    case 'P': // Predicate register.
      return C_RegisterClass;
    }
  }

  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass *>
PISATargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                                 StringRef Constraint,
                                                 MVT VT) const {
  using namespace PISA;
  static const TargetRegisterClass *Vector8BitClass[] = {
      &Reg8bRegClass, &RegV2_8bRegClass, &RegV3_8bRegClass, &RegV4_8bRegClass};
  static const TargetRegisterClass *Vector16BitClass[] = {
      &Reg16bRegClass, &RegV2_16bRegClass, &RegV3_16bRegClass,
      &RegV4_16bRegClass};
  static const TargetRegisterClass *Vector32BitClass[] = {
      &Reg32bRegClass,    &RegV2_32bRegClass, &RegV3_32bRegClass,
      &RegV4_32bRegClass, &RegV5_32bRegClass, &RegV6_32bRegClass,
      &RegV7_32bRegClass, &RegV8_32bRegClass,
  };
  static const TargetRegisterClass *Vector64BitClass[] = {
      &Reg64bRegClass, &RegV2_64bRegClass, &RegV3_64bRegClass,
      &RegV4_64bRegClass};

  if (Constraint.size() == 1) {
    const TargetRegisterClass *RC = nullptr;
    switch (Constraint[0]) {
    default:
      break;
    case 'P':
      RC = &PredRegClass;
      break;
    case 'r': {
      // FIXME: we already have a method to get the register class from LLT
      auto VectorSize = VT.isVector() ? VT.getVectorNumElements() : 1;
      auto ElementSize = VT.getScalarSizeInBits();
      assert(VectorSize >= 1 &&
             (ElementSize == 32 ? VectorSize <= 8 || VectorSize == 16 ||
                                      VectorSize == 32 || VectorSize == 64
                                : VectorSize <= 4));

      switch (ElementSize) {
      case 8:
        RC = Vector8BitClass[VectorSize - 1];
        break;
      case 16:
        RC = Vector16BitClass[VectorSize - 1];
        break;
      case 32:
        if (VectorSize == 64)
          RC = &RegV64_32bRegClass;
        else if (VectorSize == 32)
          RC = &RegV32_32bRegClass;
        else if (VectorSize == 16)
          RC = &RegV16_32bRegClass;
        else
          RC = Vector32BitClass[VectorSize - 1];
        break;
      case 64:
        RC = Vector64BitClass[VectorSize - 1];
        break;
      }
    } break;
    }

    if (RC)
      return std::make_pair(0u, RC);
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

MachineMemOperand::Flags
PISATargetLowering::getTargetMMOFlags(const Instruction &I) const {
  MachineMemOperand::Flags Flags = MachineMemOperand::MONone;
  if (auto Hint = PISA::getCacheCtrlFromMMRA(I)) {
    auto HintValue = *Hint & 0xF;
    Flags |= static_cast<MachineMemOperand::Flags>(HintValue << 6);
  }
  return Flags;
}

void PISATargetLowering::computeKnownBitsForTargetInstr(
    GISelValueTracking &Analysis, Register R, KnownBits &Known,
    const APInt &DemandedElts, const MachineRegisterInfo &MRI,
    unsigned Depth) const {
  MachineInstr *MI = MRI.getVRegDef(R);

  // As we go we can add more cases here for now only enable for
  // G_INTRINSIC for using for folding range attributes
  if (!MI || MI->getOpcode() != TargetOpcode::G_INTRINSIC)
    return;

  Intrinsic::ID IID = cast<GIntrinsic>(*MI).getIntrinsicID();

  if (Intrinsic::isOverloaded(IID))
    return;

  auto *Ctx = &MI->getMF()->getFunction().getContext();
  FunctionType *FT = Intrinsic::getType(*Ctx, IID);
  AttributeList Attrs = Intrinsic::getAttributes(*Ctx, IID, FT);

  if (Attrs.hasRetAttr(Attribute::Range)) {
    const ConstantRange &CR =
        Attrs.getRetAttr(Attribute::Range).getValueAsConstantRange();
    Known = CR.toKnownBits();
  }
}

//===----------------------------------------------------------------------===//
// Atomic legalization hooks
//===----------------------------------------------------------------------===//

TargetLowering::AtomicExpansionKind
PISATargetLowering::shouldExpandAtomicRMWInIR(const AtomicRMWInst *RMW) const {
  return computeRMWExpansion(RMW);
}

TargetLowering::AtomicExpansionKind
PISATargetLowering::shouldExpandAtomicCmpXchgInIR(
    const AtomicCmpXchgInst *CI) const {
  // Private memory is single work-item, so cmpxchg can become a plain
  // load/compare/conditional-store. Other address spaces keep cmpxchg semantics
  // and are left for native backend selection.
  return CI->getPointerAddressSpace() == PrivateAS
             ? AtomicExpansionKind::NotAtomic
             : AtomicExpansionKind::None;
}

TargetLowering::AtomicExpansionKind
PISATargetLowering::shouldExpandAtomicLoadInIR(LoadInst *LI) const {
  unsigned AS = LI->getPointerAddressSpace();
  if (AS == PrivateAS)
    return AtomicExpansionKind::NotAtomic; // plain load
  unsigned BW = LI->getDataLayout().getTypeSizeInBits(LI->getType());
  if (BW < 16 || BW > 128)
    return AtomicExpansionKind::None; // inverted vs rmw: leave native
  if (AS == GlobalAS || AS == SharedAS || AS == GenericAS) {
    if (isLegalLoadStore(BW, AS))
      return AtomicExpansionKind::None;
    // Illegal-width atomic load: emulate with an integer cmpxchg in
    // emitExpandAtomicLoad. AtomicExpand's generic CmpXChg path
    // (expandAtomicLoadToCmpXchg) is unsuitable for PISA:
    //   - Pointers: it casts only FP/vector to integer, so a pointer load
    //     stays a pointer cmpxchg, which asserts (isScalar) in IRTranslator.
    // emitExpandAtomicLoad instead emits a same-width integer cmpxchg that
    // preserves the syncscope, then casts the loaded bits back.
    return AtomicExpansionKind::CustomExpand;
  }
  return AtomicExpansionKind::None;
}

TargetLowering::AtomicExpansionKind
PISATargetLowering::shouldExpandAtomicStoreInIR(StoreInst *SI) const {
  unsigned AS = SI->getPointerAddressSpace();
  if (AS == PrivateAS)
    return AtomicExpansionKind::NotAtomic; // plain store
  Type *ValTy = SI->getValueOperand()->getType();
  unsigned BW = SI->getDataLayout().getTypeSizeInBits(ValTy);
  if (BW < 16 || BW > 128)
    return AtomicExpansionKind::None; // leave native
  if (AS == GlobalAS || AS == SharedAS || AS == GenericAS) {
    if (isLegalLoadStore(BW, AS))
      return AtomicExpansionKind::None;
    // Illegal-width atomic store: emulate with an integer xchg in
    // emitExpandAtomicStore. AtomicExpand's generic Expand path
    // (store -> xchg, expandAtomicStore) is unsuitable for PISA:
    //   - Fences: it re-expands via a direct tryExpandAtomicRMW that skips
    //     the driver's fence-hoisting path, so no leading fence is hoisted.
    //   - Pointers: it feeds a pointer value into the CAS loop, where
    //     createCmpXchgInstFun asserts on the pointer operand.
    // emitExpandAtomicStore casts the value to a same-width integer and
    // splits the block so the driver re-walks the xchg and applies fence
    // hoisting + gpu-shared narrowing, with the scope preserved.
    return AtomicExpansionKind::CustomExpand;
  }
  return AtomicExpansionKind::None;
}

bool PISATargetLowering::shouldInsertFencesForAtomic(
    const Instruction *I) const {
  // Use AtomicExpand fence splitting only for RMWs that become CAS loops.
  // Native atomics keep their ordering.
  if (const auto *RMW = dyn_cast<AtomicRMWInst>(I))
    return computeRMWExpansion(RMW) == AtomicExpansionKind::CmpXChg;
  return false;
}

Instruction *PISATargetLowering::emitLeadingFence(IRBuilderBase &Builder,
                                                  Instruction *Inst,
                                                  AtomicOrdering Ord) const {
  if (!isReleaseOrStronger(Ord))
    return nullptr;
  AtomicOrdering FenceOrd = (Ord == AtomicOrdering::SequentiallyConsistent)
                                ? Ord
                                : AtomicOrdering::Release;
  return Builder.CreateFence(FenceOrd, hoistedFenceScope(Inst));
}

Instruction *PISATargetLowering::emitTrailingFence(IRBuilderBase &Builder,
                                                   Instruction *Inst,
                                                   AtomicOrdering Ord) const {
  if (!isAcquireOrStronger(Ord))
    return nullptr;
  AtomicOrdering FenceOrd = (Ord == AtomicOrdering::SequentiallyConsistent)
                                ? Ord
                                : AtomicOrdering::Acquire;
  return Builder.CreateFence(FenceOrd, hoistedFenceScope(Inst));
}

void PISATargetLowering::emitExpandAtomicRMW(AtomicRMWInst *A) const {
  // Generic pointer in atomicrmw points to GLOBAL or SHARED (PRIVATE is UB per
  // OpenCL). Probe the real address space at runtime with pisa_isaddr_shared
  // and branch; AtomicExpandPass then re-walks and expands each arm
  // independently (global -> native, shared -> CAS loop + hoisted fences).
  LLVMContext &Ctx = A->getContext();
  Function *F = A->getFunction();
  IRBuilder<> IR(A);

  BasicBlock *OrigBB = A->getParent();
  BasicBlock *JoinBB = OrigBB->splitBasicBlock(A->getNextNode());

  Type *I32 = IR.getInt32Ty();
  Value *IsShared = IR.CreateIntrinsic(I32, Intrinsic::pisa_isaddr_shared,
                                       {A->getPointerOperand()});
  Value *SharedCmp = IR.CreateICmpNE(IsShared, ConstantInt::get(I32, 0));

  BasicBlock *GlobalBB = BasicBlock::Create(Ctx, "", F, JoinBB);
  BasicBlock *SharedBB = BasicBlock::Create(Ctx, "", F, JoinBB);

  IR.CreateCondBr(SharedCmp, SharedBB, GlobalBB);
  OrigBB->getTerminator()->eraseFromParent();

  IR.SetInsertPoint(JoinBB->begin());
  PHINode *Res = IR.CreatePHI(A->getType(), 2);

  auto EmitArm = [&](BasicBlock *BB, unsigned AS) {
    IR.SetInsertPoint(BB);
    Value *Cast = IR.CreateAddrSpaceCast(A->getPointerOperand(),
                                         PointerType::get(Ctx, AS));
    AtomicRMWInst *NewA = IR.CreateAtomicRMW(
        A->getOperation(), Cast, A->getValOperand(), A->getAlign(),
        A->getOrdering(), A->getSyncScopeID());
    Res->addIncoming(NewA, IR.GetInsertBlock());
    IR.CreateBr(JoinBB);
  };
  EmitArm(GlobalBB, GlobalAS);
  EmitArm(SharedBB, SharedAS);

  A->replaceAllUsesWith(Res);
  A->eraseFromParent();
}

void PISATargetLowering::emitExpandAtomicStore(StoreInst *SI) const {
  // Reached for every illegal-width atomic store (integer, FP, or pointer).
  // PISA's xchg/CAS expansion operates on integers, so cast the value to an
  // integer of the same width and rewrite the store as an integer atomicrmw
  // xchg while preserving the sync scope. Place it in a fresh block so the
  // AtomicExpand driver loop revisits and CAS-expands it with hoisted,
  // scope-narrowed fences (it does not re-walk ops created in the block
  // currently being processed).
  BasicBlock *BB = SI->getParent();
  BB->splitBasicBlock(SI->getIterator());

  IRBuilder<> IR(SI);
  Value *Val = SI->getValueOperand();
  Type *IntTy =
      IR.getIntNTy(SI->getDataLayout().getTypeSizeInBits(Val->getType()));
  Value *IntVal =
      Val->getType() == IntTy ? Val : IR.CreateBitOrPointerCast(Val, IntTy);
  AtomicOrdering Ord = SI->getOrdering() == AtomicOrdering::Unordered
                           ? AtomicOrdering::Monotonic
                           : SI->getOrdering();
  IR.CreateAtomicRMW(AtomicRMWInst::Xchg, SI->getPointerOperand(), IntVal,
                     SI->getAlign(), Ord, SI->getSyncScopeID());
  SI->eraseFromParent();
}

void PISATargetLowering::emitExpandAtomicLoad(LoadInst *LI) const {
  // Emulate an illegal-width atomic load with an integer cmpxchg (PISA's CAS
  // needs scalar operands) using zero for both compare and new value, then cast
  // the loaded bits back to the original type. The integer representation keeps
  // FP/pointer loads compatible with cmpxchg while preserving the sync scope.
  //
  // Note the deliberate load/store asymmetry: the load's emulating cmpxchg
  // keeps the original ordering and scope (it is not an AtomicRMWInst, so the
  // fence gate does not hoist/narrow it), whereas an illegal store becomes a
  // relaxed CAS loop bracketed by hoisted fences in emitExpandAtomicStore.
  IRBuilder<> IR(LI);
  Type *Ty = LI->getType();
  unsigned BW = LI->getDataLayout().getTypeSizeInBits(Ty);
  Type *IntTy = IR.getIntNTy(BW);
  AtomicOrdering Ord = LI->getOrdering() == AtomicOrdering::Unordered
                           ? AtomicOrdering::Monotonic
                           : LI->getOrdering();
  Constant *Zero = ConstantInt::get(IntTy, 0);
  Value *Pair = IR.CreateAtomicCmpXchg(
      LI->getPointerOperand(), Zero, Zero, LI->getAlign(), Ord,
      AtomicCmpXchgInst::getStrongestFailureOrdering(Ord),
      LI->getSyncScopeID());
  Value *Loaded = IR.CreateExtractValue(Pair, 0);
  if (Loaded->getType() != Ty)
    Loaded = IR.CreateBitOrPointerCast(Loaded, Ty);
  LI->replaceAllUsesWith(Loaded);
  LI->eraseFromParent();
}
