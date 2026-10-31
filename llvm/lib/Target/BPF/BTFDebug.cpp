//===- BTFDebug.cpp - BTF Generator ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing BTF debug info.
//
//===----------------------------------------------------------------------===//

#include "BTFDebug.h"
#include "BPF.h"
#include "BPFCORE.h"
#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "btf-debug"

#define GET_CC_REGISTER_LISTS
#include "BPFGenCallingConv.inc"

static const char *BTFKindStr[] = {
#define HANDLE_BTF_KIND(ID, NAME) "BTF_KIND_" #NAME,
#include "llvm/DebugInfo/BTF/BTF.def"
};

static const DIType *tryRemoveAtomicType(const DIType *Ty) {
  if (!Ty)
    return Ty;
  auto DerivedTy = dyn_cast<DIDerivedType>(Ty);
  if (DerivedTy && DerivedTy->getTag() == dwarf::DW_TAG_atomic_type)
    return DerivedTy->getBaseType();
  return Ty;
}

static const DIType *stripDITypeAttributes(const DIType *Ty) {
  while (const auto *DTy = dyn_cast_or_null<DIDerivedType>(Ty)) {
    switch (DTy->getTag()) {
    case dwarf::DW_TAG_atomic_type:
    case dwarf::DW_TAG_const_type:
    case dwarf::DW_TAG_restrict_type:
    case dwarf::DW_TAG_typedef:
    case dwarf::DW_TAG_volatile_type:
      Ty = DTy->getBaseType();
      break;
    default:
      return Ty;
    }
  }
  return Ty;
}

static bool sourceArgMatchesIRType(const DIType *SourceTy, Type *IRTy) {
  SourceTy = stripDITypeAttributes(SourceTy);

  // All pointers are opaque in LLVM IR, so any source-level pointer matches any
  // IR pointer regardless of pointee type.
  if (const auto *DTy = dyn_cast<DIDerivedType>(SourceTy))
    return DTy->getTag() == dwarf::DW_TAG_pointer_type && IRTy->isPointerTy();

  if (const auto *BTy = dyn_cast<DIBasicType>(SourceTy)) {
    uint64_t SizeInBits = BTy->getSizeInBits();
    if (BTy->getEncoding() == dwarf::DW_ATE_float)
      return IRTy->isFloatingPointTy() &&
             IRTy->getPrimitiveSizeInBits() == SizeInBits;
    // _Bool is 8 bits in DWARF/source but lowered to i1 in LLVM IR.
    if (BTy->getEncoding() == dwarf::DW_ATE_boolean && IRTy->isIntegerTy(1))
      return true;
    return IRTy->isIntegerTy(SizeInBits);
  }

  const auto *CTy = dyn_cast<DICompositeType>(SourceTy);
  if (!CTy)
    return false;

  switch (CTy->getTag()) {
  case dwarf::DW_TAG_enumeration_type:
    return IRTy->isIntegerTy(CTy->getSizeInBits());
  default:
    return false;
  }
}

/// Collect the physical register each source argument lives in by scanning
/// DBG_VALUE instructions in the entry block.  A DBG_VALUE is only recorded
/// when its register either (a) has not been redefined by any preceding
/// non-debug instruction (i.e. it still holds the caller-passed value), or
/// (b) was most recently loaded from the stack via $r11 (a stack-passed
/// argument beyond the first five register args). For each argument only the
/// first eligible DBG_VALUE is recorded, since that is its entry location.
///
/// There is another case where DBG_VALUE is not emitted due to
/// AssignmentTrackingAnalysis which determines that a variable is
/// always stack-homed, and describes the variable via MachineFunction's
/// VariableDbgInfo (setVariableDbgInfo with a frame index). To recover the
/// register for those arguments, we also track stores of un-redefined physical
/// registers to stack frame objects during the entry-block walk (using
/// MachineMemOperands to identify the target frame index), then match them
/// against VariableDbgInfo entries after the scan.
static SmallVector<std::pair<uint32_t, Register>, 8>
collectNocallEntryArgRegs(const MachineFunction &MF) {
  SmallDenseMap<uint32_t, Register> EntryRegMap;
  const DISubprogram *SP = MF.getFunction().getSubprogram();
  SmallDenseSet<Register> DefinedRegs, StackLoadRegs;

  // Build a reverse map from IR alloca to frame index so we can
  // identify which frame object a store targets via its MachineMemOperand.
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  SmallDenseMap<const Value *, int> AllocaToFI;
  for (int I = 0, N = MFI.getObjectIndexEnd(); I < N; ++I)
    if (const AllocaInst *AI = MFI.getObjectAllocation(I))
      AllocaToFI[AI] = I;

  // Maps frame index → first physical register stored there before
  // that register is redefined.
  SmallDenseMap<int, Register> FrameIndexToReg;

  for (const MachineInstr &MI : MF.front()) {
    if (MI.isDebugValue()) {
      // Skip indirect DBG_VALUEs — the register is a base address for a
      // memory location, not the argument value itself.
      if (MI.isIndirectDebugValue())
        continue;

      const DILocalVariable *DV = MI.getDebugVariable();
      if (!DV || !DV->getArg() || DV->getScope()->getSubprogram() != SP)
        continue;

      uint32_t Arg = DV->getArg();
      const MachineOperand &MO = MI.getDebugOperand(0);
      if (!MO.isReg() || !MO.getReg().isPhysical())
        continue;

      if (!DefinedRegs.contains(MO.getReg()) ||
          StackLoadRegs.contains(MO.getReg()))
        EntryRegMap.try_emplace(Arg, MO.getReg());
      continue;
    }

    // Track stores of unredefined physical registers to stack frame
    // objects.  Use MachineMemOperands to identify the target frame
    // index rather than assuming a particular addressing mode.
    if (MI.mayStore() && !MI.isCall() && MI.getOperand(0).isReg()) {
      Register SrcReg = MI.getOperand(0).getReg();
      if (SrcReg.isPhysical() && !DefinedRegs.contains(SrcReg)) {
        for (const MachineMemOperand *MMO : MI.memoperands()) {
          const Value *V = MMO->getValue();
          if (!V)
            continue;
          auto It = AllocaToFI.find(V);
          if (It != AllocaToFI.end())
            FrameIndexToReg.try_emplace(It->second, SrcReg);
        }
      }
    }

    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.isDef() && MO.getReg().isPhysical()) {
        DefinedRegs.insert(MO.getReg());
        StackLoadRegs.erase(MO.getReg());
      }

    // Detect stack argument loads: $rX = LDD $r11, offset.
    if (MI.getOpcode() == BPF::LDD && MI.getOperand(1).getReg() == BPF::R11)
      StackLoadRegs.insert(MI.getOperand(0).getReg());
  }

  // Check VariableDbgInfo for args that AssignmentTrackingAnalysis described
  // via setVariableDbgInfo (single-loc stack-homed variables) rather than
  // DBG_VALUE instructions.
  for (const auto &VI : MF.getVariableDbgInfo()) {
    if (!VI.Var || !VI.Var->getArg() || !VI.inStackSlot())
      continue;
    if (VI.Var->getScope()->getSubprogram() != SP)
      continue;
    uint32_t Arg = VI.Var->getArg();
    if (EntryRegMap.count(Arg))
      continue;
    auto It = FrameIndexToReg.find(VI.getStackSlot());
    if (It != FrameIndexToReg.end())
      EntryRegMap[Arg] = It->second;
  }

  SmallVector<std::pair<uint32_t, Register>, 8> AliveArgs(EntryRegMap.begin(),
                                                          EntryRegMap.end());
  llvm::sort(AliveArgs, llvm::less_first());
  return AliveArgs;
}

/// Check whether the optimized IR signature matches the surviving source
/// arguments precisely enough to emit a filtered BTF prototype.
/// Requires exact IR/source arg count match, matching types, and correct
/// BPF register order (R1..R5) for register args.
static bool canUseNocallOptimizedSignature(
    const MachineFunction &MF, DITypeArray Elements,
    ArrayRef<std::pair<uint32_t, Register>> AliveArgs,
    const TargetRegisterInfo &TRI) {
  if (Elements.empty())
    return false;
  if (MF.getFunction().arg_size() != AliveArgs.size()) {
    LLVM_DEBUG(dbgs() << "BTF skip " << MF.getName() << ": IR arg count ("
                      << MF.getFunction().arg_size() << ") != alive arg count ("
                      << AliveArgs.size() << ")\n");
    return false;
  }

  auto ArgIt = MF.getFunction().arg_begin();
  for (unsigned I = 0, N = AliveArgs.size(); I < N; ++I, ++ArgIt) {
    auto [ArgNo, Reg] = AliveArgs[I];
    if (ArgNo >= Elements.size() ||
        !sourceArgMatchesIRType(Elements[ArgNo], ArgIt->getType())) {
      LLVM_DEBUG(dbgs() << "BTF skip " << MF.getName()
                        << ": type mismatch for source arg " << ArgNo
                        << " at IR position " << I << "\n");
      return false;
    }

    if (I >= std::size(CC_BPF64_ArgRegs))
      continue;

    int DwarfReg = TRI.getDwarfRegNum(Reg, false);
    if (DwarfReg != static_cast<int>(I + 1)) {
      LLVM_DEBUG(dbgs() << "BTF skip " << MF.getName() << ": arg " << ArgNo
                        << " in DWARF reg " << DwarfReg << ", expected "
                        << (I + 1) << "\n");
      return false;
    }
  }

  return true;
}

BTFTypeBase::BTFTypeBase(BTF::TypeKinds Kind, uint32_t VLen, bool KindFlag)
    : BTFType{/*NameOff=*/0,
              (static_cast<uint32_t>(KindFlag) << 31) |
                  (static_cast<uint32_t>(Kind) << 24) | VLen,
              /*Size=*/{0}} {}

/// Emit a BTF common type.
void BTFTypeBase::emitType(MCStreamer &OS) {
  OS.AddComment(std::string(BTFKindStr[BTFType.getKind()]) +
                "(id = " + std::to_string(Id) + ")");
  OS.emitInt32(BTFType.NameOff);
  OS.AddComment("0x" + Twine::utohexstr(BTFType.Info));
  OS.emitInt32(BTFType.Info);
  OS.emitInt32(BTFType.Size);
}

BTFTypeDerived::BTFTypeDerived(const DIDerivedType *DTy, BTF::TypeKinds Kind,
                               bool NeedsFixup)
    : BTFTypeBase(Kind), DTy(DTy), NeedsFixup(NeedsFixup),
      Name(DTy->getName()) {}

BTFTypeDerived::BTFTypeDerived(uint32_t NextTypeId, BTF::TypeKinds Kind,
                               StringRef Name)
    : BTFTypeBase(Kind), DTy(nullptr), NeedsFixup(false), Name(Name) {
  BTFType.Type = NextTypeId;
}

void BTFTypeDerived::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  // The kernel rejects names on pointer and qualifier records even when the
  // debug info supplies a name. Only a typedef takes its name from the DI.
  // https://elixir.bootlin.com/linux/v6.17.1/source/kernel/bpf/btf.c#L2586
  if (BTFType.getKind() == BTF::BTF_KIND_TYPEDEF)
    BTFType.NameOff = BDebug.addString(Name);

  if (!NeedsFixup && DTy)
    BTFType.Type = BDebug.getDerivedBaseTypeId(DTy);
}

void BTFTypeDerived::emitType(MCStreamer &OS) { BTFTypeBase::emitType(OS); }

void BTFTypeDerived::setPointeeType(uint32_t PointeeType) {
  BTFType.Type = PointeeType;
}

/// Represent a struct/union forward declaration.
BTFTypeFwd::BTFTypeFwd(StringRef Name, bool IsUnion)
    : BTFTypeBase(BTF::BTF_KIND_FWD, 0, IsUnion), Name(Name) {}

void BTFTypeFwd::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

void BTFTypeFwd::emitType(MCStreamer &OS) { BTFTypeBase::emitType(OS); }

BTFTypeInt::BTFTypeInt(uint32_t Encoding, uint32_t SizeInBits,
                       uint32_t OffsetInBits, StringRef TypeName)
    : BTFTypeBase(BTF::BTF_KIND_INT), Name(TypeName) {
  // Translate IR int encoding to BTF int encoding.
  uint8_t BTFEncoding;
  switch (Encoding) {
  case dwarf::DW_ATE_boolean:
    BTFEncoding = BTF::INT_BOOL;
    break;
  case dwarf::DW_ATE_signed:
  case dwarf::DW_ATE_signed_char:
    BTFEncoding = BTF::INT_SIGNED;
    break;
  case dwarf::DW_ATE_unsigned:
  case dwarf::DW_ATE_unsigned_char:
  case dwarf::DW_ATE_UTF:
    BTFEncoding = 0;
    break;
  default:
    llvm_unreachable("Unknown BTFTypeInt Encoding");
  }

  BTFType.Size = roundupToBytes(SizeInBits);
  IntVal = (BTFEncoding << 24) | OffsetInBits << 16 | SizeInBits;
}

void BTFTypeInt::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

void BTFTypeInt::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.AddComment("0x" + Twine::utohexstr(IntVal));
  OS.emitInt32(IntVal);
}

BTFTypeEnum::BTFTypeEnum(const DICompositeType *ETy, uint32_t VLen,
                         bool IsSigned)
    : BTFTypeBase(BTF::BTF_KIND_ENUM, VLen, IsSigned), ETy(ETy) {
  BTFType.Size = roundupToBytes(ETy->getSizeInBits());
}

void BTFTypeEnum::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(ETy->getName());

  DINodeArray Elements = ETy->getElements();
  for (const auto Element : Elements) {
    const auto *Enum = cast<DIEnumerator>(Element);

    struct BTF::BTFEnum BTFEnum;
    BTFEnum.NameOff = BDebug.addString(Enum->getName());
    // BTF enum value is 32bit, enforce it.
    uint32_t Value;
    if (Enum->isUnsigned())
      Value = static_cast<uint32_t>(Enum->getValue().getZExtValue());
    else
      Value = static_cast<uint32_t>(Enum->getValue().getSExtValue());
    BTFEnum.Val = Value;
    EnumValues.push_back(BTFEnum);
  }
}

void BTFTypeEnum::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Enum : EnumValues) {
    OS.emitInt32(Enum.NameOff);
    OS.emitInt32(Enum.Val);
  }
}

BTFTypeEnum64::BTFTypeEnum64(const DICompositeType *ETy, uint32_t VLen,
                             bool IsSigned)
    : BTFTypeBase(BTF::BTF_KIND_ENUM64, VLen, IsSigned), ETy(ETy) {
  BTFType.Size = roundupToBytes(ETy->getSizeInBits());
}

void BTFTypeEnum64::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(ETy->getName());

  DINodeArray Elements = ETy->getElements();
  for (const auto Element : Elements) {
    const auto *Enum = cast<DIEnumerator>(Element);

    struct BTF::BTFEnum64 BTFEnum;
    BTFEnum.NameOff = BDebug.addString(Enum->getName());
    uint64_t Value;
    if (Enum->isUnsigned())
      Value = Enum->getValue().getZExtValue();
    else
      Value = static_cast<uint64_t>(Enum->getValue().getSExtValue());
    BTFEnum.Val_Lo32 = Value;
    BTFEnum.Val_Hi32 = Value >> 32;
    EnumValues.push_back(BTFEnum);
  }
}

void BTFTypeEnum64::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Enum : EnumValues) {
    OS.emitInt32(Enum.NameOff);
    OS.AddComment("0x" + Twine::utohexstr(Enum.Val_Lo32));
    OS.emitInt32(Enum.Val_Lo32);
    OS.AddComment("0x" + Twine::utohexstr(Enum.Val_Hi32));
    OS.emitInt32(Enum.Val_Hi32);
  }
}

BTFTypeArray::BTFTypeArray(uint32_t ElemTypeId, uint32_t NumElems)
    : BTFTypeBase(BTF::BTF_KIND_ARRAY) {
  ArrayInfo.ElemType = ElemTypeId;
  ArrayInfo.Nelems = NumElems;
}

/// Represent a BTF array.
void BTFTypeArray::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  // The IR does not really have a type for the index.
  // A special type for array index should have been
  // created during initial type traversal. Just
  // retrieve that type id.
  ArrayInfo.IndexType = BDebug.getArrayIndexTypeId();
}

void BTFTypeArray::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.emitInt32(ArrayInfo.ElemType);
  OS.emitInt32(ArrayInfo.IndexType);
  OS.emitInt32(ArrayInfo.Nelems);
}

/// Represent either a struct or a union.
BTFTypeStruct::BTFTypeStruct(const DICompositeType *STy,
                             ArrayRef<const DINode *> Elements, bool IsStruct,
                             bool HasBitField, uint32_t Vlen)
    : BTFTypeBase(IsStruct ? BTF::BTF_KIND_STRUCT : BTF::BTF_KIND_UNION, Vlen,
                  HasBitField),
      STy(STy), Elements(Elements.begin(), Elements.end()),
      HasBitField(HasBitField) {
  BTFType.Size = roundupToBytes(STy->getSizeInBits());
}

void BTFTypeStruct::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(STy->getName());

  if (STy->getTag() == dwarf::DW_TAG_variant_part) {
    // Variant parts might have a discriminator, which has its own memory
    // location, and variants, which share the memory location afterwards. LLVM
    // DI doesn't consider discriminator as an element and instead keeps
    // it as a separate reference.
    // To keep BTF simple, let's represent the structure as an union with
    // discriminator as the first element.
    // The offsets inside variant types are already handled correctly in the
    // DI.
    const auto *DTy = STy->getDiscriminator();
    if (DTy) {
      struct BTF::BTFMember Discriminator;

      Discriminator.NameOff = BDebug.addString(DTy->getName());
      Discriminator.Offset = DTy->getOffsetInBits();
      const auto *BaseTy = DTy->getBaseType();
      Discriminator.Type = BDebug.getTypeId(BaseTy);

      Members.push_back(Discriminator);
    }
  }

  // Add struct/union members.
  for (const auto *Element : Elements) {
    struct BTF::BTFMember BTFMember;

    switch (Element->getTag()) {
    case dwarf::DW_TAG_member: {
      const auto *DDTy = cast<DIDerivedType>(Element);

      BTFMember.NameOff = BDebug.addString(DDTy->getName());
      if (HasBitField) {
        uint8_t BitFieldSize = DDTy->isBitField() ? DDTy->getSizeInBits() : 0;
        BTFMember.Offset = BitFieldSize << 24 | DDTy->getOffsetInBits();
      } else {
        BTFMember.Offset = DDTy->getOffsetInBits();
      }
      const auto *BaseTy = tryRemoveAtomicType(DDTy->getBaseType());
      BTFMember.Type = BDebug.getTypeId(BaseTy);
      break;
    }
    case dwarf::DW_TAG_variant_part: {
      const auto *DCTy = dyn_cast<DICompositeType>(Element);

      BTFMember.NameOff = BDebug.addString(DCTy->getName());
      BTFMember.Offset = DCTy->getOffsetInBits();
      BTFMember.Type = BDebug.getTypeId(DCTy);
      break;
    }
    default:
      llvm_unreachable("Unexpected DI tag of a struct/union element");
    }
    Members.push_back(BTFMember);
  }
}

void BTFTypeStruct::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Member : Members) {
    OS.emitInt32(Member.NameOff);
    OS.emitInt32(Member.Type);
    OS.AddComment("0x" + Twine::utohexstr(Member.Offset));
    OS.emitInt32(Member.Offset);
  }
}

std::string BTFTypeStruct::getName() { return std::string(STy->getName()); }

/// The Func kind represents both subprogram and pointee of function
/// pointers. If the FuncName is empty, it represents a pointee of function
/// pointer. Otherwise, it represents a subprogram. The func arg names
/// are empty for pointee of function pointer case, and are valid names
/// for subprogram.
BTFTypeFuncProto::BTFTypeFuncProto(
    const DISubroutineType *STy, uint32_t VLen,
    const SmallDenseMap<uint32_t, StringRef> &FuncArgNames,
    bool UseFilteredParams, ArrayRef<uint32_t> AliveParamIndices,
    bool VoidReturn)
    : BTFTypeBase(BTF::BTF_KIND_FUNC_PROTO, VLen), STy(STy),
      FuncArgNames(FuncArgNames), AliveParamIndices(AliveParamIndices),
      UseFilteredParams(UseFilteredParams), VoidReturn(VoidReturn) {}

void BTFTypeFuncProto::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  DITypeArray Elements = STy->getTypeArray();
  if (VoidReturn) {
    BTFType.Type = 0;
  } else {
    auto RetType = tryRemoveAtomicType(Elements[0]);
    BTFType.Type = RetType ? BDebug.getTypeId(RetType) : 0;
  }
  BTFType.NameOff = 0;

  auto EmitParam = [&](uint32_t I) {
    struct BTF::BTFParam Param;
    auto Element = tryRemoveAtomicType(Elements[I]);
    if (Element) {
      auto It = FuncArgNames.find(I);
      Param.NameOff =
          It != FuncArgNames.end() ? BDebug.addString(It->second) : 0;
      Param.Type = BDebug.getTypeId(Element);
    } else {
      Param.NameOff = 0;
      Param.Type = 0;
    }
    Parameters.push_back(Param);
  };

  if (UseFilteredParams) {
    for (uint32_t I : AliveParamIndices)
      EmitParam(I);
    return;
  }

  for (unsigned I = 1, N = Elements.size(); I < N; ++I)
    EmitParam(I);
}

void BTFTypeFuncProto::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  for (const auto &Param : Parameters) {
    OS.emitInt32(Param.NameOff);
    OS.emitInt32(Param.Type);
  }
}

BTFTypeFunc::BTFTypeFunc(StringRef FuncName, uint32_t ProtoTypeId,
                         uint32_t Scope)
    : BTFTypeBase(BTF::BTF_KIND_FUNC, Scope), Name(FuncName) {
  BTFType.Type = ProtoTypeId;
}

void BTFTypeFunc::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

void BTFTypeFunc::emitType(MCStreamer &OS) { BTFTypeBase::emitType(OS); }

BTFKindVar::BTFKindVar(StringRef VarName, uint32_t TypeId, uint32_t VarInfo)
    : BTFTypeBase(BTF::BTF_KIND_VAR), Name(VarName), Info(VarInfo) {
  BTFType.Type = TypeId;
}

void BTFKindVar::completeType(BTFDebug &BDebug) {
  BTFType.NameOff = BDebug.addString(Name);
}

void BTFKindVar::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.emitInt32(Info);
}

BTFKindDataSec::BTFKindDataSec(AsmPrinter *AsmPrt, std::string SecName)
    : BTFTypeBase(BTF::BTF_KIND_DATASEC), Asm(AsmPrt), Name(SecName) {}

void BTFKindDataSec::completeType(BTFDebug &BDebug) {
  BTFType.NameOff = BDebug.addString(Name);
  BTFType.Info |= Vars.size();
}

void BTFKindDataSec::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);

  for (const auto &V : Vars) {
    OS.emitInt32(std::get<0>(V));
    Asm->emitLabelReference(std::get<1>(V), 4);
    OS.emitInt32(std::get<2>(V));
  }
}

BTFTypeFloat::BTFTypeFloat(uint32_t SizeInBits, StringRef TypeName)
    : BTFTypeBase(BTF::BTF_KIND_FLOAT), Name(TypeName) {
  BTFType.Size = roundupToBytes(SizeInBits);
}

void BTFTypeFloat::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Name);
}

BTFTypeDeclTag::BTFTypeDeclTag(uint32_t BaseTypeId, int ComponentIdx,
                               StringRef Tag)
    : BTFTypeBase(BTF::BTF_KIND_DECL_TAG), Info(ComponentIdx), Tag(Tag) {
  BTFType.Type = BaseTypeId;
}

void BTFTypeDeclTag::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;

  BTFType.NameOff = BDebug.addString(Tag);
}

void BTFTypeDeclTag::emitType(MCStreamer &OS) {
  BTFTypeBase::emitType(OS);
  OS.emitInt32(Info);
}

BTFTypeTypeTag::BTFTypeTypeTag(uint32_t NextTypeId, StringRef Tag)
    : BTFTypeBase(BTF::BTF_KIND_TYPE_TAG), DTy(nullptr), Tag(Tag) {
  BTFType.Type = NextTypeId;
}

BTFTypeTypeTag::BTFTypeTypeTag(const DIDerivedType *DTy, StringRef Tag)
    : BTFTypeBase(BTF::BTF_KIND_TYPE_TAG), DTy(DTy), Tag(Tag) {}

void BTFTypeTypeTag::completeType(BTFDebug &BDebug) {
  if (IsCompleted)
    return;
  IsCompleted = true;
  BTFType.NameOff = BDebug.addString(Tag);
  if (DTy)
    BTFType.Type = BDebug.getDerivedBaseTypeId(DTy);
}

uint32_t BTFStringTable::addString(StringRef S) {
  // Check whether the string already exists.
  for (auto &OffsetM : OffsetToIdMap) {
    if (Table[OffsetM.second] == S)
      return OffsetM.first;
  }
  // Not find, add to the string table.
  uint32_t Offset = Size;
  OffsetToIdMap[Offset] = Table.size();
  Table.push_back(std::string(S));
  Size += S.size() + 1;
  return Offset;
}

BTFDebug::BTFDebug(AsmPrinter *AP)
    : DebugHandlerBase(AP), OS(*Asm->OutStreamer), SkipInstruction(false),
      LineInfoGenerated(false), SecNameOff(0), ArrayIndexTypeId(0),
      MapDefNotCollected(true) {
  addString("\0");
}

uint32_t BTFDebug::addType(std::unique_ptr<BTFTypeBase> TypeEntry,
                           const DIType *Ty) {
  const uint32_t Id = TypeEntries.size() + 1;
  TypeEntry->setId(Id);
  if (Ty)
    DIToIdMap[Ty] = Id;
  TypeEntries.push_back(std::move(TypeEntry));
  return Id;
}

static StringRef getDITypeName(const DIType *Ty) {
  if (!Ty)
    return "void";
  StringRef Name = Ty->getName();
  return Name.empty() ? dwarf::TagString(Ty->getTag()) : Name;
}

uint32_t BTFDebug::getTypeId(const DIType *Ty) const {
  auto It = DIToIdMap.find(Ty);
  if (It != DIToIdMap.end())
    return It->second;
  OS.getContext().reportError({},
                              "BTF record references unrepresentable type '" +
                                  getDITypeName(Ty) + "'");
  return 0;
}

uint32_t BTFDebug::getDerivedBaseTypeId(const DIDerivedType *DTy) const {
  const DIType *Ty = tryRemoveAtomicType(DTy->getBaseType());
  if (!Ty)
    return 0;
  // Use void for a function pointee when its signature has no BTF record.
  if (isa<DISubroutineType>(Ty))
    return DIToIdMap.lookup(Ty);
  return getTypeId(Ty);
}

std::optional<uint32_t> BTFDebug::visitBasicType(const DIBasicType *BTy) {
  // Only int and binary floating point types are supported in BTF.
  const uint32_t Encoding = BTy->getEncoding();
  switch (Encoding) {
  case dwarf::DW_ATE_boolean:
  case dwarf::DW_ATE_signed:
  case dwarf::DW_ATE_signed_char:
  case dwarf::DW_ATE_unsigned:
  case dwarf::DW_ATE_unsigned_char:
  case dwarf::DW_ATE_UTF:
    return addType(std::make_unique<BTFTypeInt>(Encoding, BTy->getSizeInBits(),
                                                BTy->getOffsetInBits(),
                                                BTy->getName()),
                   BTy);
  case dwarf::DW_ATE_float:
    return addType(
        std::make_unique<BTFTypeFloat>(BTy->getSizeInBits(), BTy->getName()),
        BTy);
  default:
    return std::nullopt;
  }
}

/// Handle subprogram or subroutine types.
std::optional<uint32_t> BTFDebug::visitSubroutineType(
    const DISubroutineType *STy, bool ForSubprog,
    const SmallDenseMap<uint32_t, StringRef> &FuncArgNames, bool VoidReturn) {
  DITypeArray Elements = STy->getTypeArray();
  // Even void(void) has a return type slot. An empty list has no prototype.
  if (Elements.empty() || Elements.size() - 1 > BTF::MAX_VLEN)
    return std::nullopt;
  const uint32_t VLen = Elements.size() - 1;

  // Functions sharing STy can have different parameter names or optimized
  // return types. Keep each definition's prototype out of the shared type map.
  const uint32_t TypeId = addType(
      std::make_unique<BTFTypeFuncProto>(STy, VLen, FuncArgNames, false,
                                         ArrayRef<uint32_t>(), VoidReturn),
      ForSubprog ? nullptr : STy);

  // Visit return type and func arg types.
  if (!VoidReturn) {
    for (const auto Element : Elements)
      visitTypeEntry(Element);
  } else {
    for (unsigned I = 1, N = Elements.size(); I < N; ++I)
      visitTypeEntry(Elements[I]);
  }
  return TypeId;
}

void BTFDebug::processDeclAnnotations(DINodeArray Annotations,
                                      uint32_t BaseTypeId,
                                      int ComponentIdx) {
  if (!Annotations)
     return;

  for (const Metadata *Annotation : Annotations->operands()) {
    const MDNode *MD = cast<MDNode>(Annotation);
    const MDString *Name = cast<MDString>(MD->getOperand(0));
    if (Name->getString() != "btf_decl_tag")
      continue;

    const MDString *Value = cast<MDString>(MD->getOperand(1));
    addType(std::make_unique<BTFTypeDeclTag>(BaseTypeId, ComponentIdx,
                                             Value->getString()));
  }
}

uint32_t BTFDebug::processDISubprogram(
    const DISubprogram *SP, uint32_t ProtoTypeId, uint8_t Scope,
    const SmallDenseMap<uint32_t, uint32_t> *ArgIndexMap) {
  const uint32_t FuncId =
      addType(std::make_unique<BTFTypeFunc>(SP->getName(), ProtoTypeId, Scope));

  // Process argument annotations.
  for (const MDNode *DN : SP->getRetainedNodes()) {
    if (const auto *DV = dyn_cast<DILocalVariable>(DN)) {
      uint32_t Arg = DV->getArg();
      if (Arg) {
        if (ArgIndexMap) {
          auto It = ArgIndexMap->find(Arg);
          if (It != ArgIndexMap->end())
            processDeclAnnotations(DV->getAnnotations(), FuncId, It->second);
        } else {
          processDeclAnnotations(DV->getAnnotations(), FuncId, Arg - 1);
        }
      }
    }
  }
  processDeclAnnotations(SP->getAnnotations(), FuncId, -1);

  return FuncId;
}

/// Generate btf_type_tag chains.
std::optional<uint32_t>
BTFDebug::genBTFTypeTags(const DIDerivedType *DTy,
                         std::optional<uint32_t> BaseTypeId) {
  SmallVector<const MDString *, 4> MDStrs;
  DINodeArray Annots = DTy->getAnnotations();
  if (Annots) {
    // For type with "int __tag1 __tag2 *p", the MDStrs will have
    // content: [__tag1, __tag2].
    for (const Metadata *Annotations : Annots->operands()) {
      const MDNode *MD = cast<MDNode>(Annotations);
      const MDString *Name = cast<MDString>(MD->getOperand(0));
      if (Name->getString() != "btf_type_tag")
        continue;
      MDStrs.push_back(cast<MDString>(MD->getOperand(1)));
    }
  }

  if (MDStrs.empty())
    return std::nullopt;

  // With MDStrs [__tag1, __tag2], the output type chain looks like
  //   PTR -> __tag2 -> __tag1 -> BaseType
  // In the below, we construct BTF types with the order of __tag1, __tag2
  // and PTR.
  const StringRef FirstName = MDStrs.front()->getString();
  uint32_t TagTypeId = addType(
      BaseTypeId.has_value()
          ? std::make_unique<BTFTypeTypeTag>(BaseTypeId.value(), FirstName)
          : std::make_unique<BTFTypeTypeTag>(DTy, FirstName));
  for (const MDString *Value : llvm::drop_begin(MDStrs))
    TagTypeId = addType(
        std::make_unique<BTFTypeTypeTag>(TagTypeId, Value->getString()));
  return TagTypeId;
}

/// Handle structure/union types.
std::optional<uint32_t> BTFDebug::visitStructType(const DICompositeType *CTy,
                                                  bool IsStruct) {
  DINodeArray DIElements = CTy->getElements();
  SmallVector<const DINode *, 8> Elements(DIElements.begin(), DIElements.end());
  // Structure elements must have nondecreasing offsets in BTF. Preserve DI
  // order for union and variant-part records.
  if (CTy->getTag() == dwarf::DW_TAG_structure_type)
    llvm::stable_sort(Elements, [](const DINode *LHS, const DINode *RHS) {
      return getBTFRecordElementOffset(LHS) < getBTFRecordElementOffset(RHS);
    });
  uint32_t VLen = Elements.size();
  // Variant parts might have a discriminator. LLVM DI doesn't consider it as
  // an element and instead keeps it as a separate reference. But we represent
  // it as an element in BTF.
  if (CTy->getTag() == dwarf::DW_TAG_variant_part) {
    const auto *DTy = CTy->getDiscriminator();
    if (DTy) {
      visitTypeEntry(DTy);
      VLen++;
    }
  }
  if (VLen > BTF::MAX_VLEN)
    return std::nullopt;

  // Check whether we have any bitfield members or not
  bool HasBitField = false;
  for (const auto *Element : Elements) {
    if (Element->getTag() == dwarf::DW_TAG_member) {
      auto E = cast<DIDerivedType>(Element);
      if (E->isBitField()) {
        HasBitField = true;
        break;
      }
    }
  }

  std::unique_ptr<BTFTypeStruct> TypeEntry = std::make_unique<BTFTypeStruct>(
      CTy, Elements, IsStruct, HasBitField, VLen);
  StructTypes.push_back(TypeEntry.get());
  const uint32_t TypeId = addType(std::move(TypeEntry), CTy);

  // Check struct/union annotations
  processDeclAnnotations(CTy->getAnnotations(), TypeId, -1);

  // Visit all struct members.
  int FieldNo = 0;
  for (const auto *Element : Elements) {
    switch (Element->getTag()) {
    case dwarf::DW_TAG_member: {
      const auto Elem = cast<DIDerivedType>(Element);
      visitTypeEntry(Elem);
      processDeclAnnotations(Elem->getAnnotations(), TypeId, FieldNo);
      break;
    }
    case dwarf::DW_TAG_variant_part: {
      const auto Elem = cast<DICompositeType>(Element);
      visitTypeEntry(Elem);
      processDeclAnnotations(Elem->getAnnotations(), TypeId, FieldNo);
      break;
    }
    default:
      llvm_unreachable("Unexpected DI tag of a struct/union element");
    }
    FieldNo++;
  }
  return TypeId;
}

std::optional<uint32_t> BTFDebug::visitArrayType(const DICompositeType *CTy) {
  // Collect dimensions before registering any part of the array.
  SmallVector<uint32_t> Counts;
  for (const DINode *Dim : CTy->getElements()) {
    const auto *Range = dyn_cast_or_null<DISubrange>(Dim);
    const auto *Size =
        Range ? dyn_cast_if_present<ConstantInt *>(Range->getCount()) : nullptr;
    if (!Size)
      return std::nullopt;
    // A flexible array member has count -1 in DI and count 0 in BTF.
    const APInt &Count = Size->getValue();
    if (!Count.isNegative() && !Count.isIntN(32))
      return std::nullopt;
    Counts.push_back(Count.isNegative() ? 0 : Count.getZExtValue());
  }
  // Rust and other languages may omit the dimensions of a zero-length array.
  if (Counts.empty())
    Counts.push_back(0);

  const std::optional ElemTypeId = visitTypeEntry(CTy->getBaseType());
  if (!ElemTypeId.has_value() || ElemTypeId.value() == 0)
    return std::nullopt;
  uint32_t TypeId = ElemTypeId.value();
  for (auto [I, Count] : llvm::enumerate(llvm::reverse(Counts)))
    TypeId = addType(std::make_unique<BTFTypeArray>(TypeId, Count),
                     I + 1 == Counts.size() ? CTy : nullptr);

  // The IR does not have a type for array index while BTF wants one.
  // So create an array index type if there is none.
  if (!ArrayIndexTypeId)
    ArrayIndexTypeId = addType(std::make_unique<BTFTypeInt>(
        dwarf::DW_ATE_unsigned, 32, 0, "__ARRAY_SIZE_TYPE__"));
  return TypeId;
}

std::optional<uint32_t> BTFDebug::visitEnumType(const DICompositeType *CTy) {
  DINodeArray Elements = CTy->getElements();
  uint32_t VLen = Elements.size();
  if (VLen > BTF::MAX_VLEN)
    return std::nullopt;

  bool IsSigned = false;
  unsigned NumBits = 32;
  // No BaseType implies forward declaration in which case a
  // BTFTypeEnum with Vlen = 0 is emitted.
  if (CTy->getBaseType() != nullptr) {
    const auto *BTy = cast<DIBasicType>(CTy->getBaseType());
    IsSigned = BTy->getEncoding() == dwarf::DW_ATE_signed ||
               BTy->getEncoding() == dwarf::DW_ATE_signed_char;
    NumBits = BTy->getSizeInBits();
  }

  // No need to visit base type as BTF does not encode it.
  if (NumBits <= 32)
    return addType(std::make_unique<BTFTypeEnum>(CTy, VLen, IsSigned), CTy);
  assert(NumBits == 64);
  return addType(std::make_unique<BTFTypeEnum64>(CTy, VLen, IsSigned), CTy);
}

/// Handle structure/union forward declarations.
uint32_t BTFDebug::visitFwdDeclType(const DICompositeType *CTy, bool IsUnion) {
  return addType(std::make_unique<BTFTypeFwd>(CTy->getName(), IsUnion), CTy);
}

/// Handle structure, union, array and enumeration types.
std::optional<uint32_t>
BTFDebug::visitCompositeType(const DICompositeType *CTy) {
  auto Tag = CTy->getTag();
  switch (Tag) {
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_variant_part:
    // Handle forward declaration differently as it does not have members.
    if (CTy->isForwardDecl())
      return visitFwdDeclType(CTy, Tag == dwarf::DW_TAG_union_type);
    return visitStructType(CTy, Tag == dwarf::DW_TAG_structure_type);
  case dwarf::DW_TAG_array_type:
    return visitArrayType(CTy);
  case dwarf::DW_TAG_enumeration_type:
    return visitEnumType(CTy);
  default:
    return std::nullopt;
  }
}

bool BTFDebug::IsForwardDeclCandidate(const DIType *Base) {
  if (const auto *CTy = dyn_cast<DICompositeType>(Base)) {
    auto CTag = CTy->getTag();
    if ((CTag == dwarf::DW_TAG_structure_type ||
         CTag == dwarf::DW_TAG_union_type) &&
        !CTy->getName().empty() && !CTy->isForwardDecl())
      return true;
  }
  return false;
}

/// Handle pointer, typedef, const, volatile, restrict and member types.
std::optional<uint32_t> BTFDebug::visitDerivedType(const DIDerivedType *DTy,
                                                   TypeUse Use) {
  const DIType *Base = DTy->getBaseType();
  BTF::TypeKinds Kind;
  switch (DTy->getTag()) {
  case dwarf::DW_TAG_atomic_type:
    return visitTypeEntry(Base, Use);
  case dwarf::DW_TAG_member:
    visitTypeEntry(Base, TypeUse::Member);
    return std::nullopt; // Members have no separate BTF record.
  case dwarf::DW_TAG_pointer_type:
    Kind = BTF::BTF_KIND_PTR;
    if (Use == TypeUse::Member && !DTy->getAnnotations())
      Use = TypeUse::MemberPointee;
    break;
  case dwarf::DW_TAG_typedef:
    Kind = BTF::BTF_KIND_TYPEDEF;
    break;
  case dwarf::DW_TAG_const_type:
    Kind = BTF::BTF_KIND_CONST;
    break;
  case dwarf::DW_TAG_volatile_type:
    Kind = BTF::BTF_KIND_VOLATILE;
    break;
  case dwarf::DW_TAG_restrict_type:
    Kind = BTF::BTF_KIND_RESTRICT;
    break;
  default:
    return std::nullopt;
  }

  // A struct member's pointee may need only a forward declaration.
  if (Use == TypeUse::MemberPointee && Base && IsForwardDeclCandidate(Base)) {
    std::unique_ptr<BTFTypeDerived> TypeEntry =
        std::make_unique<BTFTypeDerived>(DTy, Kind, true);
    auto &Fixup = FixupDerivedTypes[cast<DICompositeType>(Base)];
    Fixup.emplace_back(DTy, TypeEntry.get());
    return addType(std::move(TypeEntry), DTy);
  }

  const std::optional TagTypeId =
      Kind == BTF::BTF_KIND_PTR || Kind == BTF::BTF_KIND_TYPEDEF
          ? genBTFTypeTags(DTy)
          : std::nullopt;
  const uint32_t TypeId =
      addType(TagTypeId.has_value()
                  ? std::make_unique<BTFTypeDerived>(TagTypeId.value(), Kind,
                                                     DTy->getName())
                  : std::make_unique<BTFTypeDerived>(DTy, Kind, false),
              DTy);
  if (Kind == BTF::BTF_KIND_TYPEDEF)
    processDeclAnnotations(DTy->getAnnotations(), TypeId, -1);
  visitTypeEntry(Base, Use);
  return TypeId;
}

std::optional<uint32_t> BTFDebug::visitTypeEntry(const DIType *Ty,
                                                 TypeUse Use) {
  if (!Ty)
    return 0; // BTF void.
  if (auto It = DIToIdMap.find(Ty); It != DIToIdMap.end()) {
    const uint32_t TypeId = It->second;

    // To handle the case like the following:
    //    struct t;
    //    typedef struct t _t;
    //    struct s1 { _t *c; };
    //    int test1(struct s1 *arg) { ... }
    //
    //    struct t { int a; int b; };
    //    struct s2 { _t c; }
    //    int test2(struct s2 *arg) { ... }
    //
    // During traversing test1() argument, "_t" is recorded
    // in DIToIdMap and a forward declaration fixup is created
    // for "struct t" to avoid pointee type traversal.
    //
    // During traversing test2() argument, even if we see "_t" is
    // already defined, we should keep moving to eventually
    // bring in types for "struct t". Otherwise, the "struct s2"
    // definition won't be correct.
    //
    // In the above, we have following debuginfo:
    //  {ptr, struct_member} ->  typedef -> struct
    // and BTF type for 'typedef' is generated while 'struct' may
    // be in FixUp. But let us generalize the above to handle
    //  {different types} -> [various derived types]+ -> another type.
    // For example,
    //  {func_param, struct_member} -> const -> ptr -> volatile -> struct
    // We will traverse const/ptr/volatile which already have corresponding
    // BTF types and generate type for 'struct' which might be in Fixup
    // state.
    if (Use != TypeUse::MemberPointee) {
      if (const auto *DTy = dyn_cast<DIDerivedType>(Ty)) {
        while (DTy) {
          const DIType *BaseTy = DTy->getBaseType();
          if (!BaseTy)
            break;

          if (DIToIdMap.find(BaseTy) != DIToIdMap.end()) {
            DTy = dyn_cast<DIDerivedType>(BaseTy);
          } else {
            if (Use == TypeUse::Member &&
                DTy->getTag() == dwarf::DW_TAG_pointer_type &&
                !DTy->getAnnotations()) {
              Use = TypeUse::MemberPointee;
              if (IsForwardDeclCandidate(BaseTy))
                break;
            }
            visitTypeEntry(BaseTy, Use);
            break;
          }
        }
      }
    }

    return TypeId;
  }

  if (const auto *BTy = dyn_cast<DIBasicType>(Ty))
    return visitBasicType(BTy);
  if (const auto *STy = dyn_cast<DISubroutineType>(Ty))
    return visitSubroutineType(STy, false, {});
  if (const auto *CTy = dyn_cast<DICompositeType>(Ty))
    return visitCompositeType(CTy);
  if (const auto *DTy = dyn_cast<DIDerivedType>(Ty))
    return visitDerivedType(DTy, Use);
  return std::nullopt;
}

std::optional<uint32_t> BTFDebug::visitMapDefType(const DIType *Ty) {
  if (!Ty)
    return 0;
  if (auto It = DIToIdMap.find(Ty); It != DIToIdMap.end())
    return It->second;

  switch (Ty->getTag()) {
  case dwarf::DW_TAG_typedef:
  case dwarf::DW_TAG_const_type:
  case dwarf::DW_TAG_volatile_type:
  case dwarf::DW_TAG_restrict_type:
  case dwarf::DW_TAG_pointer_type:
    visitMapDefType(cast<DIDerivedType>(Ty)->getBaseType());
    break;
  case dwarf::DW_TAG_array_type:
    // Visit nested map array and jump to the element type
    visitMapDefType(cast<DICompositeType>(Ty)->getBaseType());
    break;
  case dwarf::DW_TAG_structure_type: {
    // Visit all struct members to ensure their types are visited.
    const auto *CTy = cast<DICompositeType>(Ty);
    const DINodeArray Elements = CTy->getElements();
    for (const auto *Element : Elements) {
      const auto *MemberType = cast<DIDerivedType>(Element);
      const DIType *MemberBaseType = MemberType->getBaseType();
      // If the member is a composite type, that may indicate the currently
      // visited composite type is a wrapper, and the member represents the
      // actual map definition.
      // In that case, visit the member with `visitMapDefType` instead of
      // `visitTypeEntry`, treating it specifically as a map definition rather
      // than as a regular composite type.
      if (isa_and_nonnull<DICompositeType>(MemberBaseType))
        visitMapDefType(MemberBaseType);
      else
        visitTypeEntry(MemberBaseType);
    }
    break;
  }
  default:
    break;
  }

  // Visit this type, struct or a const/typedef/volatile/restrict type
  return visitTypeEntry(Ty);
}

/// Read file contents from the actual file or from the source
std::string BTFDebug::populateFileContent(const DIFile *File) {
  if (!File)
    return {};
  std::string FileName;

  if (!File->getFilename().starts_with("/") && File->getDirectory().size())
    FileName = File->getDirectory().str() + "/" + File->getFilename().str();
  else
    FileName = std::string(File->getFilename());

  // No need to populate the contends if it has been populated!
  if (FileContent.contains(FileName))
    return FileName;

  std::vector<std::string> Content;
  std::string Line;
  Content.push_back(Line); // Line 0 for empty string

  auto LoadFile = [&File](StringRef FileName) {
    return File->getContext().getVirtualFileSystem().getBufferForFile(FileName);
  };

  std::unique_ptr<MemoryBuffer> Buf;
  if (const std::optional Source = File->getSource(); Source.has_value())
    Buf = MemoryBuffer::getMemBufferCopy(Source.value());
  else if (ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr = LoadFile(FileName))
    Buf = std::move(*BufOrErr);
  if (Buf)
    for (line_iterator I(*Buf, false), E; I != E; ++I)
      Content.push_back(std::string(*I));

  FileContent[FileName] = std::move(Content);
  return FileName;
}

void BTFDebug::constructLineInfo(MCSymbol *Label, const DIFile *File,
                                 uint32_t Line, uint32_t Column) {
  std::string FileName = populateFileContent(File);
  BTFLineInfo LineInfo;

  LineInfo.Label = Label;
  LineInfo.FileNameOff = addString(FileName);
  // If file content is not available, let LineOff = 0.
  const auto &Content = FileContent[FileName];
  if (Line < Content.size())
    LineInfo.LineOff = addString(Content[Line]);
  else
    LineInfo.LineOff = 0;
  LineInfo.LineNum = Line;
  LineInfo.ColumnNum = Column;
  LineInfoTable[SecNameOff].push_back(LineInfo);
}

void BTFDebug::emitCommonHeader() {
  OS.AddComment("0x" + Twine::utohexstr(BTF::MAGIC));
  OS.emitIntValue(BTF::MAGIC, 2);
  OS.emitInt8(BTF::VERSION);
  OS.emitInt8(0);
}

void BTFDebug::emitBTFSection() {
  if (TypeEntries.empty())
    return;

  MCContext &Ctx = OS.getContext();
  MCSectionELF *Sec = Ctx.getELFSection(".BTF", ELF::SHT_PROGBITS, 0);
  Sec->setAlignment(Align(4));
  OS.switchSection(Sec);

  // Emit header.
  emitCommonHeader();
  OS.emitInt32(BTF::HeaderSize);

  uint32_t TypeLen = 0, StrLen;
  for (BTFTypeBase &TypeEntry : make_pointee_range(TypeEntries))
    TypeLen += TypeEntry.getSize();
  StrLen = StringTable.getSize();

  OS.emitInt32(0);
  OS.emitInt32(TypeLen);
  OS.emitInt32(TypeLen);
  OS.emitInt32(StrLen);

  // Emit type table.
  for (BTFTypeBase &TypeEntry : make_pointee_range(TypeEntries))
    TypeEntry.emitType(OS);

  // Emit string table.
  uint32_t StringOffset = 0;
  for (const auto &S : StringTable.getTable()) {
    OS.AddComment("string offset=" + std::to_string(StringOffset));
    OS.emitBytes(S);
    OS.emitBytes(StringRef("\0", 1));
    StringOffset += S.size() + 1;
  }
}

void BTFDebug::emitBTFExtSection() {
  // If present, func_info must cover every function in the program.
  if (HasIncompleteFuncInfo)
    FuncInfoTable.clear();

  // Do not emit section if empty FuncInfoTable and LineInfoTable
  // and FieldRelocTable.
  if (!FuncInfoTable.size() && !LineInfoTable.size() &&
      !FieldRelocTable.size())
    return;

  MCContext &Ctx = OS.getContext();
  MCSectionELF *Sec = Ctx.getELFSection(".BTF.ext", ELF::SHT_PROGBITS, 0);
  Sec->setAlignment(Align(4));
  OS.switchSection(Sec);

  // Emit header.
  emitCommonHeader();
  OS.emitInt32(BTF::ExtHeaderSize);

  // Account for FuncInfo/LineInfo record size as well.
  uint32_t FuncLen = FuncInfoTable.empty() ? 0 : 4;
  uint32_t LineLen = LineInfoTable.empty() ? 0 : 4;
  // Do not account for optional FieldReloc.
  uint32_t FieldRelocLen = 0;
  for (const auto &FuncSec : FuncInfoTable) {
    FuncLen += BTF::SecFuncInfoSize;
    FuncLen += FuncSec.second.size() * BTF::BPFFuncInfoSize;
  }
  for (const auto &LineSec : LineInfoTable) {
    LineLen += BTF::SecLineInfoSize;
    LineLen += LineSec.second.size() * BTF::BPFLineInfoSize;
  }
  for (const auto &FieldRelocSec : FieldRelocTable) {
    FieldRelocLen += BTF::SecFieldRelocSize;
    FieldRelocLen += FieldRelocSec.second.size() * BTF::BPFFieldRelocSize;
  }

  if (FieldRelocLen)
    FieldRelocLen += 4;

  OS.emitInt32(0);
  OS.emitInt32(FuncLen);
  OS.emitInt32(FuncLen);
  OS.emitInt32(LineLen);
  OS.emitInt32(FuncLen + LineLen);
  OS.emitInt32(FieldRelocLen);

  // Emit func_info table.
  if (FuncLen) {
    OS.AddComment("FuncInfo");
    OS.emitInt32(BTF::BPFFuncInfoSize);
  }
  for (const auto &FuncSec : FuncInfoTable) {
    OS.AddComment("FuncInfo section string offset=" +
                  std::to_string(FuncSec.first));
    OS.emitInt32(FuncSec.first);
    OS.emitInt32(FuncSec.second.size());
    for (const auto &FuncInfo : FuncSec.second) {
      Asm->emitLabelReference(FuncInfo.Label, 4);
      OS.emitInt32(FuncInfo.TypeId);
    }
  }

  // Emit line_info table.
  if (LineLen) {
    OS.AddComment("LineInfo");
    OS.emitInt32(BTF::BPFLineInfoSize);
  }
  for (const auto &LineSec : LineInfoTable) {
    OS.AddComment("LineInfo section string offset=" +
                  std::to_string(LineSec.first));
    OS.emitInt32(LineSec.first);
    OS.emitInt32(LineSec.second.size());
    for (const auto &LineInfo : LineSec.second) {
      Asm->emitLabelReference(LineInfo.Label, 4);
      OS.emitInt32(LineInfo.FileNameOff);
      OS.emitInt32(LineInfo.LineOff);
      OS.AddComment("Line " + std::to_string(LineInfo.LineNum) + " Col " +
                    std::to_string(LineInfo.ColumnNum));
      OS.emitInt32(LineInfo.LineNum << 10 | LineInfo.ColumnNum);
    }
  }

  // Emit field reloc table.
  if (FieldRelocLen) {
    OS.AddComment("FieldReloc");
    OS.emitInt32(BTF::BPFFieldRelocSize);
    for (const auto &FieldRelocSec : FieldRelocTable) {
      OS.AddComment("Field reloc section string offset=" +
                    std::to_string(FieldRelocSec.first));
      OS.emitInt32(FieldRelocSec.first);
      OS.emitInt32(FieldRelocSec.second.size());
      for (const auto &FieldRelocInfo : FieldRelocSec.second) {
        Asm->emitLabelReference(FieldRelocInfo.Label, 4);
        OS.emitInt32(FieldRelocInfo.TypeID);
        OS.emitInt32(FieldRelocInfo.OffsetNameOff);
        OS.emitInt32(FieldRelocInfo.RelocKind);
      }
    }
  }
}

void BTFDebug::beginFunctionImpl(const MachineFunction *MF) {
  auto *SP = MF->getFunction().getSubprogram();
  auto *Unit = SP->getUnit();

  if (Unit->getEmissionKind() == DICompileUnit::NoDebug) {
    SkipInstruction = true;
    return;
  }
  SkipInstruction = false;

  // Collect MapDef types. Map definition needs to collect
  // pointee types. Do it first. Otherwise, for the following
  // case:
  //    struct m { ...};
  //    struct t {
  //      struct m *key;
  //    };
  //    foo(struct t *arg);
  //
  //    struct mapdef {
  //      ...
  //      struct m *key;
  //      ...
  //    } __attribute__((section(".maps"))) hash_map;
  //
  // If subroutine foo is traversed first, a type chain
  // "ptr->struct m(fwd)" will be created and later on
  // when traversing mapdef, since "ptr->struct m" exists,
  // the traversal of "struct m" will be omitted.
  if (MapDefNotCollected) {
    processGlobals(true);
    MapDefNotCollected = false;
  }

  // Collect all types locally referenced in this function.
  // Use RetainedNodes so we can collect all argument names
  // even if the argument is not used.
  SmallDenseMap<uint32_t, StringRef> FuncArgNames;
  for (const MDNode *DN : SP->getRetainedNodes()) {
    if (const auto *DV = dyn_cast<DILocalVariable>(DN)) {
      // Collect function arguments for subprogram func type.
      uint32_t Arg = DV->getArg();
      if (Arg) {
        visitTypeEntry(DV->getType());
        FuncArgNames[Arg] = DV->getName();
      }
    }
  }

  // Construct subprogram func proto type.
  std::optional<uint32_t> FuncTypeId;
  uint8_t Scope = SP->isLocalToUnit() ? BTF::FUNC_STATIC : BTF::FUNC_GLOBAL;
  bool IsNocall = SP->getType()->getCC() == dwarf::DW_CC_nocall;
  bool VoidReturn = MF->getFunction().getReturnType()->isVoidTy();

  if (IsNocall) {
    // For DW_CC_nocall functions, try to build a FUNC_PROTO reflecting
    // the true ABI: only parameters that survived optimization and whose
    // first 5 arguments map to the correct BPF registers (R1-R5).
    const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
    DITypeArray Elements = SP->getType()->getTypeArray();

    SmallVector<std::pair<uint32_t, Register>, 8> AliveArgs =
        collectNocallEntryArgRegs(*MF);

    if (canUseNocallOptimizedSignature(*MF, Elements, AliveArgs, *TRI)) {
      SmallVector<uint32_t, 8> AliveParamIndices;
      SmallDenseMap<uint32_t, uint32_t> ArgIndexMap;
      for (auto [I, ArgReg] : llvm::enumerate(AliveArgs)) {
        AliveParamIndices.push_back(ArgReg.first);
        ArgIndexMap[ArgReg.first] = I;
      }

      if (!VoidReturn)
        visitTypeEntry(Elements[0]);
      for (uint32_t ArgNo : AliveParamIndices)
        visitTypeEntry(Elements[ArgNo]);

      const uint32_t ProtoTypeId = addType(std::make_unique<BTFTypeFuncProto>(
          SP->getType(), AliveParamIndices.size(), FuncArgNames, true,
          AliveParamIndices, VoidReturn));
      FuncTypeId = processDISubprogram(SP, ProtoTypeId, Scope, &ArgIndexMap);
    }
  }

  if (!FuncTypeId.has_value()) {
    // Fall back to the full source prototype, still voiding the return
    // type if compiler removed it.
    const std::optional ProtoTypeId =
        visitSubroutineType(SP->getType(), true, FuncArgNames, VoidReturn);
    if (ProtoTypeId.has_value())
      FuncTypeId = processDISubprogram(SP, ProtoTypeId.value(), Scope);
  }

  for (BTFTypeBase &TypeEntry : make_pointee_range(TypeEntries))
    TypeEntry.completeType(*this);

  // Construct funcinfo and the first lineinfo for the function.
  MCSymbol *FuncLabel = Asm->getFunctionBegin();
  if (FuncLabel->isInSection()) {
    auto &Sec = static_cast<const MCSectionELF &>(FuncLabel->getSection());
    SecNameOff = addString(Sec.getName());
  } else {
    SecNameOff = addString(".text");
  }
  if (FuncTypeId.has_value())
    FuncInfoTable[SecNameOff].push_back({FuncLabel, FuncTypeId.value()});
  else
    HasIncompleteFuncInfo = true;
}

void BTFDebug::endFunctionImpl(const MachineFunction *MF) {
  SkipInstruction = false;
  LineInfoGenerated = false;
  SecNameOff = 0;
}

/// Generate a struct member field relocation.
void BTFDebug::generatePatchImmReloc(const MCSymbol *ORSym, uint32_t RootId,
                                     const GlobalVariable *GVar, bool IsAma) {
  BTFFieldReloc FieldReloc;
  FieldReloc.Label = ORSym;
  FieldReloc.TypeID = RootId;

  if (IsAma) {
    auto Reloc = BPFCoreSharedInfo::AmaRelocation::decode(GVar->getName());
    FieldReloc.OffsetNameOff = addString(Reloc.AccessString);
    FieldReloc.RelocKind = Reloc.Kind;
    PatchImms[GVar] = {Reloc.PatchImm, Reloc.Kind};
  } else {
    StringRef RelocStr = GVar->getName().split('$').second;
    FieldReloc.OffsetNameOff = addString("0");
    FieldReloc.RelocKind = std::stoull(std::string(RelocStr));
    PatchImms[GVar] = std::make_pair(RootId, FieldReloc.RelocKind);
  }
  FieldRelocTable[SecNameOff].push_back(FieldReloc);
}

void BTFDebug::processGlobalValue(const MachineOperand &MO) {
  // check whether this is a candidate or not
  if (MO.isGlobal()) {
    const GlobalValue *GVal = MO.getGlobal();
    auto *GVar = dyn_cast<GlobalVariable>(GVal);
    if (!GVar) {
      // Not a global variable. Maybe an extern function reference.
      processFuncPrototypes(dyn_cast<Function>(GVal));
      return;
    }

    if (!GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr) &&
        !GVar->hasAttribute(BPFCoreSharedInfo::TypeIdAttr))
      return;

    MDNode *MDN = GVar->getMetadata(LLVMContext::MD_preserve_access_index);
    const auto *Ty = dyn_cast_or_null<DIType>(MDN);
    const std::optional RootId = visitTypeEntry(Ty);
    if (!RootId.has_value() || RootId.value() == 0) {
      OS.getContext().reportError({}, "CO-RE relocation has unrepresentable "
                                      "BTF root type '" +
                                          getDITypeName(Ty) + "'");
      return;
    }
    for (BTFTypeBase &TypeEntry : make_pointee_range(TypeEntries))
      TypeEntry.completeType(*this);
    MCSymbol *ORSym = OS.getContext().createTempSymbol();
    OS.emitLabel(ORSym);
    generatePatchImmReloc(ORSym, RootId.value(), GVar,
                          GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr));
  }
}

void BTFDebug::beginInstruction(const MachineInstr *MI) {
  DebugHandlerBase::beginInstruction(MI);

  if (SkipInstruction || MI->isMetaInstruction() ||
      MI->getFlag(MachineInstr::FrameSetup))
    return;

  if (MI->isInlineAsm()) {
    // Count the number of register definitions to find the asm string.
    unsigned NumDefs = 0;
    while (true) {
      const MachineOperand &MO = MI->getOperand(NumDefs);
      if (MO.isReg() && MO.isDef()) {
        ++NumDefs;
        continue;
      }
      // Skip this inline asm instruction if the asmstr is empty.
      const char *AsmStr = MO.getSymbolName();
      if (AsmStr[0] == 0)
        return;
      break;
    }
  }

  if (MI->getOpcode() == BPF::LD_imm64) {
    // If the insn is "r2 = LD_imm64 @<an AmaAttr global>",
    // add this insn into the .BTF.ext FieldReloc subsection.
    // Relocation looks like:
    //  . SecName:
    //    . InstOffset
    //    . TypeID
    //    . OffSetNameOff
    //    . RelocType
    // Later, the insn is replaced with "r2 = <offset>"
    // where "<offset>" equals to the offset based on current
    // type definitions.
    //
    // If the insn is "r2 = LD_imm64 @<an TypeIdAttr global>",
    // The LD_imm64 result will be replaced with a btf type id.
    processGlobalValue(MI->getOperand(1));
  } else if (MI->getOpcode() == BPF::CORE_LD64 ||
             MI->getOpcode() == BPF::CORE_LD32 ||
             MI->getOpcode() == BPF::CORE_ST ||
             MI->getOpcode() == BPF::CORE_SHIFT) {
    // relocation insn is a load, store or shift insn.
    processGlobalValue(MI->getOperand(3));
  } else if (MI->getOpcode() == BPF::JAL) {
    // check extern function references
    const MachineOperand &MO = MI->getOperand(0);
    if (MO.isGlobal()) {
      processFuncPrototypes(dyn_cast<Function>(MO.getGlobal()));
    }
  }

  if (!CurMI) // no debug info
    return;

  // Skip this instruction if no DebugLoc, the DebugLoc
  // is the same as the previous instruction or Line is 0.
  const DebugLoc &DL = MI->getDebugLoc();
  if (!DL || PrevInstLoc == DL || DL.getLine() == 0) {
    // This instruction will be skipped, no LineInfo has
    // been generated, construct one based on function signature.
    if (LineInfoGenerated == false) {
      auto *S = MI->getMF()->getFunction().getSubprogram();
      if (!S)
        return;
      MCSymbol *FuncLabel = Asm->getFunctionBegin();
      constructLineInfo(FuncLabel, S->getFile(), S->getLine(), 0);
      LineInfoGenerated = true;
    }

    return;
  }

  // Create a temporary label to remember the insn for lineinfo.
  MCSymbol *LineSym = OS.getContext().createTempSymbol();
  OS.emitLabel(LineSym);

  // Construct the lineinfo.
  constructLineInfo(LineSym, DL->getFile(), DL.getLine(), DL.getCol());

  LineInfoGenerated = true;
  PrevInstLoc = DL;
}

void BTFDebug::processGlobals(bool ProcessingMapDef) {
  // Collect all types referenced by globals.
  const Module *M = MMI->getModule();
  for (const GlobalVariable &Global : M->globals()) {
    // Decide the section name.
    StringRef SecName;
    std::optional<SectionKind> GVKind;

    if (!Global.isDeclarationForLinker())
      GVKind = TargetLoweringObjectFile::getKindForGlobal(&Global, Asm->TM);

    if (!GVKind.has_value())
      SecName = Global.hasSection() ? Global.getSection() : "";
    else if (GVKind.value().isCommon())
      SecName = ".bss";
    else {
      TargetLoweringObjectFile *TLOF = Asm->TM.getObjFileLowering();
      MCSection *Sec = TLOF->SectionForGlobal(&Global, Asm->TM);
      SecName = Sec->getName();
    }

    if (ProcessingMapDef != SecName.starts_with(".maps"))
      continue;

    // Create a .rodata datasec if the global variable is an initialized
    // constant with private linkage and if it won't be in .rodata.str<#>
    // and .rodata.cst<#> sections.
    if (SecName == ".rodata" && Global.hasPrivateLinkage() &&
        DataSecEntries.find(SecName) == DataSecEntries.end()) {
      // skip .rodata.str<#> and .rodata.cst<#> sections
      const SectionKind Kind = GVKind.value();
      if (!Kind.isMergeableCString() && !Kind.isMergeableConst()) {
        DataSecEntries[std::string(SecName)] =
            std::make_unique<BTFKindDataSec>(Asm, std::string(SecName));
      }
    }

    SmallVector<DIGlobalVariableExpression *, 1> GVs;
    Global.getDebugInfo(GVs);

    // No type information, mostly internal, skip it.
    if (GVs.size() == 0)
      continue;

    const DIGlobalVariable *DIGlobal = GVs.front()->getVariable();
    const DIType *Ty = DIGlobal->getType();
    const std::optional GVTypeId =
        ProcessingMapDef ? visitMapDefType(Ty)
                         : visitTypeEntry(tryRemoveAtomicType(Ty));

    // Only support the following globals:
    //  . static variables
    //  . non-static weak or non-weak global variables
    //  . weak or non-weak extern global variables
    // Whether DataSec is readonly or not can be found from corresponding ELF
    // section flags. Whether a BTF_KIND_VAR is a weak symbol or not
    // can be found from the corresponding ELF symbol table.
    auto Linkage = Global.getLinkage();
    if (Linkage != GlobalValue::InternalLinkage &&
        Linkage != GlobalValue::ExternalLinkage &&
        Linkage != GlobalValue::WeakAnyLinkage &&
        Linkage != GlobalValue::WeakODRLinkage &&
        Linkage != GlobalValue::ExternalWeakLinkage)
      continue;

    if (!GVTypeId.has_value() || GVTypeId.value() == 0) {
      if (ProcessingMapDef)
        OS.getContext().reportError({}, "BTF map '" + Global.getName() +
                                            "' has unrepresentable type '" +
                                            getDITypeName(Ty) + "'");
      if (Global.hasInitializer())
        processGlobalInitializer(Global.getInitializer());
      continue;
    }

    uint32_t GVarInfo;
    if (Linkage == GlobalValue::InternalLinkage) {
      GVarInfo = BTF::VAR_STATIC;
    } else if (Global.hasInitializer()) {
      GVarInfo = BTF::VAR_GLOBAL_ALLOCATED;
    } else {
      GVarInfo = BTF::VAR_GLOBAL_EXTERNAL;
    }

    const uint32_t VarId = addType(std::make_unique<BTFKindVar>(
        Global.getName(), GVTypeId.value(), GVarInfo));

    processDeclAnnotations(DIGlobal->getAnnotations(), VarId, -1);

    // An empty SecName means an extern variable without section attribute.
    if (SecName.empty())
      continue;

    // Find or create a DataSec
    auto [It, Inserted] = DataSecEntries.try_emplace(std::string(SecName));
    if (Inserted)
      It->second = std::make_unique<BTFKindDataSec>(Asm, std::string(SecName));

    // Calculate symbol size
    const DataLayout &DL = Global.getDataLayout();
    uint32_t Size = Global.getGlobalSize(DL);

    It->second->addDataSecEntry(VarId, Asm->getSymbol(&Global), Size);

    if (Global.hasInitializer())
      processGlobalInitializer(Global.getInitializer());
  }
}

/// Process global variable initializer in pursuit for function
/// pointers. Add discovered (extern) functions to BTF. Some (extern)
/// functions might have been missed otherwise. Every symbol needs BTF
/// info when linking with bpftool. Primary use case: "static"
/// initialization of BPF maps.
///
/// struct {
///   __uint(type, BPF_MAP_TYPE_PROG_ARRAY);
///   ...
/// } prog_map SEC(".maps") = { .values = { extern_func } };
///
void BTFDebug::processGlobalInitializer(const Constant *C) {
  if (auto *Fn = dyn_cast<Function>(C))
    processFuncPrototypes(Fn);
  if (auto *CA = dyn_cast<ConstantAggregate>(C)) {
    for (unsigned I = 0, N = CA->getNumOperands(); I < N; ++I)
      processGlobalInitializer(CA->getOperand(I));
  }
}

/// Emit proper patchable instructions.
bool BTFDebug::InstLower(const MachineInstr *MI, MCInst &OutMI) {
  if (MI->getOpcode() == BPF::LD_imm64) {
    const MachineOperand &MO = MI->getOperand(1);
    if (MO.isGlobal()) {
      const GlobalValue *GVal = MO.getGlobal();
      auto *GVar = dyn_cast<GlobalVariable>(GVal);
      if (GVar) {
        if (!GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr) &&
            !GVar->hasAttribute(BPFCoreSharedInfo::TypeIdAttr))
          return false;

        // Emit "mov ri, <imm>"
        auto [Imm, Reloc] = PatchImms[GVar];
        if (Reloc == BTF::ENUM_VALUE_EXISTENCE || Reloc == BTF::ENUM_VALUE ||
            Reloc == BTF::BTF_TYPE_ID_LOCAL || Reloc == BTF::BTF_TYPE_ID_REMOTE)
          OutMI.setOpcode(BPF::LD_imm64);
        else
          OutMI.setOpcode(BPF::MOV_ri);
        OutMI.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
        OutMI.addOperand(MCOperand::createImm(Imm));
        return true;
      }
    }
  } else if (MI->getOpcode() == BPF::CORE_LD64 ||
             MI->getOpcode() == BPF::CORE_LD32 ||
             MI->getOpcode() == BPF::CORE_ST ||
             MI->getOpcode() == BPF::CORE_SHIFT) {
    const MachineOperand &MO = MI->getOperand(3);
    if (MO.isGlobal()) {
      const GlobalValue *GVal = MO.getGlobal();
      auto *GVar = dyn_cast<GlobalVariable>(GVal);
      if (GVar && GVar->hasAttribute(BPFCoreSharedInfo::AmaAttr)) {
        uint32_t Imm = PatchImms[GVar].first;
        OutMI.setOpcode(MI->getOperand(1).getImm());
        if (MI->getOperand(0).isImm())
          OutMI.addOperand(MCOperand::createImm(MI->getOperand(0).getImm()));
        else
          OutMI.addOperand(MCOperand::createReg(MI->getOperand(0).getReg()));
        OutMI.addOperand(MCOperand::createReg(MI->getOperand(2).getReg()));
        OutMI.addOperand(MCOperand::createImm(Imm));
        return true;
      }
    }
  }
  return false;
}

void BTFDebug::processFuncPrototypes(const Function *F) {
  if (!F)
    return;

  const DISubprogram *SP = F->getSubprogram();
  if (!SP || SP->isDefinition())
    return;

  // Do not emit again if already emitted.
  if (!ProtoFunctions.insert(F).second)
    return;

  const std::optional ProtoTypeId =
      visitSubroutineType(SP->getType(), false, {});
  if (!ProtoTypeId.has_value())
    return;
  const uint32_t FuncId =
      processDISubprogram(SP, ProtoTypeId.value(), BTF::FUNC_EXTERN);

  if (F->hasSection()) {
    StringRef SecName = F->getSection();

    auto [It, Inserted] = DataSecEntries.try_emplace(std::string(SecName));
    if (Inserted)
      It->second = std::make_unique<BTFKindDataSec>(Asm, std::string(SecName));

    // We really don't know func size, set it to 0.
    It->second->addDataSecEntry(FuncId, Asm->getSymbol(F), 0);
  }
}

void BTFDebug::endModule() {
  // Collect MapDef globals if not collected yet.
  if (MapDefNotCollected) {
    processGlobals(true);
    MapDefNotCollected = false;
  }

  // Collect global types/variables except MapDef globals.
  processGlobals(false);

  // In case that BPF_TRAP usage is removed during machine-level optimization,
  // generate btf for BPF_TRAP function here.
  for (const Function &F : *MMI->getModule()) {
    if (F.getName() == BPF_TRAP)
      processFuncPrototypes(&F);
  }

  for (auto &DataSec : DataSecEntries)
    addType(std::move(DataSec.second));

  // Fixups
  for (auto &Fixup : FixupDerivedTypes) {
    const DICompositeType *CTy = Fixup.first;
    StringRef TypeName = CTy->getName();
    bool IsUnion = CTy->getTag() == dwarf::DW_TAG_union_type;

    // Search through struct types
    uint32_t StructTypeId = 0;
    for (const auto &StructType : StructTypes) {
      if (StructType->getName() == TypeName) {
        StructTypeId = StructType->getId();
        break;
      }
    }

    if (StructTypeId == 0)
      StructTypeId = addType(std::make_unique<BTFTypeFwd>(TypeName, IsUnion));

    for (auto &TypeInfo : Fixup.second) {
      const DIDerivedType *DTy = TypeInfo.first;
      BTFTypeDerived *BDType = TypeInfo.second;

      BDType->setPointeeType(
          genBTFTypeTags(DTy, StructTypeId).value_or(StructTypeId));
    }
  }

  // The kernel requires a nonempty base BTF type section for line info.
  if (TypeEntries.empty() && !LineInfoTable.empty())
    addType(
        std::make_unique<BTFTypeDerived>(0, BTF::BTF_KIND_PTR, StringRef()));

  // Complete BTF type cross refereences.
  for (BTFTypeBase &TypeEntry : make_pointee_range(TypeEntries))
    TypeEntry.completeType(*this);

  // Emit BTF sections.
  emitBTFSection();
  emitBTFExtSection();
}
