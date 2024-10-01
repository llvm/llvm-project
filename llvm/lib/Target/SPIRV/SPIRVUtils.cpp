//===--- SPIRVUtils.cpp ---- SPIR-V Utility Functions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions.
//
//===----------------------------------------------------------------------===//

#include "SPIRVUtils.h"
#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "SPIRV.h"
#include "SPIRVInstrInfo.h"
#include "SPIRVSubtarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include <queue>
#include <vector>

namespace llvm {

// The following functions are used to add these string literals as a series of
// 32-bit integer operands with the correct format, and unpack them if necessary
// when making string comparisons in compiler passes.
// SPIR-V requires null-terminated UTF-8 strings padded to 32-bit alignment.
static uint32_t convertCharsToWord(const StringRef &Str, unsigned i) {
  uint32_t Word = 0u; // Build up this 32-bit word from 4 8-bit chars.
  for (unsigned WordIndex = 0; WordIndex < 4; ++WordIndex) {
    unsigned StrIndex = i + WordIndex;
    uint8_t CharToAdd = 0;       // Initilize char as padding/null.
    if (StrIndex < Str.size()) { // If it's within the string, get a real char.
      CharToAdd = Str[StrIndex];
    }
    Word |= (CharToAdd << (WordIndex * 8));
  }
  return Word;
}

// Get length including padding and null terminator.
static size_t getPaddedLen(const StringRef &Str) {
  return (Str.size() + 4) & ~3;
}

void addStringImm(const StringRef &Str, MCInst &Inst) {
  const size_t PaddedLen = getPaddedLen(Str);
  for (unsigned i = 0; i < PaddedLen; i += 4) {
    // Add an operand for the 32-bits of chars or padding.
    Inst.addOperand(MCOperand::createImm(convertCharsToWord(Str, i)));
  }
}

void addStringImm(const StringRef &Str, MachineInstrBuilder &MIB) {
  const size_t PaddedLen = getPaddedLen(Str);
  for (unsigned i = 0; i < PaddedLen; i += 4) {
    // Add an operand for the 32-bits of chars or padding.
    MIB.addImm(convertCharsToWord(Str, i));
  }
}

void addStringImm(const StringRef &Str, IRBuilder<> &B,
                  std::vector<Value *> &Args) {
  const size_t PaddedLen = getPaddedLen(Str);
  for (unsigned i = 0; i < PaddedLen; i += 4) {
    // Add a vector element for the 32-bits of chars or padding.
    Args.push_back(B.getInt32(convertCharsToWord(Str, i)));
  }
}

std::string getStringImm(const MachineInstr &MI, unsigned StartIndex) {
  return getSPIRVStringOperand(MI, StartIndex);
}

void addNumImm(const APInt &Imm, MachineInstrBuilder &MIB) {
  const auto Bitwidth = Imm.getBitWidth();
  if (Bitwidth == 1)
    return; // Already handled
  else if (Bitwidth <= 32) {
    MIB.addImm(Imm.getZExtValue());
    // Asm Printer needs this info to print floating-type correctly
    if (Bitwidth == 16)
      MIB.getInstr()->setAsmPrinterFlag(SPIRV::ASM_PRINTER_WIDTH16);
    return;
  } else if (Bitwidth <= 64) {
    uint64_t FullImm = Imm.getZExtValue();
    uint32_t LowBits = FullImm & 0xffffffff;
    uint32_t HighBits = (FullImm >> 32) & 0xffffffff;
    MIB.addImm(LowBits).addImm(HighBits);
    return;
  }
  report_fatal_error("Unsupported constant bitwidth");
}

void buildOpName(Register Target, const StringRef &Name,
                 MachineIRBuilder &MIRBuilder) {
  if (!Name.empty()) {
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpName).addUse(Target);
    addStringImm(Name, MIB);
  }
}

static void finishBuildOpDecorate(MachineInstrBuilder &MIB,
                                  const std::vector<uint32_t> &DecArgs,
                                  StringRef StrImm) {
  if (!StrImm.empty())
    addStringImm(StrImm, MIB);
  for (const auto &DecArg : DecArgs)
    MIB.addImm(DecArg);
}

void buildOpDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                     SPIRV::Decoration::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs, StringRef StrImm) {
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpDecorate)
                 .addUse(Reg)
                 .addImm(static_cast<uint32_t>(Dec));
  finishBuildOpDecorate(MIB, DecArgs, StrImm);
}

void buildOpDecorate(Register Reg, MachineInstr &I, const SPIRVInstrInfo &TII,
                     SPIRV::Decoration::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs, StringRef StrImm) {
  MachineBasicBlock &MBB = *I.getParent();
  auto MIB = BuildMI(MBB, I, I.getDebugLoc(), TII.get(SPIRV::OpDecorate))
                 .addUse(Reg)
                 .addImm(static_cast<uint32_t>(Dec));
  finishBuildOpDecorate(MIB, DecArgs, StrImm);
}

void buildOpSpirvDecorations(Register Reg, MachineIRBuilder &MIRBuilder,
                             const MDNode *GVarMD) {
  for (unsigned I = 0, E = GVarMD->getNumOperands(); I != E; ++I) {
    auto *OpMD = dyn_cast<MDNode>(GVarMD->getOperand(I));
    if (!OpMD)
      report_fatal_error("Invalid decoration");
    if (OpMD->getNumOperands() == 0)
      report_fatal_error("Expect operand(s) of the decoration");
    ConstantInt *DecorationId =
        mdconst::dyn_extract<ConstantInt>(OpMD->getOperand(0));
    if (!DecorationId)
      report_fatal_error("Expect SPIR-V <Decoration> operand to be the first "
                         "element of the decoration");
    auto MIB = MIRBuilder.buildInstr(SPIRV::OpDecorate)
                   .addUse(Reg)
                   .addImm(static_cast<uint32_t>(DecorationId->getZExtValue()));
    for (unsigned OpI = 1, OpE = OpMD->getNumOperands(); OpI != OpE; ++OpI) {
      if (ConstantInt *OpV =
              mdconst::dyn_extract<ConstantInt>(OpMD->getOperand(OpI)))
        MIB.addImm(static_cast<uint32_t>(OpV->getZExtValue()));
      else if (MDString *OpV = dyn_cast<MDString>(OpMD->getOperand(OpI)))
        addStringImm(OpV->getString(), MIB);
      else
        report_fatal_error("Unexpected operand of the decoration");
    }
  }
}

SPIRV::StorageClass::StorageClass
addressSpaceToStorageClass(unsigned AddrSpace, const SPIRVSubtarget &STI) {
  switch (AddrSpace) {
  case 0:
    return SPIRV::StorageClass::Function;
  case 1:
    return SPIRV::StorageClass::CrossWorkgroup;
  case 2:
    return SPIRV::StorageClass::UniformConstant;
  case 3:
    return SPIRV::StorageClass::Workgroup;
  case 4:
    return SPIRV::StorageClass::Generic;
  case 5:
    return STI.canUseExtension(SPIRV::Extension::SPV_INTEL_usm_storage_classes)
               ? SPIRV::StorageClass::DeviceOnlyINTEL
               : SPIRV::StorageClass::CrossWorkgroup;
  case 6:
    return STI.canUseExtension(SPIRV::Extension::SPV_INTEL_usm_storage_classes)
               ? SPIRV::StorageClass::HostOnlyINTEL
               : SPIRV::StorageClass::CrossWorkgroup;
  case 7:
    return SPIRV::StorageClass::Input;
  default:
    report_fatal_error("Unknown address space");
  }
}

SPIRV::MemorySemantics::MemorySemantics
getMemSemanticsForStorageClass(SPIRV::StorageClass::StorageClass SC) {
  switch (SC) {
  case SPIRV::StorageClass::StorageBuffer:
  case SPIRV::StorageClass::Uniform:
    return SPIRV::MemorySemantics::UniformMemory;
  case SPIRV::StorageClass::Workgroup:
    return SPIRV::MemorySemantics::WorkgroupMemory;
  case SPIRV::StorageClass::CrossWorkgroup:
    return SPIRV::MemorySemantics::CrossWorkgroupMemory;
  case SPIRV::StorageClass::AtomicCounter:
    return SPIRV::MemorySemantics::AtomicCounterMemory;
  case SPIRV::StorageClass::Image:
    return SPIRV::MemorySemantics::ImageMemory;
  default:
    return SPIRV::MemorySemantics::None;
  }
}

SPIRV::MemorySemantics::MemorySemantics getMemSemantics(AtomicOrdering Ord) {
  switch (Ord) {
  case AtomicOrdering::Acquire:
    return SPIRV::MemorySemantics::Acquire;
  case AtomicOrdering::Release:
    return SPIRV::MemorySemantics::Release;
  case AtomicOrdering::AcquireRelease:
    return SPIRV::MemorySemantics::AcquireRelease;
  case AtomicOrdering::SequentiallyConsistent:
    return SPIRV::MemorySemantics::SequentiallyConsistent;
  case AtomicOrdering::Unordered:
  case AtomicOrdering::Monotonic:
  case AtomicOrdering::NotAtomic:
    return SPIRV::MemorySemantics::None;
  }
  llvm_unreachable(nullptr);
}

SPIRV::Scope::Scope getMemScope(LLVMContext &Ctx, SyncScope::ID Id) {
  // Named by
  // https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_scope_id.
  // We don't need aliases for Invocation and CrossDevice, as we already have
  // them covered by "singlethread" and "" strings respectively (see
  // implementation of LLVMContext::LLVMContext()).
  static const llvm::SyncScope::ID SubGroup =
      Ctx.getOrInsertSyncScopeID("subgroup");
  static const llvm::SyncScope::ID WorkGroup =
      Ctx.getOrInsertSyncScopeID("workgroup");
  static const llvm::SyncScope::ID Device =
      Ctx.getOrInsertSyncScopeID("device");

  if (Id == llvm::SyncScope::SingleThread)
    return SPIRV::Scope::Invocation;
  else if (Id == llvm::SyncScope::System)
    return SPIRV::Scope::CrossDevice;
  else if (Id == SubGroup)
    return SPIRV::Scope::Subgroup;
  else if (Id == WorkGroup)
    return SPIRV::Scope::Workgroup;
  else if (Id == Device)
    return SPIRV::Scope::Device;
  return SPIRV::Scope::CrossDevice;
}

MachineInstr *getDefInstrMaybeConstant(Register &ConstReg,
                                       const MachineRegisterInfo *MRI) {
  MachineInstr *MI = MRI->getVRegDef(ConstReg);
  MachineInstr *ConstInstr =
      MI->getOpcode() == SPIRV::G_TRUNC || MI->getOpcode() == SPIRV::G_ZEXT
          ? MRI->getVRegDef(MI->getOperand(1).getReg())
          : MI;
  if (auto *GI = dyn_cast<GIntrinsic>(ConstInstr)) {
    if (GI->is(Intrinsic::spv_track_constant)) {
      ConstReg = ConstInstr->getOperand(2).getReg();
      return MRI->getVRegDef(ConstReg);
    }
  } else if (ConstInstr->getOpcode() == SPIRV::ASSIGN_TYPE) {
    ConstReg = ConstInstr->getOperand(1).getReg();
    return MRI->getVRegDef(ConstReg);
  }
  return MRI->getVRegDef(ConstReg);
}

uint64_t getIConstVal(Register ConstReg, const MachineRegisterInfo *MRI) {
  const MachineInstr *MI = getDefInstrMaybeConstant(ConstReg, MRI);
  assert(MI && MI->getOpcode() == TargetOpcode::G_CONSTANT);
  return MI->getOperand(1).getCImm()->getValue().getZExtValue();
}

bool isSpvIntrinsic(const MachineInstr &MI, Intrinsic::ID IntrinsicID) {
  if (const auto *GI = dyn_cast<GIntrinsic>(&MI))
    return GI->is(IntrinsicID);
  return false;
}

Type *getMDOperandAsType(const MDNode *N, unsigned I) {
  Type *ElementTy = cast<ValueAsMetadata>(N->getOperand(I))->getType();
  return toTypedPointer(ElementTy);
}

// The set of names is borrowed from the SPIR-V translator.
// TODO: may be implemented in SPIRVBuiltins.td.
static bool isPipeOrAddressSpaceCastBI(const StringRef MangledName) {
  return MangledName == "write_pipe_2" || MangledName == "read_pipe_2" ||
         MangledName == "write_pipe_2_bl" || MangledName == "read_pipe_2_bl" ||
         MangledName == "write_pipe_4" || MangledName == "read_pipe_4" ||
         MangledName == "reserve_write_pipe" ||
         MangledName == "reserve_read_pipe" ||
         MangledName == "commit_write_pipe" ||
         MangledName == "commit_read_pipe" ||
         MangledName == "work_group_reserve_write_pipe" ||
         MangledName == "work_group_reserve_read_pipe" ||
         MangledName == "work_group_commit_write_pipe" ||
         MangledName == "work_group_commit_read_pipe" ||
         MangledName == "get_pipe_num_packets_ro" ||
         MangledName == "get_pipe_max_packets_ro" ||
         MangledName == "get_pipe_num_packets_wo" ||
         MangledName == "get_pipe_max_packets_wo" ||
         MangledName == "sub_group_reserve_write_pipe" ||
         MangledName == "sub_group_reserve_read_pipe" ||
         MangledName == "sub_group_commit_write_pipe" ||
         MangledName == "sub_group_commit_read_pipe" ||
         MangledName == "to_global" || MangledName == "to_local" ||
         MangledName == "to_private";
}

static bool isEnqueueKernelBI(const StringRef MangledName) {
  return MangledName == "__enqueue_kernel_basic" ||
         MangledName == "__enqueue_kernel_basic_events" ||
         MangledName == "__enqueue_kernel_varargs" ||
         MangledName == "__enqueue_kernel_events_varargs";
}

static bool isKernelQueryBI(const StringRef MangledName) {
  return MangledName == "__get_kernel_work_group_size_impl" ||
         MangledName == "__get_kernel_sub_group_count_for_ndrange_impl" ||
         MangledName == "__get_kernel_max_sub_group_size_for_ndrange_impl" ||
         MangledName == "__get_kernel_preferred_work_group_size_multiple_impl";
}

static bool isNonMangledOCLBuiltin(StringRef Name) {
  if (!Name.starts_with("__"))
    return false;

  return isEnqueueKernelBI(Name) || isKernelQueryBI(Name) ||
         isPipeOrAddressSpaceCastBI(Name.drop_front(2)) ||
         Name == "__translate_sampler_initializer";
}

std::string getOclOrSpirvBuiltinDemangledName(StringRef Name) {
  bool IsNonMangledOCL = isNonMangledOCLBuiltin(Name);
  bool IsNonMangledSPIRV = Name.starts_with("__spirv_");
  bool IsNonMangledHLSL = Name.starts_with("__hlsl_");
  bool IsMangled = Name.starts_with("_Z");

  // Otherwise use simple demangling to return the function name.
  if (IsNonMangledOCL || IsNonMangledSPIRV || IsNonMangledHLSL || !IsMangled)
    return Name.str();

  // Try to use the itanium demangler.
  if (char *DemangledName = itaniumDemangle(Name.data())) {
    std::string Result = DemangledName;
    free(DemangledName);
    return Result;
  }

  // Autocheck C++, maybe need to do explicit check of the source language.
  // OpenCL C++ built-ins are declared in cl namespace.
  // TODO: consider using 'St' abbriviation for cl namespace mangling.
  // Similar to ::std:: in C++.
  size_t Start, Len = 0;
  size_t DemangledNameLenStart = 2;
  if (Name.starts_with("_ZN")) {
    // Skip CV and ref qualifiers.
    size_t NameSpaceStart = Name.find_first_not_of("rVKRO", 3);
    // All built-ins are in the ::cl:: namespace.
    if (Name.substr(NameSpaceStart, 11) != "2cl7__spirv")
      return std::string();
    DemangledNameLenStart = NameSpaceStart + 11;
  }
  Start = Name.find_first_not_of("0123456789", DemangledNameLenStart);
  Name.substr(DemangledNameLenStart, Start - DemangledNameLenStart)
      .getAsInteger(10, Len);
  return Name.substr(Start, Len).str();
}

bool hasBuiltinTypePrefix(StringRef Name) {
  if (Name.starts_with("opencl.") || Name.starts_with("ocl_") ||
      Name.starts_with("spirv."))
    return true;
  return false;
}

bool isSpecialOpaqueType(const Type *Ty) {
  if (const TargetExtType *EType = dyn_cast<TargetExtType>(Ty))
    return hasBuiltinTypePrefix(EType->getName());

  return false;
}

bool isEntryPoint(const Function &F) {
  // OpenCL handling: any function with the SPIR_KERNEL
  // calling convention will be a potential entry point.
  if (F.getCallingConv() == CallingConv::SPIR_KERNEL)
    return true;

  // HLSL handling: special attribute are emitted from the
  // front-end.
  if (F.getFnAttribute("hlsl.shader").isValid())
    return true;

  return false;
}

Type *parseBasicTypeName(StringRef &TypeName, LLVMContext &Ctx) {
  TypeName.consume_front("atomic_");
  if (TypeName.consume_front("void"))
    return Type::getVoidTy(Ctx);
  else if (TypeName.consume_front("bool"))
    return Type::getIntNTy(Ctx, 1);
  else if (TypeName.consume_front("char") ||
           TypeName.consume_front("unsigned char") ||
           TypeName.consume_front("uchar"))
    return Type::getInt8Ty(Ctx);
  else if (TypeName.consume_front("short") ||
           TypeName.consume_front("unsigned short") ||
           TypeName.consume_front("ushort"))
    return Type::getInt16Ty(Ctx);
  else if (TypeName.consume_front("int") ||
           TypeName.consume_front("unsigned int") ||
           TypeName.consume_front("uint"))
    return Type::getInt32Ty(Ctx);
  else if (TypeName.consume_front("long") ||
           TypeName.consume_front("unsigned long") ||
           TypeName.consume_front("ulong"))
    return Type::getInt64Ty(Ctx);
  else if (TypeName.consume_front("half"))
    return Type::getHalfTy(Ctx);
  else if (TypeName.consume_front("float"))
    return Type::getFloatTy(Ctx);
  else if (TypeName.consume_front("double"))
    return Type::getDoubleTy(Ctx);

  // Unable to recognize SPIRV type name
  return nullptr;
}

std::unordered_set<BasicBlock *>
PartialOrderingVisitor::getReachableFrom(BasicBlock *Start) {
  std::queue<BasicBlock *> ToVisit;
  ToVisit.push(Start);

  std::unordered_set<BasicBlock *> Output;
  while (ToVisit.size() != 0) {
    BasicBlock *BB = ToVisit.front();
    ToVisit.pop();

    if (Output.count(BB) != 0)
      continue;
    Output.insert(BB);

    for (BasicBlock *Successor : successors(BB)) {
      if (DT.dominates(Successor, BB))
        continue;
      ToVisit.push(Successor);
    }
  }

  return Output;
}

size_t PartialOrderingVisitor::visit(BasicBlock *BB, size_t Rank) {
  if (Visited.count(BB) != 0)
    return Rank;

  Loop *L = LI.getLoopFor(BB);
  const bool isLoopHeader = LI.isLoopHeader(BB);

  if (BlockToOrder.count(BB) == 0) {
    OrderInfo Info = {Rank, Visited.size()};
    BlockToOrder.emplace(BB, Info);
  } else {
    BlockToOrder[BB].Rank = std::max(BlockToOrder[BB].Rank, Rank);
  }

  for (BasicBlock *Predecessor : predecessors(BB)) {
    if (isLoopHeader && L->contains(Predecessor)) {
      continue;
    }

    if (BlockToOrder.count(Predecessor) == 0) {
      return Rank;
    }
  }

  Visited.insert(BB);

  SmallVector<BasicBlock *, 2> OtherSuccessors;
  SmallVector<BasicBlock *, 2> LoopSuccessors;

  for (BasicBlock *Successor : successors(BB)) {
    // Ignoring back-edges.
    if (DT.dominates(Successor, BB))
      continue;

    if (isLoopHeader && L->contains(Successor)) {
      LoopSuccessors.push_back(Successor);
    } else
      OtherSuccessors.push_back(Successor);
  }

  for (BasicBlock *BB : LoopSuccessors)
    Rank = std::max(Rank, visit(BB, Rank + 1));

  size_t OutputRank = Rank;
  for (BasicBlock *Item : OtherSuccessors)
    OutputRank = std::max(OutputRank, visit(Item, Rank + 1));
  return OutputRank;
}

PartialOrderingVisitor::PartialOrderingVisitor(Function &F) {
  DT.recalculate(F);
  LI = LoopInfo(DT);

  visit(&*F.begin(), 0);

  Order.reserve(F.size());
  for (auto &[BB, Info] : BlockToOrder)
    Order.emplace_back(BB);

  std::sort(Order.begin(), Order.end(), [&](const auto &LHS, const auto &RHS) {
    return compare(LHS, RHS);
  });
}

bool PartialOrderingVisitor::compare(const BasicBlock *LHS,
                                     const BasicBlock *RHS) const {
  const OrderInfo &InfoLHS = BlockToOrder.at(const_cast<BasicBlock *>(LHS));
  const OrderInfo &InfoRHS = BlockToOrder.at(const_cast<BasicBlock *>(RHS));
  if (InfoLHS.Rank != InfoRHS.Rank)
    return InfoLHS.Rank < InfoRHS.Rank;
  return InfoLHS.TraversalIndex < InfoRHS.TraversalIndex;
}

void PartialOrderingVisitor::partialOrderVisit(
    BasicBlock &Start, std::function<bool(BasicBlock *)> Op) {
  std::unordered_set<BasicBlock *> Reachable = getReachableFrom(&Start);
  assert(BlockToOrder.count(&Start) != 0);

  // Skipping blocks with a rank inferior to |Start|'s rank.
  auto It = Order.begin();
  while (It != Order.end() && *It != &Start)
    ++It;

  // This is unexpected. Worst case |Start| is the last block,
  // so It should point to the last block, not past-end.
  assert(It != Order.end());

  // By default, there is no rank limit. Setting it to the maximum value.
  std::optional<size_t> EndRank = std::nullopt;
  for (; It != Order.end(); ++It) {
    if (EndRank.has_value() && BlockToOrder[*It].Rank > *EndRank)
      break;

    if (Reachable.count(*It) == 0) {
      continue;
    }

    if (!Op(*It)) {
      EndRank = BlockToOrder[*It].Rank;
    }
  }
}

bool sortBlocks(Function &F) {
  if (F.size() == 0)
    return false;

  bool Modified = false;

  std::vector<BasicBlock *> Order;
  Order.reserve(F.size());

  PartialOrderingVisitor Visitor(F);
  Visitor.partialOrderVisit(*F.begin(), [&Order](BasicBlock *Block) {
    Order.push_back(Block);
    return true;
  });

  assert(&*F.begin() == Order[0]);
  BasicBlock *LastBlock = &*F.begin();
  for (BasicBlock *BB : Order) {
    if (BB != LastBlock && &*LastBlock->getNextNode() != BB) {
      Modified = true;
      BB->moveAfter(LastBlock);
    }
    LastBlock = BB;
  }

  return Modified;
}

MachineInstr *getVRegDef(MachineRegisterInfo &MRI, Register Reg) {
  MachineInstr *MaybeDef = MRI.getVRegDef(Reg);
  if (MaybeDef && MaybeDef->getOpcode() == SPIRV::ASSIGN_TYPE)
    MaybeDef = MRI.getVRegDef(MaybeDef->getOperand(1).getReg());
  return MaybeDef;
}

} // namespace llvm
