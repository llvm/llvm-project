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
#include "SPIRVGlobalRegistry.h"
#include "SPIRVInstrInfo.h"
#include "SPIRVSubtarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/IntrinsicInst.h"
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

std::string getStringValueFromReg(Register Reg, MachineRegisterInfo &MRI) {
  MachineInstr *Def = getVRegDef(MRI, Reg);
  assert(Def && Def->getOpcode() == TargetOpcode::G_GLOBAL_VALUE &&
         "Expected G_GLOBAL_VALUE");
  const GlobalValue *GV = Def->getOperand(1).getGlobal();
  Value *V = GV->getOperand(0);
  const ConstantDataArray *CDA = cast<ConstantDataArray>(V);
  return CDA->getAsCString().str();
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

void buildOpName(Register Target, const StringRef &Name, MachineInstr &I,
                 const SPIRVInstrInfo &TII) {
  if (!Name.empty()) {
    auto MIB =
        BuildMI(*I.getParent(), I, I.getDebugLoc(), TII.get(SPIRV::OpName))
            .addUse(Target);
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

void buildOpMemberDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                           SPIRV::Decoration::Decoration Dec, uint32_t Member,
                           const std::vector<uint32_t> &DecArgs,
                           StringRef StrImm) {
  auto MIB = MIRBuilder.buildInstr(SPIRV::OpMemberDecorate)
                 .addUse(Reg)
                 .addImm(Member)
                 .addImm(static_cast<uint32_t>(Dec));
  finishBuildOpDecorate(MIB, DecArgs, StrImm);
}

void buildOpMemberDecorate(Register Reg, MachineInstr &I,
                           const SPIRVInstrInfo &TII,
                           SPIRV::Decoration::Decoration Dec, uint32_t Member,
                           const std::vector<uint32_t> &DecArgs,
                           StringRef StrImm) {
  MachineBasicBlock &MBB = *I.getParent();
  auto MIB = BuildMI(MBB, I, I.getDebugLoc(), TII.get(SPIRV::OpMemberDecorate))
                 .addUse(Reg)
                 .addImm(Member)
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

MachineBasicBlock::iterator getOpVariableMBBIt(MachineInstr &I) {
  MachineFunction *MF = I.getParent()->getParent();
  MachineBasicBlock *MBB = &MF->front();
  MachineBasicBlock::iterator It = MBB->SkipPHIsAndLabels(MBB->begin()),
                              E = MBB->end();
  bool IsHeader = false;
  unsigned Opcode;
  for (; It != E && It != I; ++It) {
    Opcode = It->getOpcode();
    if (Opcode == SPIRV::OpFunction || Opcode == SPIRV::OpFunctionParameter) {
      IsHeader = true;
    } else if (IsHeader &&
               !(Opcode == SPIRV::ASSIGN_TYPE || Opcode == SPIRV::OpLabel)) {
      ++It;
      break;
    }
  }
  return It;
}

MachineBasicBlock::iterator getInsertPtValidEnd(MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator I = MBB->end();
  if (I == MBB->begin())
    return I;
  --I;
  while (I->isTerminator() || I->isDebugValue()) {
    if (I == MBB->begin())
      break;
    --I;
  }
  return I;
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
  case 8:
    return SPIRV::StorageClass::Output;
  case 9:
    return SPIRV::StorageClass::CodeSectionINTEL;
  case 10:
    return SPIRV::StorageClass::Private;
  case 11:
    return SPIRV::StorageClass::StorageBuffer;
  case 12:
    return SPIRV::StorageClass::Uniform;
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
  } else if (ConstInstr->getOpcode() == TargetOpcode::G_CONSTANT ||
             ConstInstr->getOpcode() == TargetOpcode::G_FCONSTANT) {
    ConstReg = ConstInstr->getOperand(0).getReg();
    return ConstInstr;
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
  [[maybe_unused]] bool Error =
      Name.substr(DemangledNameLenStart, Start - DemangledNameLenStart)
          .getAsInteger(10, Len);
  assert(!Error && "Failed to parse demangled name length");
  return Name.substr(Start, Len).str();
}

bool hasBuiltinTypePrefix(StringRef Name) {
  if (Name.starts_with("opencl.") || Name.starts_with("ocl_") ||
      Name.starts_with("spirv."))
    return true;
  return false;
}

bool isSpecialOpaqueType(const Type *Ty) {
  if (const TargetExtType *ExtTy = dyn_cast<TargetExtType>(Ty))
    return isTypedPointerWrapper(ExtTy)
               ? false
               : hasBuiltinTypePrefix(ExtTy->getName());

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
  else if (TypeName.consume_front("bool") || TypeName.consume_front("_Bool"))
    return Type::getIntNTy(Ctx, 1);
  else if (TypeName.consume_front("char") ||
           TypeName.consume_front("signed char") ||
           TypeName.consume_front("unsigned char") ||
           TypeName.consume_front("uchar"))
    return Type::getInt8Ty(Ctx);
  else if (TypeName.consume_front("short") ||
           TypeName.consume_front("signed short") ||
           TypeName.consume_front("unsigned short") ||
           TypeName.consume_front("ushort"))
    return Type::getInt16Ty(Ctx);
  else if (TypeName.consume_front("int") ||
           TypeName.consume_front("signed int") ||
           TypeName.consume_front("unsigned int") ||
           TypeName.consume_front("uint"))
    return Type::getInt32Ty(Ctx);
  else if (TypeName.consume_front("long") ||
           TypeName.consume_front("signed long") ||
           TypeName.consume_front("unsigned long") ||
           TypeName.consume_front("ulong"))
    return Type::getInt64Ty(Ctx);
  else if (TypeName.consume_front("half") ||
           TypeName.consume_front("_Float16") ||
           TypeName.consume_front("__fp16"))
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

bool PartialOrderingVisitor::CanBeVisited(BasicBlock *BB) const {
  for (BasicBlock *P : predecessors(BB)) {
    // Ignore back-edges.
    if (DT.dominates(BB, P))
      continue;

    // One of the predecessor hasn't been visited. Not ready yet.
    if (BlockToOrder.count(P) == 0)
      return false;

    // If the block is a loop exit, the loop must be finished before
    // we can continue.
    Loop *L = LI.getLoopFor(P);
    if (L == nullptr || L->contains(BB))
      continue;

    // SPIR-V requires a single back-edge. And the backend first
    // step transforms loops into the simplified format. If we have
    // more than 1 back-edge, something is wrong.
    assert(L->getNumBackEdges() <= 1);

    // If the loop has no latch, loop's rank won't matter, so we can
    // proceed.
    BasicBlock *Latch = L->getLoopLatch();
    assert(Latch);
    if (Latch == nullptr)
      continue;

    // The latch is not ready yet, let's wait.
    if (BlockToOrder.count(Latch) == 0)
      return false;
  }

  return true;
}

size_t PartialOrderingVisitor::GetNodeRank(BasicBlock *BB) const {
  auto It = BlockToOrder.find(BB);
  if (It != BlockToOrder.end())
    return It->second.Rank;

  size_t result = 0;
  for (BasicBlock *P : predecessors(BB)) {
    // Ignore back-edges.
    if (DT.dominates(BB, P))
      continue;

    auto Iterator = BlockToOrder.end();
    Loop *L = LI.getLoopFor(P);
    BasicBlock *Latch = L ? L->getLoopLatch() : nullptr;

    // If the predecessor is either outside a loop, or part of
    // the same loop, simply take its rank + 1.
    if (L == nullptr || L->contains(BB) || Latch == nullptr) {
      Iterator = BlockToOrder.find(P);
    } else {
      // Otherwise, take the loop's rank (highest rank in the loop) as base.
      // Since loops have a single latch, highest rank is easy to find.
      // If the loop has no latch, then it doesn't matter.
      Iterator = BlockToOrder.find(Latch);
    }

    assert(Iterator != BlockToOrder.end());
    result = std::max(result, Iterator->second.Rank + 1);
  }

  return result;
}

size_t PartialOrderingVisitor::visit(BasicBlock *BB, size_t Unused) {
  ToVisit.push(BB);
  Queued.insert(BB);

  size_t QueueIndex = 0;
  while (ToVisit.size() != 0) {
    BasicBlock *BB = ToVisit.front();
    ToVisit.pop();

    if (!CanBeVisited(BB)) {
      ToVisit.push(BB);
      if (QueueIndex >= ToVisit.size())
        llvm::report_fatal_error(
            "No valid candidate in the queue. Is the graph reducible?");
      QueueIndex++;
      continue;
    }

    QueueIndex = 0;
    size_t Rank = GetNodeRank(BB);
    OrderInfo Info = {Rank, BlockToOrder.size()};
    BlockToOrder.emplace(BB, Info);

    for (BasicBlock *S : successors(BB)) {
      if (Queued.count(S) != 0)
        continue;
      ToVisit.push(S);
      Queued.insert(S);
    }
  }

  return 0;
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

  ReversePostOrderTraversal<Function *> RPOT(&F);
  llvm::append_range(Order, RPOT);

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

bool getVacantFunctionName(Module &M, std::string &Name) {
  // It's a bit of paranoia, but still we don't want to have even a chance that
  // the loop will work for too long.
  constexpr unsigned MaxIters = 1024;
  for (unsigned I = 0; I < MaxIters; ++I) {
    std::string OrdName = Name + Twine(I).str();
    if (!M.getFunction(OrdName)) {
      Name = std::move(OrdName);
      return true;
    }
  }
  return false;
}

// Assign SPIR-V type to the register. If the register has no valid assigned
// class, set register LLT type and class according to the SPIR-V type.
void setRegClassType(Register Reg, SPIRVType *SpvType, SPIRVGlobalRegistry *GR,
                     MachineRegisterInfo *MRI, const MachineFunction &MF,
                     bool Force) {
  GR->assignSPIRVTypeToVReg(SpvType, Reg, MF);
  if (!MRI->getRegClassOrNull(Reg) || Force) {
    MRI->setRegClass(Reg, GR->getRegClass(SpvType));
    MRI->setType(Reg, GR->getRegType(SpvType));
  }
}

// Create a SPIR-V type, assign SPIR-V type to the register. If the register has
// no valid assigned class, set register LLT type and class according to the
// SPIR-V type.
void setRegClassType(Register Reg, const Type *Ty, SPIRVGlobalRegistry *GR,
                     MachineIRBuilder &MIRBuilder,
                     SPIRV::AccessQualifier::AccessQualifier AccessQual,
                     bool EmitIR, bool Force) {
  setRegClassType(Reg,
                  GR->getOrCreateSPIRVType(Ty, MIRBuilder, AccessQual, EmitIR),
                  GR, MIRBuilder.getMRI(), MIRBuilder.getMF(), Force);
}

// Create a virtual register and assign SPIR-V type to the register. Set
// register LLT type and class according to the SPIR-V type.
Register createVirtualRegister(SPIRVType *SpvType, SPIRVGlobalRegistry *GR,
                               MachineRegisterInfo *MRI,
                               const MachineFunction &MF) {
  Register Reg = MRI->createVirtualRegister(GR->getRegClass(SpvType));
  MRI->setType(Reg, GR->getRegType(SpvType));
  GR->assignSPIRVTypeToVReg(SpvType, Reg, MF);
  return Reg;
}

// Create a virtual register and assign SPIR-V type to the register. Set
// register LLT type and class according to the SPIR-V type.
Register createVirtualRegister(SPIRVType *SpvType, SPIRVGlobalRegistry *GR,
                               MachineIRBuilder &MIRBuilder) {
  return createVirtualRegister(SpvType, GR, MIRBuilder.getMRI(),
                               MIRBuilder.getMF());
}

// Create a SPIR-V type, virtual register and assign SPIR-V type to the
// register. Set register LLT type and class according to the SPIR-V type.
Register createVirtualRegister(
    const Type *Ty, SPIRVGlobalRegistry *GR, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual, bool EmitIR) {
  return createVirtualRegister(
      GR->getOrCreateSPIRVType(Ty, MIRBuilder, AccessQual, EmitIR), GR,
      MIRBuilder);
}

CallInst *buildIntrWithMD(Intrinsic::ID IntrID, ArrayRef<Type *> Types,
                          Value *Arg, Value *Arg2, ArrayRef<Constant *> Imms,
                          IRBuilder<> &B) {
  SmallVector<Value *, 4> Args;
  Args.push_back(Arg2);
  Args.push_back(buildMD(Arg));
  llvm::append_range(Args, Imms);
  return B.CreateIntrinsic(IntrID, {Types}, Args);
}

// Return true if there is an opaque pointer type nested in the argument.
bool isNestedPointer(const Type *Ty) {
  if (Ty->isPtrOrPtrVectorTy())
    return true;
  if (const FunctionType *RefTy = dyn_cast<FunctionType>(Ty)) {
    if (isNestedPointer(RefTy->getReturnType()))
      return true;
    for (const Type *ArgTy : RefTy->params())
      if (isNestedPointer(ArgTy))
        return true;
    return false;
  }
  if (const ArrayType *RefTy = dyn_cast<ArrayType>(Ty))
    return isNestedPointer(RefTy->getElementType());
  return false;
}

bool isSpvIntrinsic(const Value *Arg) {
  if (const auto *II = dyn_cast<IntrinsicInst>(Arg))
    if (Function *F = II->getCalledFunction())
      if (F->getName().starts_with("llvm.spv."))
        return true;
  return false;
}

// Function to create continued instructions for SPV_INTEL_long_composites
// extension
SmallVector<MachineInstr *, 4>
createContinuedInstructions(MachineIRBuilder &MIRBuilder, unsigned Opcode,
                            unsigned MinWC, unsigned ContinuedOpcode,
                            ArrayRef<Register> Args, Register ReturnRegister,
                            Register TypeID) {

  SmallVector<MachineInstr *, 4> Instructions;
  constexpr unsigned MaxWordCount = UINT16_MAX;
  const size_t NumElements = Args.size();
  size_t MaxNumElements = MaxWordCount - MinWC;
  size_t SPIRVStructNumElements = NumElements;

  if (NumElements > MaxNumElements) {
    // Do adjustments for continued instructions which always had only one
    // minumum word count.
    SPIRVStructNumElements = MaxNumElements;
    MaxNumElements = MaxWordCount - 1;
  }

  auto MIB =
      MIRBuilder.buildInstr(Opcode).addDef(ReturnRegister).addUse(TypeID);

  for (size_t I = 0; I < SPIRVStructNumElements; ++I)
    MIB.addUse(Args[I]);

  Instructions.push_back(MIB.getInstr());

  for (size_t I = SPIRVStructNumElements; I < NumElements;
       I += MaxNumElements) {
    auto MIB = MIRBuilder.buildInstr(ContinuedOpcode);
    for (size_t J = I; J < std::min(I + MaxNumElements, NumElements); ++J)
      MIB.addUse(Args[J]);
    Instructions.push_back(MIB.getInstr());
  }
  return Instructions;
}

SmallVector<unsigned, 1> getSpirvLoopControlOperandsFromLoopMetadata(Loop *L) {
  unsigned LC = SPIRV::LoopControl::None;
  // Currently used only to store PartialCount value. Later when other
  // LoopControls are added - this map should be sorted before making
  // them loop_merge operands to satisfy 3.23. Loop Control requirements.
  std::vector<std::pair<unsigned, unsigned>> MaskToValueMap;
  if (getBooleanLoopAttribute(L, "llvm.loop.unroll.disable")) {
    LC |= SPIRV::LoopControl::DontUnroll;
  } else {
    if (getBooleanLoopAttribute(L, "llvm.loop.unroll.enable") ||
        getBooleanLoopAttribute(L, "llvm.loop.unroll.full")) {
      LC |= SPIRV::LoopControl::Unroll;
    }
    std::optional<int> Count =
        getOptionalIntLoopAttribute(L, "llvm.loop.unroll.count");
    if (Count && Count != 1) {
      LC |= SPIRV::LoopControl::PartialCount;
      MaskToValueMap.emplace_back(
          std::make_pair(SPIRV::LoopControl::PartialCount, *Count));
    }
  }
  SmallVector<unsigned, 1> Result = {LC};
  for (auto &[Mask, Val] : MaskToValueMap)
    Result.push_back(Val);
  return Result;
}

const std::set<unsigned> &getTypeFoldingSupportedOpcodes() {
  // clang-format off
  static const std::set<unsigned> TypeFoldingSupportingOpcs = {
    TargetOpcode::G_ADD,
    TargetOpcode::G_FADD,
    TargetOpcode::G_STRICT_FADD,
    TargetOpcode::G_SUB,
    TargetOpcode::G_FSUB,
    TargetOpcode::G_STRICT_FSUB,
    TargetOpcode::G_MUL,
    TargetOpcode::G_FMUL,
    TargetOpcode::G_STRICT_FMUL,
    TargetOpcode::G_SDIV,
    TargetOpcode::G_UDIV,
    TargetOpcode::G_FDIV,
    TargetOpcode::G_STRICT_FDIV,
    TargetOpcode::G_SREM,
    TargetOpcode::G_UREM,
    TargetOpcode::G_FREM,
    TargetOpcode::G_STRICT_FREM,
    TargetOpcode::G_FNEG,
    TargetOpcode::G_CONSTANT,
    TargetOpcode::G_FCONSTANT,
    TargetOpcode::G_AND,
    TargetOpcode::G_OR,
    TargetOpcode::G_XOR,
    TargetOpcode::G_SHL,
    TargetOpcode::G_ASHR,
    TargetOpcode::G_LSHR,
    TargetOpcode::G_SELECT,
    TargetOpcode::G_EXTRACT_VECTOR_ELT,
  };
  // clang-format on
  return TypeFoldingSupportingOpcs;
}

bool isTypeFoldingSupported(unsigned Opcode) {
  return getTypeFoldingSupportedOpcodes().count(Opcode) > 0;
}

// Traversing [g]MIR accounting for pseudo-instructions.
MachineInstr *passCopy(MachineInstr *Def, const MachineRegisterInfo *MRI) {
  return (Def->getOpcode() == SPIRV::ASSIGN_TYPE ||
          Def->getOpcode() == TargetOpcode::COPY)
             ? MRI->getVRegDef(Def->getOperand(1).getReg())
             : Def;
}

MachineInstr *getDef(const MachineOperand &MO, const MachineRegisterInfo *MRI) {
  if (MachineInstr *Def = MRI->getVRegDef(MO.getReg()))
    return passCopy(Def, MRI);
  return nullptr;
}

MachineInstr *getImm(const MachineOperand &MO, const MachineRegisterInfo *MRI) {
  if (MachineInstr *Def = getDef(MO, MRI)) {
    if (Def->getOpcode() == TargetOpcode::G_CONSTANT ||
        Def->getOpcode() == SPIRV::OpConstantI)
      return Def;
  }
  return nullptr;
}

int64_t foldImm(const MachineOperand &MO, const MachineRegisterInfo *MRI) {
  if (MachineInstr *Def = getImm(MO, MRI)) {
    if (Def->getOpcode() == SPIRV::OpConstantI)
      return Def->getOperand(2).getImm();
    if (Def->getOpcode() == TargetOpcode::G_CONSTANT)
      return Def->getOperand(1).getCImm()->getZExtValue();
  }
  llvm_unreachable("Unexpected integer constant pattern");
}

unsigned getArrayComponentCount(const MachineRegisterInfo *MRI,
                                const MachineInstr *ResType) {
  return foldImm(ResType->getOperand(2), MRI);
}

MachineBasicBlock::iterator
getFirstValidInstructionInsertPoint(MachineBasicBlock &BB) {
  // Find the position to insert the OpVariable instruction.
  // We will insert it after the last OpFunctionParameter, if any, or
  // after OpFunction otherwise.
  MachineBasicBlock::iterator VarPos = BB.begin();
  while (VarPos != BB.end() && VarPos->getOpcode() != SPIRV::OpFunction) {
    ++VarPos;
  }
  // Advance VarPos to the next instruction after OpFunction, it will either
  // be an OpFunctionParameter, so that we can start the next loop, or the
  // position to insert the OpVariable instruction.
  ++VarPos;
  while (VarPos != BB.end() &&
         VarPos->getOpcode() == SPIRV::OpFunctionParameter) {
    ++VarPos;
  }
  // VarPos is now pointing at after the last OpFunctionParameter, if any,
  // or after OpFunction, if no parameters.
  return VarPos != BB.end() && VarPos->getOpcode() == SPIRV::OpLabel ? ++VarPos
                                                                     : VarPos;
}

} // namespace llvm
