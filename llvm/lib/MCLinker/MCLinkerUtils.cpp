//===--- MCLinkerUtils.cpp - MCLinkerUtils-----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "MCLinkerUtils.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace {

// Helpers to access private field of llvm::MachineModuleInfo::MachineFunctions.
using MFAccessor = llvm::DenseMap<const llvm::Function *,
                                  std::unique_ptr<llvm::MachineFunction>>
    llvm::MachineModuleInfo::*;
MFAccessor getMFAccessor();
template <MFAccessor Instance> struct RobberMFFromMachineModuleInfo {
  friend MFAccessor getMFAccessor() { return Instance; }
};
template struct RobberMFFromMachineModuleInfo<
    &llvm::MachineModuleInfo::MachineFunctions>;

// Helpers to access private field of llvm::MachineFunction::FunctionNumber.
using MFNumberAccessor = unsigned llvm::MachineFunction::*;
MFNumberAccessor getMFNumberAccessor();
template <MFNumberAccessor Instance> struct RobberMFNumberFromMachineFunction {
  friend MFNumberAccessor getMFNumberAccessor() { return Instance; }
};
template struct RobberMFNumberFromMachineFunction<
    &llvm::MachineFunction::FunctionNumber>;

// Helpers to access private field of llvm::MachineFunction::STI.
using STIAccessor = const llvm::TargetSubtargetInfo *llvm::MachineFunction::*;
STIAccessor getSTIAccessor();
template <STIAccessor Instance> struct RobberSTIFromMachineFunction {
  friend STIAccessor getSTIAccessor() { return Instance; }
};
template struct RobberSTIFromMachineFunction<&llvm::MachineFunction::STI>;

// Helpers to access private field of llvm::MachineModuleInfo::NextFnNum.
using NextFnNumAccessor = unsigned llvm::MachineModuleInfo::*;
NextFnNumAccessor getNextFnNumAccessor();
template <NextFnNumAccessor Instance>
struct RobberNextFnNumFromMachineModuleInfo {
  friend NextFnNumAccessor getNextFnNumAccessor() { return Instance; }
};
template struct RobberNextFnNumFromMachineModuleInfo<
    &llvm::MachineModuleInfo::NextFnNum>;

// Helpers to access private field of llvm::TargetMachine::STI.
using MCSubtargetInfoAccessor =
    std::unique_ptr<const llvm::MCSubtargetInfo> llvm::TargetMachine::*;
MCSubtargetInfoAccessor getMCSubtargetInfo();
template <MCSubtargetInfoAccessor Instance>
struct RobberMCSubtargetInfoFromTargetMachine {
  friend MCSubtargetInfoAccessor getMCSubtargetInfo() { return Instance; }
};
template struct RobberMCSubtargetInfoFromTargetMachine<
    &llvm::TargetMachine::STI>;

// Helpers to access private functions
template <typename Tag> struct LLVMPrivateFnAccessor {
  /* export it ... */
  using type = typename Tag::type;
  static type Ptr;
};

template <typename Tag>
typename LLVMPrivateFnAccessor<Tag>::type LLVMPrivateFnAccessor<Tag>::Ptr;

template <typename Tag, typename Tag::type p>
struct LLVMPrivateFnAccessorRob : LLVMPrivateFnAccessor<Tag> {
  /* fill it ... */
  struct Filler {
    Filler() { LLVMPrivateFnAccessor<Tag>::Ptr = p; }
  };
  static Filler FillerObj;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
template <typename Tag, typename Tag::type P>
typename LLVMPrivateFnAccessorRob<Tag, P>::Filler
    LLVMPrivateFnAccessorRob<Tag, P>::FillerObj;
#pragma GCC diagnostic pop

// Helpers to access private functions of llvm::MachineModuleInfo::NextFnNum.
struct MCContextGetSymbolEntryAccessor {
  using type = llvm::MCSymbolTableEntry &(llvm::MCContext::*)(llvm::StringRef);
};
template struct LLVMPrivateFnAccessorRob<MCContextGetSymbolEntryAccessor,
                                         &llvm::MCContext::getSymbolTableEntry>;

// Helpers to access private field of llvm::LLVMTargetMachine::reset.
struct TargetMachineClearSubtargetMapAccessor {
  using type = void (llvm::CodeGenTargetMachineImpl::*)();
};
template struct LLVMPrivateFnAccessorRob<
    TargetMachineClearSubtargetMapAccessor,
    &llvm::CodeGenTargetMachineImpl::reset>;

} // namespace

llvm::DenseMap<const llvm::Function *, std::unique_ptr<llvm::MachineFunction>> &
llvm::mclinker::getMachineFunctionsFromMachineModuleInfo(
    llvm::MachineModuleInfo &MachineModuleInfo) {
  return std::invoke(getMFAccessor(), MachineModuleInfo);
}

void llvm::mclinker::setMachineFunctionNumber(llvm::MachineFunction &Mf,
                                              unsigned Number) {
  unsigned &OrigNumber = std::invoke(getMFNumberAccessor(), Mf);
  OrigNumber = Number;
}

void llvm::mclinker::setNextFnNum(llvm::MachineModuleInfo &MMI,
                                  unsigned Value) {
  unsigned &NextFnNum = std::invoke(getNextFnNumAccessor(), MMI);
  NextFnNum = Value;
}

llvm::MCSymbolTableEntry &
llvm::mclinker::getMCContextSymbolTableEntry(llvm::StringRef Name,
                                             llvm::MCContext &McContext) {
  return (McContext.*
          LLVMPrivateFnAccessor<MCContextGetSymbolEntryAccessor>::Ptr)(Name);
}

void llvm::mclinker::releaseTargetMachineConstants(llvm::TargetMachine &TM) {
  std::unique_ptr<const llvm::MCSubtargetInfo> &McSubtargetInfo =
      std::invoke(getMCSubtargetInfo(), TM);
  McSubtargetInfo.reset();

  llvm::CodeGenTargetMachineImpl &TgtMachine =
      static_cast<llvm::CodeGenTargetMachineImpl &>(TM);
  (TgtMachine.*
   LLVMPrivateFnAccessor<TargetMachineClearSubtargetMapAccessor>::Ptr)();
}

void llvm::mclinker::resetSubtargetInfo(llvm::TargetMachine &Dst,
                                        llvm::MachineModuleInfo &MMI) {

  llvm::DenseMap<const llvm::Function *, std::unique_ptr<llvm::MachineFunction>>
      &MFs = getMachineFunctionsFromMachineModuleInfo(MMI);

  for (auto &[Fn, MF] : MFs) {
    const llvm::TargetSubtargetInfo *NewSTI = Dst.getSubtargetImpl(*Fn);
    const llvm::TargetSubtargetInfo *&STI = std::invoke(getSTIAccessor(), MF);
    STI = NewSTI;
  }
}
