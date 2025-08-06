//===- llvm/CodeGen/TailDuplication.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_TAILDUPLICATIONPASS_H
#define LLVM_CODEGEN_TAILDUPLICATIONPASS_H

#include "llvm/CodeGen/MBFIWrapper.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

template <typename DerivedT, bool PreRegAlloc>
class TailDuplicatePassBase : public PassInfoMixin<DerivedT> {
private:
  std::unique_ptr<MBFIWrapper> MBFIW;

public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class EarlyTailDuplicatePass
    : public TailDuplicatePassBase<EarlyTailDuplicatePass, true> {
public:
  MachineFunctionProperties getClearedProperties() const {
    return MachineFunctionProperties().setNoPHIs();
  }
};

class TailDuplicatePass
    : public TailDuplicatePassBase<TailDuplicatePass, false> {};

} // namespace llvm

extern template class llvm::TailDuplicatePassBase<llvm::EarlyTailDuplicatePass,
                                                  true>;
extern template class llvm::TailDuplicatePassBase<llvm::TailDuplicatePass,
                                                  false>;

#endif // LLVM_CODEGEN_TAILDUPLICATIONPASS_H
