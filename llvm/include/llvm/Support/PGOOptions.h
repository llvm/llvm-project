//===------ PGOOptions.h -- PGO option tunables ----------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Define option tunables for PGO.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_PGOOPTIONS_H
#define LLVM_SUPPORT_PGOOPTIONS_H

#include "llvm/Support/Error.h"

namespace llvm {

/// A struct capturing PGO tunables.
struct PGOOptions {
  enum PGOAction { NoAction, IRInstr, IRUse, SampleUse };
  enum CSPGOAction { NoCSAction, CSIRInstr, CSIRUse };
  PGOOptions(std::string ProfileFile = "", std::string CSProfileGenFile = "",
             std::string ProfileRemappingFile = "", PGOAction Action = NoAction,
             CSPGOAction CSAction = NoCSAction,
             bool DebugInfoForProfiling = false,
             bool PseudoProbeForProfiling = false)
      : ProfileFile(ProfileFile), CSProfileGenFile(CSProfileGenFile),
        ProfileRemappingFile(ProfileRemappingFile), Action(Action),
        CSAction(CSAction), DebugInfoForProfiling(DebugInfoForProfiling ||
                                                  (Action == SampleUse &&
                                                   !PseudoProbeForProfiling)),
        PseudoProbeForProfiling(PseudoProbeForProfiling) {
    // Note, we do allow ProfileFile.empty() for Action=IRUse LTO can
    // callback with IRUse action without ProfileFile.

    // If there is a CSAction, PGOAction cannot be IRInstr or SampleUse.
    assert(this->CSAction == NoCSAction ||
           (this->Action != IRInstr && this->Action != SampleUse));

    // For CSIRInstr, CSProfileGenFile also needs to be nonempty.
    assert(this->CSAction != CSIRInstr || !this->CSProfileGenFile.empty());

    // If CSAction is CSIRUse, PGOAction needs to be IRUse as they share
    // a profile.
    assert(this->CSAction != CSIRUse || this->Action == IRUse);

    // If neither Action nor CSAction, DebugInfoForProfiling or
    // PseudoProbeForProfiling needs to be true.
    assert(this->Action != NoAction || this->CSAction != NoCSAction ||
           this->DebugInfoForProfiling || this->PseudoProbeForProfiling);
  }
  std::string ProfileFile;
  std::string CSProfileGenFile;
  std::string ProfileRemappingFile;
  PGOAction Action;
  CSPGOAction CSAction;
  bool DebugInfoForProfiling;
  bool PseudoProbeForProfiling;
};
} // namespace llvm

#endif
