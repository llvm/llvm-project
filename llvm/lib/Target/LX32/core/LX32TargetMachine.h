//===-- LX32TargetMachine.h - LX32 TargetMachine Declaration -------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file declares LX32TargetMachine, the top-level entry point for the
// LX32 LLVM backend.
//
// It is organized into the following sections:
//
//   Section 0 — Role in the backend pipeline
//   Section 1 — Includes and dependencies
//   Section 2 — Class declaration
//   Section 3 — Public interface
//
//===----------------------------------------------------------------------===//
//
// Section 0 — Role in the backend pipeline
//
// LX32TargetMachine is the first object the LLVM driver instantiates when it
// targets LX32 (via -march=lx32 or a lx32-unknown-elf triple).  It is
// responsible for three things:
//
//   1. Global target configuration.
//      Stores the DataLayout string, relocation model, code model, and
//      optimization level.  These values are immutable after construction and
//      are shared by all functions compiled in the same module.
//
//      DataLayout "e-m:e-p:32:32-i64:64-n32-S32" encodes:
//        e      — little-endian byte order
//        m:e    — ELF symbol mangling
//        p:32:32 — 32-bit pointers, 32-bit ABI alignment
//        i64:64  — 64-bit integers aligned to 8 bytes
//        n32     — the native integer width is 32 bits
//        S32     — minimum stack alignment is 4 bytes (call sites use 16)
//
//   2. Subtarget management.
//      Creates and caches LX32Subtarget instances.  A subtarget encapsulates
//      the combination of CPU and feature flags for a particular function.
//      Because LLVM allows per-function target overrides via function
//      attributes (e.g., __attribute__((target("cpu=fast-lx32")))), the
//      TargetMachine maintains a StringMap cache rather than a single global
//      subtarget.
//
//   3. Pass pipeline construction.
//      createPassConfig returns a TargetPassConfig subclass (LX32PassConfig)
//      that registers the LX32-specific code generation passes in the correct
//      order.  The most important pass is the instruction selector, added in
//      LX32PassConfig::addInstSelector (Day 10).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LX32_CORE_LX32TARGETMACHINE_H
#define LLVM_LIB_TARGET_LX32_CORE_LX32TARGETMACHINE_H

//===----------------------------------------------------------------------===//
// Section 1 — Includes and dependencies
//===----------------------------------------------------------------------===//

// LX32Subtarget must be included before CodeGenTargetMachineImpl because the
// subtarget cache uses std::unique_ptr<LX32Subtarget>, which requires the
// complete type at the point of the StringMap declaration.
#include "LX32Subtarget.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/CodeGenTargetMachineImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <optional>

namespace llvm {

//===----------------------------------------------------------------------===//
// Section 2 — Class declaration
//===----------------------------------------------------------------------===//

class LX32TargetMachine final : public CodeGenTargetMachineImpl {
  // TLOF — target lowering object file.
  //
  // Provides the MC-layer object-file policies: section naming, relocation
  // kinds, debug info format, etc.  For LX32 v1, we use the standard ELF
  // implementation (TargetLoweringObjectFileELF) because LX32 targets Linux-
  // style ELF binaries.
  //
  // Owned via unique_ptr because TargetLoweringObjectFile is polymorphic.
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

  // SubtargetMap — cache of constructed subtargets.
  //
  // Key: CPU + FeatureString (concatenated, used as a single opaque key).
  // Value: the LX32Subtarget instance for that combination.
  //
  // Most compilations use a single subtarget (the module-level -mcpu / -mattr
  // combination).  The cache exists to handle the rare case of per-function
  // target attributes without rebuilding the subtarget on every getSubtargetImpl
  // call.
  //
  // Declared mutable because getSubtargetImpl is const (required by the LLVM
  // API) but may insert into the map on the first call for a new key.
  mutable StringMap<std::unique_ptr<LX32Subtarget>> SubtargetMap;

public:
  //===--------------------------------------------------------------------===//
  // Section 3 — Public interface
  //===--------------------------------------------------------------------===//

  // Constructor — initialise the target machine with the given parameters.
  //
  //   T   — the Target object registered by LLVMInitializeLX32TargetInfo
  //   TT  — the target triple (lx32-unknown-elf, lx32-linux-elf, etc.)
  //   CPU — the CPU string from -mcpu (empty → "generic-lx32")
  //   FS  — the feature string from -mattr
  //   Options — optimisation-level flags from the driver
  //   RM  — relocation model (Static for LX32 v1; PIC is not yet supported)
  //   CM  — code model (Small; the entire image fits in a single 32-bit range)
  //   OL  — optimisation level (None / Less / Default / Aggressive)
  //   JIT — true when constructing for JIT use (not supported in LX32 v1)
  LX32TargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                    StringRef FS, const TargetOptions &Options,
                    std::optional<Reloc::Model> RM,
                    std::optional<CodeModel::Model> CM,
                    CodeGenOptLevel OL, bool JIT);

  ~LX32TargetMachine() override;

  // getSubtargetImpl — return the subtarget for the given function.
  //
  // Reads target-cpu and target-features function attributes and returns the
  // cached subtarget for that combination, constructing it on first call.
  //
  // The returned pointer is valid as long as this TargetMachine lives.
  const LX32Subtarget *getSubtargetImpl(const Function &F) const override;

  // createPassConfig — build the code generation pass pipeline.
  //
  // Returns a LX32PassConfig instance that registers the LX32-specific
  // passes in the correct order.  Called once per module by the LLVM driver.
  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;

  // getObjFileLowering — return the ELF object-file lowering instance.
  //
  // Used by AsmPrinter and the MC layer to determine section placements,
  // relocation kinds, and debug-info formats.
  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_LX32_CORE_LX32TARGETMACHINE_H
