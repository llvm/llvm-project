//===--------- EHFrameSupport.h - JITLink eh-frame utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EHFrame registration support for JITLink.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORT_H
#define LLVM_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORT_H

#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/JITLink/JITLink.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace jitlink {

/// Registers all FDEs in the given eh-frame section with the current process.
Error registerEHFrameSection(const void *EHFrameSectionAddr);

/// Deregisters all FDEs in the given eh-frame section with the current process.
Error deregisterEHFrameSection(const void *EHFrameSectionAddr);

/// Supports registration/deregistration of EH-frames in a target process.
class EHFrameRegistrar {
public:
  virtual ~EHFrameRegistrar();
  virtual Error registerEHFrames(JITTargetAddress EHFrameSectionAddr) = 0;
  virtual Error deregisterEHFrames(JITTargetAddress EHFrameSectionAddr) = 0;
};

/// Registers / Deregisters EH-frames in the current process.
class InProcessEHFrameRegistrar final : public EHFrameRegistrar {
public:
  /// Get a reference to the InProcessEHFrameRegistrar singleton.
  static InProcessEHFrameRegistrar &getInstance();

  InProcessEHFrameRegistrar(const InProcessEHFrameRegistrar &) = delete;
  InProcessEHFrameRegistrar &
  operator=(const InProcessEHFrameRegistrar &) = delete;

  InProcessEHFrameRegistrar(InProcessEHFrameRegistrar &&) = delete;
  InProcessEHFrameRegistrar &operator=(InProcessEHFrameRegistrar &&) = delete;

  Error registerEHFrames(JITTargetAddress EHFrameSectionAddr) override {
    return registerEHFrameSection(
        jitTargetAddressToPointer<void *>(EHFrameSectionAddr));
  }

  Error deregisterEHFrames(JITTargetAddress EHFrameSectionAddr) override {
    return deregisterEHFrameSection(
        jitTargetAddressToPointer<void *>(EHFrameSectionAddr));
  }

private:
  InProcessEHFrameRegistrar();
};

using StoreFrameAddressFunction = std::function<void(JITTargetAddress)>;

/// Creates a pass that records the address of the EH frame section. If no
/// eh-frame section is found, it will set EHFrameAddr to zero.
///
/// Authors of JITLinkContexts can use this function to register a post-fixup
/// pass that records the address of the eh-frame section. This address can
/// be used after finalization to register and deregister the frame.
AtomGraphPassFunction
createEHFrameRecorderPass(const Triple &TT,
                          StoreFrameAddressFunction StoreFrameAddress);

} // end namespace jitlink
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_JITLINK_EHFRAMESUPPORT_H
