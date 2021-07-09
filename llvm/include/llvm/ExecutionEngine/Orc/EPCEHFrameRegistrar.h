//===-- EPCEHFrameRegistrar.h - EPC based eh-frame registration -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ExecutorProcessControl based eh-frame registration.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EPCEHFRAMEREGISTRAR_H
#define LLVM_EXECUTIONENGINE_ORC_EPCEHFRAMEREGISTRAR_H

#include "llvm/ExecutionEngine/JITLink/EHFrameSupport.h"
#include "llvm/ExecutionEngine/Orc/ExecutorProcessControl.h"

namespace llvm {
namespace orc {

/// Register/Deregisters EH frames in a remote process via a
/// ExecutorProcessControl instance.
class EPCEHFrameRegistrar : public jitlink::EHFrameRegistrar {
public:
  /// Create from a ExecutorProcessControl instance alone. This will use
  /// the EPC's lookupSymbols method to find the registration/deregistration
  /// funciton addresses by name.
  static Expected<std::unique_ptr<EPCEHFrameRegistrar>>
  Create(ExecutorProcessControl &EPC);

  /// Create a EPCEHFrameRegistrar with the given ExecutorProcessControl
  /// object and registration/deregistration function addresses.
  EPCEHFrameRegistrar(ExecutorProcessControl &EPC,
                      JITTargetAddress RegisterEHFrameWrapperFnAddr,
                      JITTargetAddress DeregisterEHFRameWrapperFnAddr)
      : EPC(EPC), RegisterEHFrameWrapperFnAddr(RegisterEHFrameWrapperFnAddr),
        DeregisterEHFrameWrapperFnAddr(DeregisterEHFRameWrapperFnAddr) {}

  Error registerEHFrames(JITTargetAddress EHFrameSectionAddr,
                         size_t EHFrameSectionSize) override;
  Error deregisterEHFrames(JITTargetAddress EHFrameSectionAddr,
                           size_t EHFrameSectionSize) override;

private:
  ExecutorProcessControl &EPC;
  JITTargetAddress RegisterEHFrameWrapperFnAddr;
  JITTargetAddress DeregisterEHFrameWrapperFnAddr;
};

} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_EPCEHFRAMEREGISTRAR_H
