//===-- JITLinkDylib.h - JITLink Dylib type ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the JITLinkDylib API.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_JITLINK_JITLINKDYLIB_H
#define LLVM_EXECUTIONENGINE_JITLINK_JITLINKDYLIB_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"

#include <string>

namespace llvm::jitlink {

class JITLinkMemoryManager;

/// Represents a JITDylib as seen by JITLink.
class LLVM_ABI JITLinkDylib {
public:
  JITLinkDylib(std::string Name) : Name(std::move(Name)) {}

  ~JITLinkDylib();

  /// Get the name for this JITLinkDylib.
  const std::string &getName() const { return Name; }

  /// Register a JITLinkMemoryManager to be notified when this JITLinkDylib
  /// is destroyed.
  void notifyOnDestruction(JITLinkMemoryManager &MemMgr) {
    ToNotifyOnDestruction.push_back(&MemMgr);
  }

private:
  std::string Name;
  SmallVector<JITLinkMemoryManager *> ToNotifyOnDestruction;
};

} // namespace llvm::jitlink

#endif // LLVM_EXECUTIONENGINE_JITLINK_JITLINKDYLIB_H
