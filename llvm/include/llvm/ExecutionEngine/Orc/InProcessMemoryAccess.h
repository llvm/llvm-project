//===-- InProcessMemoryAccess.h - Direct, in-process mem access -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Accesses memory in the current process.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_INPROCESSMEMORYACCESS_H
#define LLVM_EXECUTIONENGINE_ORC_INPROCESSMEMORYACCESS_H

#include "llvm/ExecutionEngine/Orc/MemoryAccess.h"

namespace llvm::orc {

class LLVM_ABI InProcessMemoryAccess : public MemoryAccess {
public:
  InProcessMemoryAccess(bool IsArch64Bit) : IsArch64Bit(IsArch64Bit) {}
  void writeUInt8sAsync(ArrayRef<tpctypes::UInt8Write> Ws,
                        WriteResultFn OnWriteComplete) override;

  void writeUInt16sAsync(ArrayRef<tpctypes::UInt16Write> Ws,
                         WriteResultFn OnWriteComplete) override;

  void writeUInt32sAsync(ArrayRef<tpctypes::UInt32Write> Ws,
                         WriteResultFn OnWriteComplete) override;

  void writeUInt64sAsync(ArrayRef<tpctypes::UInt64Write> Ws,
                         WriteResultFn OnWriteComplete) override;

  void writeBuffersAsync(ArrayRef<tpctypes::BufferWrite> Ws,
                         WriteResultFn OnWriteComplete) override;

  void writePointersAsync(ArrayRef<tpctypes::PointerWrite> Ws,
                          WriteResultFn OnWriteComplete) override;

private:
  bool IsArch64Bit;
};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_INPROCESSMEMORYACCESS_H
