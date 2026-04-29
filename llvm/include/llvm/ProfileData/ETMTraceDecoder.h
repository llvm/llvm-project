//===-- ETMTraceDecoder.h - ETM Trace Decoder -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_ETMTRACEDECODER_H
#define LLVM_PROFILEDATA_ETMTRACEDECODER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>

namespace llvm {

class Triple;
namespace object {
class Binary;
}

class ETMDecoder {
public:
  virtual ~ETMDecoder() = default;

  class Callback {
  public:
    virtual ~Callback() = default;
    virtual void processInstructionRange(uint64_t Start, uint64_t End) = 0;
  };

  static Expected<std::unique_ptr<ETMDecoder>>
  create(const object::Binary &Binary, const Triple &TargetTriple,
         uint8_t TraceID = 0x10);

  virtual Error processTrace(ArrayRef<uint8_t> TraceData,
                             Callback &TraceCallback) = 0;
};

} // namespace llvm

#endif // LLVM_PROFILEDATA_ETMTRACEDECODER_H
