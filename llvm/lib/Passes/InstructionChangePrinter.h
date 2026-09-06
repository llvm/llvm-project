//===- InstructionChangePrinter.h - Print sparse IR changes -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_PASSES_INSTRUCTIONCHANGEPRINTER_H
#define LLVM_LIB_PASSES_INSTRUCTIONCHANGEPRINTER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include <cstddef>
#include <cstdint>
#include <string>

namespace llvm {
namespace detail {

struct FunctionChangeRecord {
  uint64_t ID;
  std::string Name;

  bool operator==(const FunctionChangeRecord &Other) const {
    return ID == Other.ID && Name == Other.Name;
  }
};

struct InstructionChangeRecord {
  uint64_t ID;
  uint64_t BlockID;
  unsigned FunctionIndex;
  unsigned Index;
  size_t TextOffset;
  size_t TextLength;

  bool operator==(const InstructionChangeRecord &Other) const {
    return ID == Other.ID && BlockID == Other.BlockID &&
           FunctionIndex == Other.FunctionIndex && Index == Other.Index &&
           TextLength == Other.TextLength;
  }
};

struct BasicBlockChangeRecord {
  uint64_t ID;
  unsigned FunctionIndex;
  unsigned Index;

  bool operator==(const BasicBlockChangeRecord &Other) const {
    return ID == Other.ID && FunctionIndex == Other.FunctionIndex &&
           Index == Other.Index;
  }
};

struct InstructionChangeSnapshot {
  SmallVector<FunctionChangeRecord> Functions;
  SmallVector<BasicBlockChangeRecord> Blocks;
  SmallVector<InstructionChangeRecord> Instructions;
  std::string Text;

  bool operator==(const InstructionChangeSnapshot &Other) const {
    return Functions == Other.Functions && Blocks == Other.Blocks &&
           Instructions == Other.Instructions && Text == Other.Text;
  }

  StringRef getText(const InstructionChangeRecord &Instruction) const {
    return StringRef(Text).substr(Instruction.TextOffset,
                                  Instruction.TextLength);
  }
};

} // namespace detail

class IRChangedPrinter::InstructionChangeReporter
    : public ChangeReporter<detail::InstructionChangeSnapshot> {
  using Snapshot = detail::InstructionChangeSnapshot;

public:
  explicit InstructionChangeReporter(bool Verbose);

  void registerCallbacks(PassInstrumentationCallbacks &PIC);

private:
  struct ValueIDConfig : ValueMapConfig<const Value *> {
    enum { FollowRAUW = false };
  };

  uint64_t getValueID(const Value &V);
  uint64_t getTypeID(const Type &Ty);
  void handleInitialIR(IRUnitRef IR) override;
  void generateIRRepresentation(IRUnitRef IR, StringRef PassID,
                                Snapshot &Output) override;
  void omitAfter(StringRef PassID, std::string &Name) override;
  void handleAfter(StringRef PassID, std::string &Name, const Snapshot &Before,
                   const Snapshot &After, IRUnitRef) override;
  void handleInvalidated(StringRef PassID) override;
  void handleFiltered(StringRef PassID, std::string &Name) override;
  void handleIgnored(StringRef PassID, std::string &Name) override;

  ValueMap<const Value *, uint64_t, ValueIDConfig> ValueIDs;
  DenseMap<const Type *, uint64_t> TypeIDs;
  uint64_t NextValueID = 0;
  uint64_t NextTypeID = 0;
  raw_ostream &Out;
};

} // namespace llvm

#endif // LLVM_LIB_PASSES_INSTRUCTIONCHANGEPRINTER_H
