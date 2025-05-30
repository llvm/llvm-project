//===- IndexedMemProfData.h - MemProf format support ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements IndexedMemProfData, a data structure to hold MemProf
// in a space optimized format. It also provides utility methods for writing
// MemProf data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PROFILEDATA_INDEXEDMEMPROFDATA_H
#define LLVM_PROFILEDATA_INDEXEDMEMPROFDATA_H

#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"

namespace llvm {
namespace memprof {
struct IndexedMemProfData {
  // A map to hold memprof data per function. The lower 64 bits obtained from
  // the md5 hash of the function name is used to index into the map.
  llvm::MapVector<GlobalValue::GUID, IndexedMemProfRecord> Records;

  // A map to hold frame id to frame mappings. The mappings are used to
  // convert IndexedMemProfRecord to MemProfRecords with frame information
  // inline.
  llvm::MapVector<FrameId, Frame> Frames;

  // A map to hold call stack id to call stacks.
  llvm::MapVector<CallStackId, llvm::SmallVector<FrameId>> CallStacks;

  FrameId addFrame(const Frame &F) {
    const FrameId Id = hashFrame(F);
    Frames.try_emplace(Id, F);
    return Id;
  }

  CallStackId addCallStack(ArrayRef<FrameId> CS) {
    CallStackId CSId = hashCallStack(CS);
    CallStacks.try_emplace(CSId, CS);
    return CSId;
  }

  CallStackId addCallStack(SmallVector<FrameId> &&CS) {
    CallStackId CSId = hashCallStack(CS);
    CallStacks.try_emplace(CSId, std::move(CS));
    return CSId;
  }

private:
  // Return a hash value based on the contents of the frame. Here we use a
  // cryptographic hash function to minimize the chance of hash collisions.  We
  // do persist FrameIds as part of memprof formats up to Version 2, inclusive.
  // However, the deserializer never calls this function; it uses FrameIds
  // merely as keys to look up Frames proper.
  FrameId hashFrame(const Frame &F) const {
    llvm::HashBuilder<llvm::TruncatedBLAKE3<8>, llvm::endianness::little>
        HashBuilder;
    HashBuilder.add(F.Function, F.LineOffset, F.Column, F.IsInlineFrame);
    llvm::BLAKE3Result<8> Hash = HashBuilder.final();
    FrameId Id;
    std::memcpy(&Id, Hash.data(), sizeof(Hash));
    return Id;
  }

  // Compute a CallStackId for a given call stack.
  CallStackId hashCallStack(ArrayRef<FrameId> CS) const {
    llvm::HashBuilder<llvm::TruncatedBLAKE3<8>, llvm::endianness::little>
        HashBuilder;
    for (FrameId F : CS)
      HashBuilder.add(F);
    llvm::BLAKE3Result<8> Hash = HashBuilder.final();
    CallStackId CSId;
    std::memcpy(&CSId, Hash.data(), sizeof(Hash));
    return CSId;
  }
};
} // namespace memprof

// Write the MemProf data to OS.
Error writeMemProf(ProfOStream &OS, memprof::IndexedMemProfData &MemProfData,
                   memprof::IndexedVersion MemProfVersionRequested,
                   bool MemProfFullSchema);
} // namespace llvm
#endif
