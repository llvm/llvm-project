//===- tools/dsymutil/SwiftModule.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/Bitstream/BitCodes.h"
#include "llvm/Bitstream/BitstreamReader.h"

namespace {
// Copied from swift/lib/Serialization/ModuleFormat.h
constexpr unsigned char SWIFTMODULE_SIGNATURE[] = {0xE2, 0x9C, 0xA8, 0x0E};
constexpr uint16_t expectedMajorVersion = 0;
constexpr unsigned MODULE_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID;
constexpr unsigned CONTROL_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID + 1;
constexpr unsigned METADATA = 1;
constexpr unsigned OPTIONS_BLOCK_ID = llvm::bitc::FIRST_APPLICATION_BLOCKID + 8;
constexpr unsigned IS_BUILT_FROM_INTERFACE = 11;

llvm::Error checkModuleSignature(llvm::BitstreamCursor &cursor,
                                 llvm::ArrayRef<unsigned char> signature) {
  for (unsigned char byte : signature) {
    if (cursor.AtEndOfStream())
      return llvm::createStringError("malformed bitstream");
    llvm::Expected<llvm::SimpleBitstreamCursor::word_t> maybeRead =
        cursor.Read(8);
    if (!maybeRead)
      return maybeRead.takeError();
    if (maybeRead.get() != byte)
      return llvm::createStringError("malformed bitstream");
  }
  return llvm::Error::success();
}

llvm::Error enterTopLevelModuleBlock(llvm::BitstreamCursor &cursor,
                                     unsigned ID) {
  llvm::Expected<llvm::BitstreamEntry> maybeNext = cursor.advance();
  if (!maybeNext)
    return maybeNext.takeError();
  llvm::BitstreamEntry next = maybeNext.get();

  if (next.Kind != llvm::BitstreamEntry::SubBlock)
    return llvm::createStringError("malformed bitstream");

  if (next.ID == llvm::bitc::BLOCKINFO_BLOCK_ID) {
    if (cursor.SkipBlock())
      return llvm::createStringError("malformed bitstream");
    return enterTopLevelModuleBlock(cursor, ID);
  }

  if (next.ID != ID)
    return llvm::createStringError("malformed bitstream");

  if (llvm::Error Err = cursor.EnterSubBlock(ID))
    return Err;

  return llvm::Error::success();
}

llvm::Expected<bool>
readOptionsBlock(llvm::BitstreamCursor &cursor,
                 llvm::SmallVectorImpl<uint64_t> &scratch) {
  bool is_built_from_interface = false;
  while (!cursor.AtEndOfStream()) {
    llvm::Expected<llvm::BitstreamEntry> maybeEntry = cursor.advance();
    if (!maybeEntry)
      return maybeEntry.takeError();

    llvm::BitstreamEntry entry = maybeEntry.get();
    if (entry.Kind == llvm::BitstreamEntry::EndBlock)
      break;

    if (entry.Kind == llvm::BitstreamEntry::Error)
      return llvm::createStringError("malformed bitstream");

    if (entry.Kind == llvm::BitstreamEntry::SubBlock) {
      if (cursor.SkipBlock())
        return llvm::createStringError("malformed bitstream");
      continue;
    }

    scratch.clear();
    llvm::StringRef blobData;
    llvm::Expected<unsigned> maybeKind =
        cursor.readRecord(entry.ID, scratch, &blobData);
    if (!maybeKind)
      return maybeKind.takeError();
    unsigned kind = maybeKind.get();
    switch (kind) {
    case IS_BUILT_FROM_INTERFACE:
      is_built_from_interface = true;
      continue;
    default:
      continue;
    }
  }
  return is_built_from_interface;
}

llvm::Expected<bool>
parseControlBlock(llvm::BitstreamCursor &cursor,
                  llvm::SmallVectorImpl<uint64_t> &scratch) {
  // The control block is malformed until we've at least read a major version
  // number.
  bool versionSeen = false;

  while (!cursor.AtEndOfStream()) {
    llvm::Expected<llvm::BitstreamEntry> maybeEntry = cursor.advance();
    if (!maybeEntry)
      return maybeEntry.takeError();

    llvm::BitstreamEntry entry = maybeEntry.get();
    if (entry.Kind == llvm::BitstreamEntry::EndBlock)
      break;

    if (entry.Kind == llvm::BitstreamEntry::Error)
      return llvm::createStringError("malformed bitstream");

    if (entry.Kind == llvm::BitstreamEntry::SubBlock) {
      if (entry.ID == OPTIONS_BLOCK_ID) {
        if (llvm::Error Err = cursor.EnterSubBlock(OPTIONS_BLOCK_ID))
          return Err;

        return readOptionsBlock(cursor, scratch);
      } else {
        // Unknown metadata sub-block, possibly for use by a future version of
        // the module format.
        if (cursor.SkipBlock())
          return llvm::createStringError("malformed bitstream");
      }
      continue;
    }

    scratch.clear();
    llvm::StringRef blobData;
    llvm::Expected<unsigned> maybeKind =
        cursor.readRecord(entry.ID, scratch, &blobData);
    if (!maybeKind)
      return maybeKind.takeError();

    unsigned kind = maybeKind.get();
    if (kind == METADATA) {
      if (versionSeen)
        return llvm::createStringError("multiple metadata blocks");

      uint16_t versionMajor = scratch[0];
      if (versionMajor != expectedMajorVersion)
        return llvm::createStringError("unsupported module version");

      versionSeen = true;
    }
  }
  return llvm::createStringError("could not find control block");
}

} // namespace

llvm::Expected<bool> IsBuiltFromSwiftInterface(llvm::StringRef data) {
  llvm::BitstreamCursor cursor(data);
  if (llvm::Error Err = checkModuleSignature(cursor, SWIFTMODULE_SIGNATURE))
    return llvm::joinErrors(
        llvm::createStringError("could not check signature"), std::move(Err));
  if (llvm::Error Err = enterTopLevelModuleBlock(cursor, MODULE_BLOCK_ID))
    return llvm::joinErrors(
        llvm::createStringError("could not enter top level block"),
        std::move(Err));

  llvm::BitstreamEntry topLevelEntry;
  llvm::SmallVector<uint64_t, 32> scratch;

  while (!cursor.AtEndOfStream()) {
    llvm::Expected<llvm::BitstreamEntry> maybeEntry =
        cursor.advance(llvm::BitstreamCursor::AF_DontPopBlockAtEnd);
    if (!maybeEntry)
      return maybeEntry.takeError();

    topLevelEntry = maybeEntry.get();
    if (topLevelEntry.Kind != llvm::BitstreamEntry::SubBlock)
      break;

    if (topLevelEntry.ID == CONTROL_BLOCK_ID) {
      if (llvm::Error Err = cursor.EnterSubBlock(CONTROL_BLOCK_ID))
        return Err;
      return parseControlBlock(cursor, scratch);
    }
  }
  return llvm::createStringError("no control block found");
}
