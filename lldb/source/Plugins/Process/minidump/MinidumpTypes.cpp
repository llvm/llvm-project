//===-- MinidumpTypes.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Project includes
#include "MinidumpTypes.h"
#include "MinidumpParser.h"

// Other libraries and framework includes
// C includes
// C++ includes

using namespace lldb_private;
using namespace minidump;

const MinidumpHeader *MinidumpHeader::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpHeader *header = nullptr;
  Error error = consumeObject(data, header);

  const MinidumpHeaderConstants signature =
      static_cast<const MinidumpHeaderConstants>(
          static_cast<const uint32_t>(header->signature));
  const MinidumpHeaderConstants version =
      static_cast<const MinidumpHeaderConstants>(
          static_cast<const uint32_t>(header->version) & 0x0000ffff);
  // the high 16 bits of the version field are implementation specific

  if (error.Fail() || signature != MinidumpHeaderConstants::Signature ||
      version != MinidumpHeaderConstants::Version)
    return nullptr;

  // TODO check for max number of streams ?
  // TODO more sanity checks ?

  return header;
}

// MinidumpThread
const MinidumpThread *MinidumpThread::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpThread *thread = nullptr;
  Error error = consumeObject(data, thread);
  if (error.Fail())
    return nullptr;

  return thread;
}

llvm::Optional<std::vector<const MinidumpThread *>>
MinidumpThread::ParseThreadList(llvm::ArrayRef<uint8_t> &data) {
  std::vector<const MinidumpThread *> thread_list;

  const llvm::support::ulittle32_t *thread_count;
  Error error = consumeObject(data, thread_count);
  if (error.Fail())
    return llvm::None;

  const MinidumpThread *thread;
  for (uint32_t i = 0; i < *thread_count; ++i) {
    thread = MinidumpThread::Parse(data);
    if (thread == nullptr)
      return llvm::None;
    thread_list.push_back(thread);
  }

  return llvm::Optional<std::vector<const MinidumpThread *>>(thread_list);
}

// MinidumpSystemInfo
const MinidumpSystemInfo *
MinidumpSystemInfo::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpSystemInfo *system_info;
  Error error = consumeObject(data, system_info);
  if (error.Fail())
    return nullptr;

  return system_info;
}

// MinidumpMiscInfo
const MinidumpMiscInfo *MinidumpMiscInfo::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpMiscInfo *misc_info;
  Error error = consumeObject(data, misc_info);
  if (error.Fail())
    return nullptr;

  return misc_info;
}
