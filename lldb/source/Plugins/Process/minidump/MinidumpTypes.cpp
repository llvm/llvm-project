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

// Minidump string
llvm::Optional<std::string>
lldb_private::minidump::parseMinidumpString(llvm::ArrayRef<uint8_t> &data) {
  std::string result;

  const uint32_t *source_length;
  Error error = consumeObject(data, source_length);
  if (error.Fail() || *source_length > data.size() || *source_length % 2 != 0)
    return llvm::None;

  auto source_start = reinterpret_cast<const UTF16 *>(data.data());
  // source_length is the length of the string in bytes
  // we need the length of the string in UTF-16 characters/code points (16 bits
  // per char)
  // that's why it's divided by 2
  const auto source_end = source_start + (*source_length) / 2;
  // resize to worst case length
  result.resize(UNI_MAX_UTF8_BYTES_PER_CODE_POINT * (*source_length) / 2);
  auto result_start = reinterpret_cast<UTF8 *>(&result[0]);
  const auto result_end = result_start + result.size();
  ConvertUTF16toUTF8(&source_start, source_end, &result_start, result_end,
                     strictConversion);
  const auto result_size =
      std::distance(reinterpret_cast<UTF8 *>(&result[0]), result_start);
  result.resize(result_size); // shrink to actual length

  return result;
}

// MinidumpThread
const MinidumpThread *MinidumpThread::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpThread *thread = nullptr;
  Error error = consumeObject(data, thread);
  if (error.Fail())
    return nullptr;

  return thread;
}

llvm::ArrayRef<MinidumpThread>
MinidumpThread::ParseThreadList(llvm::ArrayRef<uint8_t> &data) {
  const llvm::support::ulittle32_t *thread_count;
  Error error = consumeObject(data, thread_count);
  if (error.Fail() || *thread_count * sizeof(MinidumpThread) > data.size())
    return {};

  return llvm::ArrayRef<MinidumpThread>(
      reinterpret_cast<const MinidumpThread *>(data.data()), *thread_count);
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

llvm::Optional<lldb::pid_t> MinidumpMiscInfo::GetPid() const {
  uint32_t pid_flag =
      static_cast<const uint32_t>(MinidumpMiscInfoFlags::ProcessID);
  if (flags1 & pid_flag)
    return llvm::Optional<lldb::pid_t>(process_id);

  return llvm::None;
}

// Linux Proc Status
// it's stored as an ascii string in the file
llvm::Optional<LinuxProcStatus>
LinuxProcStatus::Parse(llvm::ArrayRef<uint8_t> &data) {
  LinuxProcStatus result;
  result.proc_status =
      llvm::StringRef(reinterpret_cast<const char *>(data.data()), data.size());
  data = data.drop_front(data.size());

  llvm::SmallVector<llvm::StringRef, 0> lines;
  result.proc_status.split(lines, '\n', 42);
  // /proc/$pid/status has 41 lines, but why not use 42?
  for (auto line : lines) {
    if (line.consume_front("Pid:")) {
      line = line.trim();
      if (!line.getAsInteger(10, result.pid))
        return result;
    }
  }

  return llvm::None;
}

lldb::pid_t LinuxProcStatus::GetPid() const { return pid; }

// Module stuff
const MinidumpModule *MinidumpModule::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpModule *module = nullptr;
  Error error = consumeObject(data, module);
  if (error.Fail())
    return nullptr;

  return module;
}

llvm::ArrayRef<MinidumpModule>
MinidumpModule::ParseModuleList(llvm::ArrayRef<uint8_t> &data) {

  const llvm::support::ulittle32_t *modules_count;
  Error error = consumeObject(data, modules_count);
  if (error.Fail() || *modules_count * sizeof(MinidumpModule) > data.size())
    return {};

  return llvm::ArrayRef<MinidumpModule>(
      reinterpret_cast<const MinidumpModule *>(data.data()), *modules_count);
}

// Exception stuff
const MinidumpExceptionStream *
MinidumpExceptionStream::Parse(llvm::ArrayRef<uint8_t> &data) {
  const MinidumpExceptionStream *exception_stream = nullptr;
  Error error = consumeObject(data, exception_stream);
  if (error.Fail())
    return nullptr;

  return exception_stream;
}
