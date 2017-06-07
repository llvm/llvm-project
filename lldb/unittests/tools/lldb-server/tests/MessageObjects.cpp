//===-- MessageObjects.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MessageObjects.h"
#include "lldb/Core/StructuredData.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;
using namespace llvm::support;
namespace llgs_tests {

Expected<ProcessInfo> ProcessInfo::Create(StringRef response) {
  ProcessInfo process_info;
  auto elements_or_error = SplitPairList("ProcessInfo", response);
  if (!elements_or_error)
    return elements_or_error.takeError();

  auto &elements = *elements_or_error;
  if (elements["pid"].getAsInteger(16, process_info.m_pid))
    return make_parsing_error("ProcessInfo: pid");
  if (elements["parent-pid"].getAsInteger(16, process_info.m_parent_pid))
    return make_parsing_error("ProcessInfo: parent-pid");
  if (elements["real-uid"].getAsInteger(16, process_info.m_real_uid))
    return make_parsing_error("ProcessInfo: real-uid");
  if (elements["real-gid"].getAsInteger(16, process_info.m_real_gid))
    return make_parsing_error("ProcessInfo: real-uid");
  if (elements["effective-uid"].getAsInteger(16, process_info.m_effective_uid))
    return make_parsing_error("ProcessInfo: effective-uid");
  if (elements["effective-gid"].getAsInteger(16, process_info.m_effective_gid))
    return make_parsing_error("ProcessInfo: effective-gid");
  if (elements["ptrsize"].getAsInteger(10, process_info.m_ptrsize))
    return make_parsing_error("ProcessInfo: ptrsize");

  process_info.m_triple = fromHex(elements["triple"]);
  StringRef endian_str = elements["endian"];
  if (endian_str == "little")
    process_info.m_endian = support::little;
  else if (endian_str == "big")
    process_info.m_endian = support::big;
  else
    return make_parsing_error("ProcessInfo: endian");

  return process_info;
}

lldb::pid_t ProcessInfo::GetPid() const { return m_pid; }

endianness ProcessInfo::GetEndian() const { return m_endian; }

//====== ThreadInfo ============================================================
ThreadInfo::ThreadInfo(StringRef name, StringRef reason,
                       const RegisterMap &registers, unsigned int signal)
    : m_name(name.str()), m_reason(reason.str()), m_registers(registers),
      m_signal(signal) {}

StringRef ThreadInfo::ReadRegister(unsigned int register_id) const {
  return m_registers.lookup(register_id);
}

bool ThreadInfo::ReadRegisterAsUint64(unsigned int register_id,
                                      uint64_t &value) const {
  StringRef value_str(m_registers.lookup(register_id));
  if (value_str.getAsInteger(16, value)) {
    GTEST_LOG_(ERROR)
        << formatv("ThreadInfo: Unable to parse register value at {0}.",
                   register_id)
               .str();
    return false;
  }

  sys::swapByteOrder(value);
  return true;
}

//====== JThreadsInfo ==========================================================
Expected<JThreadsInfo> JThreadsInfo::Create(StringRef response,
                                            endianness endian) {
  JThreadsInfo jthreads_info;

  StructuredData::ObjectSP json = StructuredData::ParseJSON(response);
  StructuredData::Array *array = json->GetAsArray();
  if (!array)
    return make_parsing_error("JThreadsInfo: JSON array");

  for (size_t i = 0; i < array->GetSize(); i++) {
    StructuredData::Dictionary *thread_info;
    array->GetItemAtIndexAsDictionary(i, thread_info);
    if (!thread_info)
      return make_parsing_error("JThreadsInfo: JSON obj at {0}", i);

    StringRef name, reason;
    thread_info->GetValueForKeyAsString("name", name);
    thread_info->GetValueForKeyAsString("reason", reason);
    uint64_t signal;
    thread_info->GetValueForKeyAsInteger("signal", signal);
    uint64_t tid;
    thread_info->GetValueForKeyAsInteger("tid", tid);

    StructuredData::Dictionary *register_dict;
    thread_info->GetValueForKeyAsDictionary("registers", register_dict);
    if (!register_dict)
      return make_parsing_error("JThreadsInfo: registers JSON obj");

    RegisterMap registers;

    auto keys_obj = register_dict->GetKeys();
    auto keys = keys_obj->GetAsArray();
    for (size_t i = 0; i < keys->GetSize(); i++) {
      StringRef key_str, value_str;
      keys->GetItemAtIndexAsString(i, key_str);
      register_dict->GetValueForKeyAsString(key_str, value_str);
      unsigned int register_id;
      if (key_str.getAsInteger(10, register_id))
        return make_parsing_error("JThreadsInfo: register key[{0}]", i);

      registers[register_id] = value_str.str();
    }

    jthreads_info.m_thread_infos[tid] =
        ThreadInfo(name, reason, registers, signal);
  }

  return jthreads_info;
}

const ThreadInfoMap &JThreadsInfo::GetThreadInfos() const {
  return m_thread_infos;
}

//====== StopReply =============================================================
const U64Map &StopReply::GetThreadPcs() const { return m_thread_pcs; }

Expected<StopReply> StopReply::Create(StringRef response,
                                      llvm::support::endianness endian) {
  StopReply stop_reply;

  auto elements_or_error = SplitPairList("StopReply", response);
  if (auto split_error = elements_or_error.takeError()) {
    return std::move(split_error);
  }

  auto elements = *elements_or_error;
  stop_reply.m_name = elements["name"];
  stop_reply.m_reason = elements["reason"];

  SmallVector<StringRef, 20> threads;
  SmallVector<StringRef, 20> pcs;
  elements["threads"].split(threads, ',');
  elements["thread-pcs"].split(pcs, ',');
  if (threads.size() != pcs.size())
    return make_parsing_error("StopReply: thread/PC count mismatch");

  for (size_t i = 0; i < threads.size(); i++) {
    lldb::tid_t thread_id;
    uint64_t pc;
    if (threads[i].getAsInteger(16, thread_id))
      return make_parsing_error("StopReply: thread ID at [{0}].", i);
    if (pcs[i].getAsInteger(16, pc))
      return make_parsing_error("StopReply: thread PC at [{0}].", i);

    stop_reply.m_thread_pcs[thread_id] = pc;
  }

  for (auto i = elements.begin(); i != elements.end(); i++) {
    StringRef key = i->getKey();
    StringRef val = i->getValue();
    if (key.size() >= 9 && key[0] == 'T' && key.substr(3, 6) == "thread") {
      if (val.getAsInteger(16, stop_reply.m_thread))
        return make_parsing_error("StopReply: thread id");
      if (key.substr(1, 2).getAsInteger(16, stop_reply.m_signal))
        return make_parsing_error("StopReply: stop signal");
    } else if (key.size() == 2) {
      unsigned int reg;
      if (!key.getAsInteger(16, reg)) {
        stop_reply.m_registers[reg] = val.str();
      }
    }
  }

  return stop_reply;
}

//====== Globals ===============================================================
Expected<StringMap<StringRef>> SplitPairList(StringRef caller, StringRef str) {
  SmallVector<StringRef, 20> elements;
  str.split(elements, ';');

  StringMap<StringRef> pairs;
  for (StringRef s : elements) {
    std::pair<StringRef, StringRef> pair = s.split(':');
    if (pairs.count(pair.first))
      return make_parsing_error("{0}: Duplicate Key: {1}", caller, pair.first);

    pairs.insert(s.split(':'));
  }

  return pairs;
}
} // namespace llgs_tests
