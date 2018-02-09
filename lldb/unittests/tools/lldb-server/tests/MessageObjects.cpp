//===-- MessageObjects.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MessageObjects.h"
#include "lldb/Interpreter/Args.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using namespace llvm;
using namespace llvm::support;
namespace llgs_tests {

Expected<ProcessInfo> ProcessInfo::create(StringRef response) {
  ProcessInfo process_info;
  auto elements_or_error = SplitUniquePairList("ProcessInfo", response);
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

Expected<uint64_t>
ThreadInfo::ReadRegisterAsUint64(unsigned int register_id) const {
  uint64_t value;
  std::string value_str(m_registers.lookup(register_id));
  if (!llvm::to_integer(value_str, value, 16))
    return make_parsing_error("ThreadInfo value for register {0}: {1}",
                              register_id, value_str);

  sys::swapByteOrder(value);
  return value;
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

Expected<RegisterInfo> RegisterInfoParser::create(StringRef Response) {
  auto ElementsOr = SplitUniquePairList("RegisterInfoParser", Response);
  if (!ElementsOr)
    return ElementsOr.takeError();
  auto &Elements = *ElementsOr;

  RegisterInfo Info = {
      nullptr,       // Name
      nullptr,       // Alt name
      0,             // byte size
      0,             // offset
      eEncodingUint, // encoding
      eFormatHex,    // format
      {
          LLDB_INVALID_REGNUM, // eh_frame reg num
          LLDB_INVALID_REGNUM, // DWARF reg num
          LLDB_INVALID_REGNUM, // generic reg num
          LLDB_INVALID_REGNUM, // process plugin reg num
          LLDB_INVALID_REGNUM  // native register number
      },
      NULL,
      NULL,
      NULL, // Dwarf expression opcode bytes pointer
      0     // Dwarf expression opcode bytes length
  };
  Info.name = ConstString(Elements["name"]).GetCString();
  if (!Info.name)
    return make_parsing_error("qRegisterInfo: name");

  Info.alt_name = ConstString(Elements["alt-name"]).GetCString();

  if (!to_integer(Elements["bitsize"], Info.byte_size, 10))
    return make_parsing_error("qRegisterInfo: bit-size");
  Info.byte_size /= CHAR_BIT;

  if (!to_integer(Elements["offset"], Info.byte_offset, 10))
    return make_parsing_error("qRegisterInfo: offset");

  Info.encoding = Args::StringToEncoding(Elements["encoding"]);
  if (Info.encoding == eEncodingInvalid)
    return make_parsing_error("qRegisterInfo: encoding");

  Info.format = StringSwitch<Format>(Elements["format"])
                    .Case("binary", eFormatBinary)
                    .Case("decimal", eFormatDecimal)
                    .Case("hex", eFormatHex)
                    .Case("float", eFormatFloat)
                    .Case("vector-sint8", eFormatVectorOfSInt8)
                    .Case("vector-uint8", eFormatVectorOfUInt8)
                    .Case("vector-sint16", eFormatVectorOfSInt16)
                    .Case("vector-uint16", eFormatVectorOfUInt16)
                    .Case("vector-sint32", eFormatVectorOfSInt32)
                    .Case("vector-uint32", eFormatVectorOfUInt32)
                    .Case("vector-float32", eFormatVectorOfFloat32)
                    .Case("vector-uint64", eFormatVectorOfUInt64)
                    .Case("vector-uint128", eFormatVectorOfUInt128)
                    .Default(eFormatInvalid);
  if (Info.format == eFormatInvalid)
    return make_parsing_error("qRegisterInfo: format");

  Info.kinds[eRegisterKindGeneric] =
      Args::StringToGenericRegister(Elements["generic"]);

  return std::move(Info);
}

//====== StopReply =============================================================
Expected<std::unique_ptr<StopReply>>
StopReply::create(StringRef Response, llvm::support::endianness Endian) {
  if (Response.size() < 3)
    return make_parsing_error("StopReply: Invalid packet");
  if (Response.consume_front("T"))
    return StopReplyStop::create(Response, Endian);
  if (Response.consume_front("W"))
    return StopReplyExit::create(Response);
  return make_parsing_error("StopReply: Invalid packet");
}

Expected<std::unique_ptr<StopReplyStop>>
StopReplyStop::create(StringRef Response, llvm::support::endianness Endian) {
  unsigned int Signal;
  StringRef SignalStr = Response.take_front(2);
  Response = Response.drop_front(2);
  if (!to_integer(SignalStr, Signal, 16))
    return make_parsing_error("StopReply: stop signal");

  auto Elements = SplitPairList(Response);
  for (StringRef Field :
       {"name", "reason", "thread", "threads", "thread-pcs"}) {
    // This will insert an empty field if there is none. In the future, we
    // should probably differentiate between these fields not being present and
    // them being empty, but right now no tests depends on this.
    if (Elements.insert({Field, {""}}).first->second.size() != 1)
      return make_parsing_error(
          "StopReply: got multiple responses for the {0} field", Field);
  }
  StringRef Name = Elements["name"][0];
  StringRef Reason = Elements["reason"][0];

  lldb::tid_t Thread;
  if (!to_integer(Elements["thread"][0], Thread, 16))
    return make_parsing_error("StopReply: thread");

  SmallVector<StringRef, 20> Threads;
  SmallVector<StringRef, 20> Pcs;
  Elements["threads"][0].split(Threads, ',');
  Elements["thread-pcs"][0].split(Pcs, ',');
  if (Threads.size() != Pcs.size())
    return make_parsing_error("StopReply: thread/PC count mismatch");

  U64Map ThreadPcs;
  for (auto ThreadPc : zip(Threads, Pcs)) {
    lldb::tid_t Id;
    uint64_t Pc;
    if (!to_integer(std::get<0>(ThreadPc), Id, 16))
      return make_parsing_error("StopReply: Thread id '{0}'",
                                std::get<0>(ThreadPc));
    if (!to_integer(std::get<1>(ThreadPc), Pc, 16))
      return make_parsing_error("StopReply Thread Pc '{0}'",
                                std::get<1>(ThreadPc));

    ThreadPcs[Id] = Pc;
  }

  RegisterMap Registers;
  for (const auto &E : Elements) {
    StringRef Key = E.getKey();
    const auto &Val = E.getValue();
    if (Key.size() != 2)
      continue;

    unsigned int Reg;
    if (!to_integer(Key, Reg, 16))
      continue;

    if (Val.size() != 1)
      return make_parsing_error(
          "StopReply: multiple entries for register field [{0:x}]", Reg);

    Registers[Reg] = Val[0].str();
  }

  return llvm::make_unique<StopReplyStop>(Signal, Thread, Name, ThreadPcs,
                                          Registers, Reason);
}

Expected<std::unique_ptr<StopReplyExit>>
StopReplyExit::create(StringRef Response) {
  uint8_t Status;
  if (!to_integer(Response, Status, 16))
    return make_parsing_error("StopReply: exit status");
  return llvm::make_unique<StopReplyExit>(Status);
}

//====== Globals ===============================================================
Expected<StringMap<StringRef>> SplitUniquePairList(StringRef caller,
                                                   StringRef str) {
  SmallVector<StringRef, 20> elements;
  str.split(elements, ';');

  StringMap<StringRef> pairs;
  for (StringRef s : elements) {
    std::pair<StringRef, StringRef> pair = s.split(':');
    if (pairs.count(pair.first))
      return make_parsing_error("{0}: Duplicate Key: {1}", caller, pair.first);

    pairs.insert(pair);
  }

  return pairs;
}

StringMap<SmallVector<StringRef, 2>> SplitPairList(StringRef str) {
  SmallVector<StringRef, 20> elements;
  str.split(elements, ';');

  StringMap<SmallVector<StringRef, 2>> pairs;
  for (StringRef s : elements) {
    std::pair<StringRef, StringRef> pair = s.split(':');
    pairs[pair.first].push_back(pair.second);
  }

  return pairs;
}
} // namespace llgs_tests
