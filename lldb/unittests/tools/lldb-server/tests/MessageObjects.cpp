//===-- MessageObjects.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MessageObjects.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/StringExtractor.h"
#include "llvm/ADT/StringExtras.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace lldb;
using namespace llvm;
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

support::endianness ProcessInfo::GetEndian() const { return m_endian; }

//====== ThreadInfo ============================================================
ThreadInfo::ThreadInfo(StringRef name, StringRef reason, RegisterMap registers,
                       unsigned int)
    : m_name(name.str()), m_reason(reason.str()),
      m_registers(std::move(registers)) {}

const RegisterValue *ThreadInfo::ReadRegister(unsigned int Id) const {
  auto Iter = m_registers.find(Id);
  return Iter == m_registers.end() ? nullptr : &Iter->getSecond();
}

//====== JThreadsInfo ==========================================================

Expected<RegisterMap>
JThreadsInfo::parseRegisters(const StructuredData::Dictionary &Dict,
                             ArrayRef<RegisterInfo> RegInfos) {
  RegisterMap Result;

  auto KeysObj = Dict.GetKeys();
  auto Keys = KeysObj->GetAsArray();
  for (size_t i = 0; i < Keys->GetSize(); i++) {
    StringRef KeyStr, ValueStr;
    Keys->GetItemAtIndexAsString(i, KeyStr);
    Dict.GetValueForKeyAsString(KeyStr, ValueStr);
    unsigned int Register;
    if (!llvm::to_integer(KeyStr, Register, 10))
      return make_parsing_error("JThreadsInfo: register key[{0}]", i);

    auto RegValOr =
        parseRegisterValue(RegInfos[Register], ValueStr, support::big);
    if (!RegValOr)
      return RegValOr.takeError();
    Result[Register] = std::move(*RegValOr);
  }
  return std::move(Result);
}

Expected<JThreadsInfo> JThreadsInfo::create(StringRef Response,
                                            ArrayRef<RegisterInfo> RegInfos) {
  JThreadsInfo jthreads_info;

  StructuredData::ObjectSP json =
      StructuredData::ParseJSON(std::string(Response));
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

    auto RegsOr = parseRegisters(*register_dict, RegInfos);
    if (!RegsOr)
      return RegsOr.takeError();
    jthreads_info.m_thread_infos[tid] =
        ThreadInfo(name, reason, std::move(*RegsOr), signal);
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
      nullptr,
      nullptr,
  };
  Info.name = ConstString(Elements["name"]).GetCString();
  if (!Info.name)
    return make_parsing_error("qRegisterInfo: name");

  Info.alt_name = ConstString(Elements["alt-name"]).GetCString();

  if (!to_integer(Elements["bitsize"], Info.byte_size, 10))
    return make_parsing_error("qRegisterInfo: bit-size");
  Info.byte_size /= CHAR_BIT;

  if (!to_integer(Elements["offset"], Info.byte_offset, 10))
    Info.byte_offset = LLDB_INVALID_INDEX32;

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

Expected<RegisterValue> parseRegisterValue(const RegisterInfo &Info,
                                           StringRef HexValue,
                                           llvm::support::endianness Endian,
                                           bool ZeroPad) {
  SmallString<128> Storage;
  if (ZeroPad && HexValue.size() < Info.byte_size * 2) {
    Storage.insert(Storage.begin(), Info.byte_size * 2 - HexValue.size(), '0');
    Storage += HexValue;
    HexValue = Storage;
  }

  SmallVector<uint8_t, 64> Bytes(HexValue.size() / 2);
  StringExtractor(HexValue).GetHexBytes(Bytes, '\xcc');
  RegisterValue Value;
  Status ST;
  Value.SetFromMemoryData(
      &Info, Bytes.data(), Bytes.size(),
      Endian == support::little ? eByteOrderLittle : eByteOrderBig, ST);
  if (ST.Fail())
    return ST.ToError();
  return Value;
}

//====== StopReply =============================================================
Expected<std::unique_ptr<StopReply>>
StopReply::create(StringRef Response, llvm::support::endianness Endian,
                  ArrayRef<RegisterInfo> RegInfos) {
  if (Response.size() < 3)
    return make_parsing_error("StopReply: Invalid packet");
  if (Response.consume_front("T"))
    return StopReplyStop::create(Response, Endian, RegInfos);
  if (Response.consume_front("W"))
    return StopReplyExit::create(Response);
  return make_parsing_error("StopReply: Invalid packet");
}

Expected<RegisterMap> StopReplyStop::parseRegisters(
    const StringMap<SmallVector<StringRef, 2>> &Elements,
    support::endianness Endian, ArrayRef<lldb_private::RegisterInfo> RegInfos) {

  RegisterMap Result;
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
          "StopReplyStop: multiple entries for register field [{0:x}]", Reg);

    auto RegValOr = parseRegisterValue(RegInfos[Reg], Val[0], Endian);
    if (!RegValOr)
      return RegValOr.takeError();
    Result[Reg] = std::move(*RegValOr);
  }
  return std::move(Result);
}

Expected<std::unique_ptr<StopReplyStop>>
StopReplyStop::create(StringRef Response, support::endianness Endian,
                      ArrayRef<RegisterInfo> RegInfos) {
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

  RegisterMap ThreadPcs;
  const RegisterInfo *PcInfo = find_if(RegInfos, [](const RegisterInfo &Info) {
    return Info.kinds[eRegisterKindGeneric] == LLDB_REGNUM_GENERIC_PC;
  });
  assert(PcInfo);

  for (auto ThreadPc : zip(Threads, Pcs)) {
    lldb::tid_t Id;
    if (!to_integer(std::get<0>(ThreadPc), Id, 16))
      return make_parsing_error("StopReply: Thread id '{0}'",
                                std::get<0>(ThreadPc));

    auto PcOr = parseRegisterValue(*PcInfo, std::get<1>(ThreadPc), Endian,
                                   /*ZeroPad*/ true);
    if (!PcOr)
      return PcOr.takeError();
    ThreadPcs[Id] = std::move(*PcOr);
  }

  auto RegistersOr = parseRegisters(Elements, Endian, RegInfos);
  if (!RegistersOr)
    return RegistersOr.takeError();

  return std::make_unique<StopReplyStop>(Signal, Thread, Name,
                                          std::move(ThreadPcs),
                                          std::move(*RegistersOr), Reason);
}

Expected<std::unique_ptr<StopReplyExit>>
StopReplyExit::create(StringRef Response) {
  uint8_t Status;
  if (!to_integer(Response, Status, 16))
    return make_parsing_error("StopReply: exit status");
  return std::make_unique<StopReplyExit>(Status);
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

std::ostream &lldb_private::operator<<(std::ostream &OS,
                                       const RegisterValue &RegVal) {
  ArrayRef<uint8_t> Bytes(static_cast<const uint8_t *>(RegVal.GetBytes()),
                          RegVal.GetByteSize());
  return OS << formatv("RegisterValue[{0}]: {1:@[x-2]}", RegVal.GetByteSize(),
                       make_range(Bytes.begin(), Bytes.end()))
                   .str();
}
