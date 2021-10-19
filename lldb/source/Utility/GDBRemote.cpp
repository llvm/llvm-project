//===-- GDBRemote.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/GDBRemote.h"

#include "lldb/Utility/Flags.h"
#include "lldb/Utility/Stream.h"

#include <cstdio>

using namespace lldb;
using namespace lldb_private::repro;
using namespace lldb_private;
using namespace llvm;

StreamGDBRemote::StreamGDBRemote() : StreamString() {}

StreamGDBRemote::StreamGDBRemote(uint32_t flags, uint32_t addr_size,
                                 ByteOrder byte_order)
    : StreamString(flags, addr_size, byte_order) {}

StreamGDBRemote::~StreamGDBRemote() = default;

int StreamGDBRemote::PutEscapedBytes(const void *s, size_t src_len) {
  int bytes_written = 0;
  const uint8_t *src = static_cast<const uint8_t *>(s);
  bool binary_is_set = m_flags.Test(eBinary);
  m_flags.Clear(eBinary);
  while (src_len) {
    uint8_t byte = *src;
    src++;
    src_len--;
    if (byte == 0x23 || byte == 0x24 || byte == 0x7d || byte == 0x2a) {
      bytes_written += PutChar(0x7d);
      byte ^= 0x20;
    }
    bytes_written += PutChar(byte);
  };
  if (binary_is_set)
    m_flags.Set(eBinary);
  return bytes_written;
}

llvm::StringRef GDBRemotePacket::GetTypeStr() const {
  switch (type) {
  case GDBRemotePacket::ePacketTypeSend:
    return "send";
  case GDBRemotePacket::ePacketTypeRecv:
    return "read";
  case GDBRemotePacket::ePacketTypeInvalid:
    return "invalid";
  }
  llvm_unreachable("All enum cases should be handled");
}

void GDBRemotePacket::Dump(Stream &strm) const {
  strm.Printf("tid=0x%4.4" PRIx64 " <%4u> %s packet: %s\n", tid,
              bytes_transmitted, GetTypeStr().data(), packet.data.c_str());
}

void yaml::ScalarEnumerationTraits<GDBRemotePacket::Type>::enumeration(
    IO &io, GDBRemotePacket::Type &value) {
  io.enumCase(value, "Invalid", GDBRemotePacket::ePacketTypeInvalid);
  io.enumCase(value, "Send", GDBRemotePacket::ePacketTypeSend);
  io.enumCase(value, "Recv", GDBRemotePacket::ePacketTypeRecv);
}

void yaml::ScalarTraits<GDBRemotePacket::BinaryData>::output(
    const GDBRemotePacket::BinaryData &Val, void *, raw_ostream &Out) {
  Out << toHex(Val.data);
}

StringRef yaml::ScalarTraits<GDBRemotePacket::BinaryData>::input(
    StringRef Scalar, void *, GDBRemotePacket::BinaryData &Val) {
  Val.data = fromHex(Scalar);
  return {};
}

void yaml::MappingTraits<GDBRemotePacket>::mapping(IO &io,
                                                   GDBRemotePacket &Packet) {
  io.mapRequired("packet", Packet.packet);
  io.mapRequired("type", Packet.type);
  io.mapRequired("bytes", Packet.bytes_transmitted);
  io.mapRequired("index", Packet.packet_idx);
  io.mapRequired("tid", Packet.tid);
}

StringRef
yaml::MappingTraits<GDBRemotePacket>::validate(IO &io,
                                               GDBRemotePacket &Packet) {
  return {};
}

void GDBRemoteProvider::Keep() {
  std::vector<std::string> files;
  for (auto &recorder : m_packet_recorders) {
    files.push_back(recorder->GetFilename().GetPath());
  }

  FileSpec file = GetRoot().CopyByAppendingPathComponent(Info::file);
  std::error_code ec;
  llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_TextWithCRLF);
  if (ec)
    return;
  yaml::Output yout(os);
  yout << files;
}

void GDBRemoteProvider::Discard() { m_packet_recorders.clear(); }

llvm::Expected<std::unique_ptr<PacketRecorder>>
PacketRecorder::Create(const FileSpec &filename) {
  std::error_code ec;
  auto recorder = std::make_unique<PacketRecorder>(std::move(filename), ec);
  if (ec)
    return llvm::errorCodeToError(ec);
  return std::move(recorder);
}

PacketRecorder *GDBRemoteProvider::GetNewPacketRecorder() {
  std::size_t i = m_packet_recorders.size() + 1;
  std::string filename = (llvm::Twine(Info::name) + llvm::Twine("-") +
                          llvm::Twine(i) + llvm::Twine(".yaml"))
                             .str();
  auto recorder_or_error =
      PacketRecorder::Create(GetRoot().CopyByAppendingPathComponent(filename));
  if (!recorder_or_error) {
    llvm::consumeError(recorder_or_error.takeError());
    return nullptr;
  }

  m_packet_recorders.push_back(std::move(*recorder_or_error));
  return m_packet_recorders.back().get();
}

void PacketRecorder::Record(const GDBRemotePacket &packet) {
  if (!m_record)
    return;
  yaml::Output yout(m_os);
  yout << const_cast<GDBRemotePacket &>(packet);
  m_os.flush();
}

llvm::raw_ostream *GDBRemoteProvider::GetHistoryStream() {
  FileSpec history_file = GetRoot().CopyByAppendingPathComponent(Info::file);

  std::error_code EC;
  m_stream_up = std::make_unique<raw_fd_ostream>(
      history_file.GetPath(), EC, sys::fs::OpenFlags::OF_TextWithCRLF);
  return m_stream_up.get();
}

char GDBRemoteProvider::ID = 0;
const char *GDBRemoteProvider::Info::file = "gdb-remote.yaml";
const char *GDBRemoteProvider::Info::name = "gdb-remote";
