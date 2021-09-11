//===------ SimpleRemoteEPCUtils.cpp - Utils for Simple Remote EPC --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Message definitions and other utilities for SimpleRemoteEPC and
// SimpleRemoteEPCServer.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/SimpleRemoteEPCUtils.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

namespace {

struct FDMsgHeader {
  static constexpr unsigned MsgSizeOffset = 0;
  static constexpr unsigned OpCOffset = MsgSizeOffset + sizeof(uint64_t);
  static constexpr unsigned SeqNoOffset = OpCOffset + sizeof(uint64_t);
  static constexpr unsigned TagAddrOffset = SeqNoOffset + sizeof(uint64_t);
  static constexpr unsigned Size = TagAddrOffset + sizeof(uint64_t);
};

} // namespace

namespace llvm {
namespace orc {
namespace SimpleRemoteEPCDefaultBootstrapSymbolNames {

const char *ExecutorSessionObjectName =
    "__llvm_orc_SimpleRemoteEPC_dispatch_ctx";
const char *DispatchFnName = "__llvm_orc_SimpleRemoteEPC_dispatch_fn";

} // end namespace SimpleRemoteEPCDefaultBootstrapSymbolNames

SimpleRemoteEPCTransportClient::~SimpleRemoteEPCTransportClient() {}
SimpleRemoteEPCTransport::~SimpleRemoteEPCTransport() {}

Expected<std::unique_ptr<FDSimpleRemoteEPCTransport>>
FDSimpleRemoteEPCTransport::Create(SimpleRemoteEPCTransportClient &C, int InFD,
                                   int OutFD) {
  if (InFD == -1)
    return make_error<StringError>("Invalid input file descriptor " +
                                       Twine(InFD),
                                   inconvertibleErrorCode());
  if (OutFD == -1)
    return make_error<StringError>("Invalid output file descriptor " +
                                       Twine(OutFD),
                                   inconvertibleErrorCode());
  std::unique_ptr<FDSimpleRemoteEPCTransport> FDT(
      new FDSimpleRemoteEPCTransport(C, InFD, OutFD));
  return FDT;
}

FDSimpleRemoteEPCTransport::FDSimpleRemoteEPCTransport(
    SimpleRemoteEPCTransportClient &C, int InFD, int OutFD)
    : C(C), InFD(InFD), OutFD(OutFD) {
  ListenerThread = std::thread([this]() { listenLoop(); });
}

FDSimpleRemoteEPCTransport::~FDSimpleRemoteEPCTransport() {
  ListenerThread.join();
}

Error FDSimpleRemoteEPCTransport::sendMessage(SimpleRemoteEPCOpcode OpC,
                                              uint64_t SeqNo,
                                              ExecutorAddress TagAddr,
                                              ArrayRef<char> ArgBytes) {
  char HeaderBuffer[FDMsgHeader::Size];

  *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::MsgSizeOffset)) =
      FDMsgHeader::Size + ArgBytes.size();
  *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::OpCOffset)) =
      static_cast<uint64_t>(OpC);
  *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::SeqNoOffset)) = SeqNo;
  *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::TagAddrOffset)) =
      TagAddr.getValue();

  std::lock_guard<std::mutex> Lock(M);
  if (OutFD == -1)
    return make_error<StringError>("FD-transport disconnected",
                                   inconvertibleErrorCode());
  if (int ErrNo = writeBytes(HeaderBuffer, FDMsgHeader::Size))
    return errorCodeToError(std::error_code(ErrNo, std::generic_category()));
  if (int ErrNo = writeBytes(ArgBytes.data(), ArgBytes.size()))
    return errorCodeToError(std::error_code(ErrNo, std::generic_category()));
  return Error::success();
}

void FDSimpleRemoteEPCTransport::disconnect() {
  int CloseInFD = -1, CloseOutFD = -1;
  {
    std::lock_guard<std::mutex> Lock(M);
    std::swap(InFD, CloseInFD);
    std::swap(OutFD, CloseOutFD);
  }

  // If CloseOutFD == CloseInFD then set CloseOutFD to -1 up-front so that we
  // don't double-close.
  if (CloseOutFD == CloseInFD)
    CloseOutFD = -1;

  // Close InFD.
  if (CloseInFD != -1)
    while (close(CloseInFD) == -1) {
      if (errno == EBADF)
        break;
    }

  // Close OutFD.
  if (CloseOutFD != -1) {
    while (close(CloseOutFD) == -1) {
      if (errno == EBADF)
        break;
    }
  }
}

static Error makeUnexpectedEOFError() {
  return make_error<StringError>("Unexpected end-of-file",
                                 inconvertibleErrorCode());
}

Error FDSimpleRemoteEPCTransport::readBytes(char *Dst, size_t Size,
                                            bool *IsEOF) {
  assert(Dst && "Attempt to read into null.");
  ssize_t Completed = 0;
  while (Completed < static_cast<ssize_t>(Size)) {
    ssize_t Read = ::read(InFD, Dst + Completed, Size - Completed);
    if (Read <= 0) {
      auto ErrNo = errno;
      if (Read == 0) {
        if (Completed == 0 && IsEOF) {
          *IsEOF = true;
          return Error::success();
        } else
          return makeUnexpectedEOFError();
      } else if (ErrNo == EAGAIN || ErrNo == EINTR)
        continue;
      else {
        std::lock_guard<std::mutex> Lock(M);
        if (InFD == -1 && IsEOF) { // Disconnected locally. Pretend this is EOF.
          *IsEOF = true;
          return Error::success();
        }
        return errorCodeToError(
            std::error_code(ErrNo, std::generic_category()));
      }
    }
    Completed += Read;
  }
  return Error::success();
}

int FDSimpleRemoteEPCTransport::writeBytes(const char *Src, size_t Size) {
  assert(Src && "Attempt to append from null.");
  ssize_t Completed = 0;
  while (Completed < static_cast<ssize_t>(Size)) {
    ssize_t Written = ::write(OutFD, Src + Completed, Size - Completed);
    if (Written < 0) {
      auto ErrNo = errno;
      if (ErrNo == EAGAIN || ErrNo == EINTR)
        continue;
      else
        return ErrNo;
    }
    Completed += Written;
  }
  return 0;
}

void FDSimpleRemoteEPCTransport::listenLoop() {
  Error Err = Error::success();
  do {

    char HeaderBuffer[FDMsgHeader::Size];
    // Read the header buffer.
    {
      bool IsEOF;
      if (auto Err2 = readBytes(HeaderBuffer, FDMsgHeader::Size, &IsEOF)) {
        Err = joinErrors(std::move(Err), std::move(Err2));
        break;
      }
      if (IsEOF)
        break;
    }

    // Decode header buffer.
    uint64_t MsgSize;
    SimpleRemoteEPCOpcode OpC;
    uint64_t SeqNo;
    ExecutorAddress TagAddr;

    MsgSize =
        *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::MsgSizeOffset));
    OpC = static_cast<SimpleRemoteEPCOpcode>(static_cast<uint64_t>(
        *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::OpCOffset))));
    SeqNo =
        *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::SeqNoOffset));
    TagAddr.setValue(
        *((support::ulittle64_t *)(HeaderBuffer + FDMsgHeader::TagAddrOffset)));

    if (MsgSize < FDMsgHeader::Size) {
      Err = joinErrors(std::move(Err),
                       make_error<StringError>("Mesasge size too small",
                                               inconvertibleErrorCode()));
      break;
    }

    // Read the argument bytes.
    SimpleRemoteEPCArgBytesVector ArgBytes;
    ArgBytes.resize(MsgSize - FDMsgHeader::Size);
    if (auto Err2 = readBytes(ArgBytes.data(), ArgBytes.size())) {
      Err = joinErrors(std::move(Err), std::move(Err2));
      break;
    }

    if (auto Action = C.handleMessage(OpC, SeqNo, TagAddr, ArgBytes)) {
      if (*Action == SimpleRemoteEPCTransportClient::EndSession)
        break;
    } else {
      Err = joinErrors(std::move(Err), Action.takeError());
      break;
    }
  } while (true);

  C.handleDisconnect(std::move(Err));
}

} // end namespace orc
} // end namespace llvm
