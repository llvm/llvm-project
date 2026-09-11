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
#include "llvm/Config/llvm-config.h" // for LLVM_ENABLE_THREADS
#include "llvm/Support/Endian.h"

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif
#ifndef _WIN32
#include <sys/socket.h>
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

shared::WrapperFunctionBuffer encodeHangupPayload(Error Err) {
  using SPSSerialize = shared::SPSArgList<shared::SPSError>;
  auto SE = shared::detail::toSPSSerializable(std::move(Err));
  auto Payload =
      shared::WrapperFunctionBuffer::allocate(SPSSerialize::size(SE));
  shared::SPSOutputBuffer OB(Payload.data(), Payload.size());
  bool Success = SPSSerialize::serialize(OB, SE);
  (void)Success;
  assert(Success && "Hangup payload serialization should not fail");
  return Payload;
}

Error decodeHangupPayload(shared::WrapperFunctionBuffer Payload) {
  assert(!Payload.getOutOfBandError() &&
         "Hangup payload should not be an out-of-band error buffer");

  shared::detail::SPSSerializableError Info;
  shared::SPSInputBuffer IB(Payload.data(), Payload.size());
  if (!shared::SPSArgList<shared::SPSError>::deserialize(IB, Info))
    return make_error<StringError>("Could not deserialize hangup info",
                                   inconvertibleErrorCode());
  return shared::detail::fromSPSSerializable(std::move(Info));
}

SimpleRemoteEPCTransportClient::~SimpleRemoteEPCTransportClient() = default;
SimpleRemoteEPCTransport::~SimpleRemoteEPCTransport() = default;

Expected<std::unique_ptr<FDSimpleRemoteEPCTransport>>
FDSimpleRemoteEPCTransport::Create(SimpleRemoteEPCTransportClient &C, int InFD,
                                   int OutFD) {
#if LLVM_ENABLE_THREADS
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
  return std::move(FDT);
#else
  return make_error<StringError>("FD-based SimpleRemoteEPC transport requires "
                                 "thread support, but llvm was built with "
                                 "LLVM_ENABLE_THREADS=Off",
                                 inconvertibleErrorCode());
#endif
}

FDSimpleRemoteEPCTransport::~FDSimpleRemoteEPCTransport() {
#if LLVM_ENABLE_THREADS
  // Ensure the listen thread is finished and FDs are closed before destruction.
  disconnect();
  if (ListenerThread.joinable())
    ListenerThread.join();
#endif
}

Error FDSimpleRemoteEPCTransport::start() {
#if LLVM_ENABLE_THREADS
  ListenerThread = std::thread([this]() { listenLoop(); });
  return Error::success();
#endif
  llvm_unreachable("Should not be called with LLVM_ENABLE_THREADS=Off");
}

Error FDSimpleRemoteEPCTransport::sendMessage(SimpleRemoteEPCOpcode OpC,
                                              uint64_t SeqNo,
                                              ExecutorAddr TagAddr,
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
  if (Disconnected)
    return make_error<StringError>("FD-transport disconnected",
                                   inconvertibleErrorCode());
  if (int ErrNo = writeBytes(HeaderBuffer, FDMsgHeader::Size))
    return errorCodeToError(std::error_code(ErrNo, std::generic_category()));
  if (int ErrNo = writeBytes(ArgBytes.data(), ArgBytes.size()))
    return errorCodeToError(std::error_code(ErrNo, std::generic_category()));
  return Error::success();
}

void FDSimpleRemoteEPCTransport::disconnect() {
  bool CloseFDs = false;
  bool CloseOutFD = false;
  {
    std::lock_guard<std::mutex> Lock(M);
    if (!Disconnected) {
      Disconnected = true;
      CloseFDs = true;
      CloseOutFD = InFD != OutFD;
    }
  }

#ifndef _WIN32
  // Wake any blocking read so the listen thread can exit before we close.
  // If the FD is not a socket, shutdown will just complain through errno
  // (instead of crashing).
  // FIXME: what about Windows?
  if (CloseFDs) {
    ::shutdown(InFD, CloseOutFD ? SHUT_RD : SHUT_RDWR);
    if (CloseOutFD)
      ::shutdown(OutFD, SHUT_WR);
  }
#endif

#if LLVM_ENABLE_THREADS
  // Join the listener before closing FDs when disconnect is called from
  // another thread. Closing while listenLoop is still in read/write races
  // with close (TSan). The listener itself calls disconnect at exit and must
  // not join itself.
  if (ListenerThread.joinable() &&
      ListenerThread.get_id() != std::this_thread::get_id())
    ListenerThread.join();
#endif

  // Close under the send mutex so we cannot close while sendMessage is mid
  // write, and set FDs to -1 so a second disconnect is a no-op.
  if (CloseFDs) {
    std::lock_guard<std::mutex> Lock(M);
    if (InFD != -1) {
      while (close(InFD) == -1) {
        if (errno == EBADF)
          break;
      }
      InFD = -1;
    }
    if (CloseOutFD && OutFD != -1) {
      while (close(OutFD) == -1) {
        if (errno == EBADF)
          break;
      }
      OutFD = -1;
    }
  }
}

static Error makeUnexpectedEOFError() {
  return make_error<StringError>("Unexpected end-of-file",
                                 inconvertibleErrorCode());
}

Error FDSimpleRemoteEPCTransport::readBytes(char *Dst, size_t Size,
                                            bool *IsEOF) {
  assert((Size == 0 || Dst) && "Attempt to read into null.");
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
        if (Disconnected && IsEOF) { // disconnect called,  pretend this is EOF.
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
  assert((Size == 0 || Src) && "Attempt to append from null.");
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
      bool IsEOF = false;
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
    ExecutorAddr TagAddr;

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
                       make_error<StringError>("Message size too small",
                                               inconvertibleErrorCode()));
      break;
    }

    // Read the argument bytes.
    auto ArgBytes =
        shared::WrapperFunctionBuffer::allocate(MsgSize - FDMsgHeader::Size);
    if (auto Err2 = readBytes(ArgBytes.data(), ArgBytes.size())) {
      Err = joinErrors(std::move(Err), std::move(Err2));
      break;
    }

    if (auto Action =
            C.handleMessage(OpC, SeqNo, TagAddr, std::move(ArgBytes))) {
      if (*Action == SimpleRemoteEPCTransportClient::EndSession)
        break;
    } else {
      Err = joinErrors(std::move(Err), Action.takeError());
      break;
    }
  } while (true);

  // Attempt to close FDs, set Disconnected to true so that subsequent
  // sendMessage calls fail.
  disconnect();

  // Call up to the client to handle the disconnection.
  C.handleDisconnect(std::move(Err));
}

} // end namespace orc
} // end namespace llvm
