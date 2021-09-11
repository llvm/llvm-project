//===--- SimpleRemoteEPCUtils.h - Utils for Simple Remote EPC ---*- C++ -*-===//
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

#ifndef LLVM_EXECUTIONENGINE_ORC_SHARED_SIMPLEREMOTEEPCUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_SHARED_SIMPLEREMOTEEPCUTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/Orc/Shared/ExecutorAddress.h"
#include "llvm/ExecutionEngine/Orc/Shared/SimplePackedSerialization.h"
#include "llvm/Support/Error.h"

#include <mutex>
#include <string>
#include <thread>

namespace llvm {
namespace orc {

namespace SimpleRemoteEPCDefaultBootstrapSymbolNames {
extern const char *ExecutorSessionObjectName;
extern const char *DispatchFnName;
} // end namespace SimpleRemoteEPCDefaultBootstrapSymbolNames

enum class SimpleRemoteEPCOpcode : uint8_t {
  FirstOpC,
  Setup = FirstOpC,
  Hangup,
  Result,
  CallWrapper,
  LastOpC = CallWrapper
};

struct SimpleRemoteEPCExecutorInfo {
  std::string TargetTriple;
  uint64_t PageSize;
  StringMap<ExecutorAddress> BootstrapSymbols;

  Expected<ExecutorAddress> getBootstrapSymbol(StringRef Name) const {
    auto I = BootstrapSymbols.find(Name);
    if (I == BootstrapSymbols.end())
      return make_error<StringError>("Symbol \"" + Name +
                                         "\" not found in "
                                         "bootstrap symbols map",
                                     inconvertibleErrorCode());
    return I->second;
  }

  Error getBootstrapSymbols(
      ArrayRef<std::pair<ExecutorAddress &, StringRef>> Pairs) const {
    for (auto &KV : Pairs) {
      if (auto A = getBootstrapSymbol(KV.second))
        KV.first = *A;
      else
        return A.takeError();
    }
    return Error::success();
  }
};

using SimpleRemoteEPCArgBytesVector = SmallVector<char, 128>;

class SimpleRemoteEPCTransportClient {
public:
  enum HandleMessageAction { ContinueSession, EndSession };

  virtual ~SimpleRemoteEPCTransportClient();

  /// Handle receipt of a message.
  ///
  /// Returns an Error if the message cannot be handled, 'EndSession' if the
  /// client will not accept any further messages, and 'ContinueSession'
  /// otherwise.
  virtual Expected<HandleMessageAction>
  handleMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                ExecutorAddress TagAddr,
                SimpleRemoteEPCArgBytesVector ArgBytes) = 0;

  /// Handle a disconnection from the underlying transport. No further messages
  /// should be sent to handleMessage after this is called.
  /// Err may contain an Error value indicating unexpected disconnection. This
  /// allows clients to log such errors, but no attempt should be made at
  /// recovery (which should be handled inside the transport class, if it is
  /// supported at all).
  virtual void handleDisconnect(Error Err) = 0;
};

class SimpleRemoteEPCTransport {
public:
  virtual ~SimpleRemoteEPCTransport();

  /// Send a SimpleRemoteEPC message.
  ///
  /// This function may be called concurrently. Subclasses should implement
  /// locking if required for the underlying transport.
  virtual Error sendMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                            ExecutorAddress TagAddr,
                            ArrayRef<char> ArgBytes) = 0;

  /// Trigger disconnection from the transport. The implementation should
  /// respond by calling handleDisconnect on the client once disconnection
  /// is complete.
  virtual void disconnect() = 0;
};

/// Uses read/write on FileDescriptors for transport.
class FDSimpleRemoteEPCTransport : public SimpleRemoteEPCTransport {
public:
  /// Create a FDSimpleRemoteEPCTransport using the given FDs for
  /// reading (InFD) and writing (OutFD).
  static Expected<std::unique_ptr<FDSimpleRemoteEPCTransport>>
  Create(SimpleRemoteEPCTransportClient &C, int InFD, int OutFD);

  /// Create a FDSimpleRemoteEPCTransport using the given FD for both
  /// reading and writing.
  static Expected<std::unique_ptr<FDSimpleRemoteEPCTransport>>
  Create(SimpleRemoteEPCTransportClient &C, int FD) {
    return Create(C, FD, FD);
  }

  ~FDSimpleRemoteEPCTransport() override;

  Error sendMessage(SimpleRemoteEPCOpcode OpC, uint64_t SeqNo,
                    ExecutorAddress TagAddr, ArrayRef<char> ArgBytes) override;

  void disconnect() override;

private:
  FDSimpleRemoteEPCTransport(SimpleRemoteEPCTransportClient &C, int InFD,
                             int OutFD);

  Error readBytes(char *Dst, size_t Size, bool *IsEOF = nullptr);
  int writeBytes(const char *Src, size_t Size);
  void listenLoop();

  std::mutex M;
  SimpleRemoteEPCTransportClient &C;
  std::thread ListenerThread;
  int InFD, OutFD;
};

struct RemoteSymbolLookupSetElement {
  std::string Name;
  bool Required;
};

using RemoteSymbolLookupSet = std::vector<RemoteSymbolLookupSetElement>;

struct RemoteSymbolLookup {
  uint64_t H;
  RemoteSymbolLookupSet Symbols;
};

namespace shared {

using SPSRemoteSymbolLookupSetElement = SPSTuple<SPSString, bool>;

using SPSRemoteSymbolLookupSet = SPSSequence<SPSRemoteSymbolLookupSetElement>;

using SPSRemoteSymbolLookup = SPSTuple<uint64_t, SPSRemoteSymbolLookupSet>;

/// Tuple containing target triple, page size, and bootstrap symbols.
using SPSSimpleRemoteEPCExecutorInfo =
    SPSTuple<SPSString, uint64_t,
             SPSSequence<SPSTuple<SPSString, SPSExecutorAddress>>>;

template <>
class SPSSerializationTraits<SPSRemoteSymbolLookupSetElement,
                             RemoteSymbolLookupSetElement> {
public:
  static size_t size(const RemoteSymbolLookupSetElement &V) {
    return SPSArgList<SPSString, bool>::size(V.Name, V.Required);
  }

  static size_t serialize(SPSOutputBuffer &OB,
                          const RemoteSymbolLookupSetElement &V) {
    return SPSArgList<SPSString, bool>::serialize(OB, V.Name, V.Required);
  }

  static size_t deserialize(SPSInputBuffer &IB,
                            RemoteSymbolLookupSetElement &V) {
    return SPSArgList<SPSString, bool>::deserialize(IB, V.Name, V.Required);
  }
};

template <>
class SPSSerializationTraits<SPSRemoteSymbolLookup, RemoteSymbolLookup> {
public:
  static size_t size(const RemoteSymbolLookup &V) {
    return SPSArgList<uint64_t, SPSRemoteSymbolLookupSet>::size(V.H, V.Symbols);
  }

  static size_t serialize(SPSOutputBuffer &OB, const RemoteSymbolLookup &V) {
    return SPSArgList<uint64_t, SPSRemoteSymbolLookupSet>::serialize(OB, V.H,
                                                                     V.Symbols);
  }

  static size_t deserialize(SPSInputBuffer &IB, RemoteSymbolLookup &V) {
    return SPSArgList<uint64_t, SPSRemoteSymbolLookupSet>::deserialize(
        IB, V.H, V.Symbols);
  }
};

template <>
class SPSSerializationTraits<SPSSimpleRemoteEPCExecutorInfo,
                             SimpleRemoteEPCExecutorInfo> {
public:
  static size_t size(const SimpleRemoteEPCExecutorInfo &SI) {
    return SPSSimpleRemoteEPCExecutorInfo::AsArgList ::size(
        SI.TargetTriple, SI.PageSize, SI.BootstrapSymbols);
  }

  static bool serialize(SPSOutputBuffer &OB,
                        const SimpleRemoteEPCExecutorInfo &SI) {
    return SPSSimpleRemoteEPCExecutorInfo::AsArgList ::serialize(
        OB, SI.TargetTriple, SI.PageSize, SI.BootstrapSymbols);
  }

  static bool deserialize(SPSInputBuffer &IB, SimpleRemoteEPCExecutorInfo &SI) {
    return SPSSimpleRemoteEPCExecutorInfo::AsArgList ::deserialize(
        IB, SI.TargetTriple, SI.PageSize, SI.BootstrapSymbols);
  }
};

using SPSRunAsMainSignature = int64_t(SPSExecutorAddress,
                                      SPSSequence<SPSString>);

using SPSLoadDylibSignature =
    SPSExpected<SPSExecutorAddress>(SPSExecutorAddress, SPSString, uint64_t);

using SPSLookupSymbolsSignature =
    SPSExpected<SPSSequence<SPSSequence<SPSExecutorAddress>>>(
        SPSExecutorAddress, SPSSequence<SPSRemoteSymbolLookup>);

} // end namespace shared
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_SHARED_SIMPLEREMOTEEPCUTILS_H
