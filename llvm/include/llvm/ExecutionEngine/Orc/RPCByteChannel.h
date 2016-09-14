//===- llvm/ExecutionEngine/Orc/RPCByteChannel.h ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RPCBYTECHANNEL_H
#define LLVM_EXECUTIONENGINE_ORC_RPCBYTECHANNEL_H

#include "OrcError.h"
#include "RPCSerialization.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace llvm {
namespace orc {
namespace remote {

/// Interface for byte-streams to be used with RPC.
class RPCByteChannel {
public:
  virtual ~RPCByteChannel() {}

  /// Read Size bytes from the stream into *Dst.
  virtual Error readBytes(char *Dst, unsigned Size) = 0;

  /// Read size bytes from *Src and append them to the stream.
  virtual Error appendBytes(const char *Src, unsigned Size) = 0;

  /// Flush the stream if possible.
  virtual Error send() = 0;

  /// Get the lock for stream reading.
  std::mutex &getReadLock() { return readLock; }

  /// Get the lock for stream writing.
  std::mutex &getWriteLock() { return writeLock; }

private:
  std::mutex readLock, writeLock;
};

/// Notify the channel that we're starting a message send.
/// Locks the channel for writing.
inline Error startSendMessage(RPCByteChannel &C) {
  C.getWriteLock().lock();
  return Error::success();
}

/// Notify the channel that we're ending a message send.
/// Unlocks the channel for writing.
inline Error endSendMessage(RPCByteChannel &C) {
  C.getWriteLock().unlock();
  return Error::success();
}

/// Notify the channel that we're starting a message receive.
/// Locks the channel for reading.
inline Error startReceiveMessage(RPCByteChannel &C) {
  C.getReadLock().lock();
  return Error::success();
}

/// Notify the channel that we're ending a message receive.
/// Unlocks the channel for reading.
inline Error endReceiveMessage(RPCByteChannel &C) {
  C.getReadLock().unlock();
  return Error::success();
}

template <typename ChannelT, typename T,
          typename =
            typename std::enable_if<
                       std::is_base_of<RPCByteChannel, ChannelT>::value>::
                         type>
class RPCByteChannelPrimitiveSerialization {
public:
  static Error serialize(ChannelT &C, T V) {
    support::endian::byte_swap<T, support::big>(V);
    return C.appendBytes(reinterpret_cast<const char *>(&V), sizeof(T));
  };

  static Error deserialize(ChannelT &C, T &V) {
    if (auto Err = C.readBytes(reinterpret_cast<char *>(&V), sizeof(T)))
      return Err;
    support::endian::byte_swap<T, support::big>(V);
    return Error::success();
  };
};

template <typename ChannelT>
class SerializationTraits<ChannelT, uint64_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, uint64_t> {
public:
  static const char* getName() { return "uint64_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, int64_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, int64_t> {
public:
  static const char* getName() { return "int64_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, uint32_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, uint32_t> {
public:
  static const char* getName() { return "uint32_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, int32_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, int32_t> {
public:
  static const char* getName() { return "int32_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, uint16_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, uint16_t> {
public:
  static const char* getName() { return "uint16_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, int16_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, int16_t> {
public:
  static const char* getName() { return "int16_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, uint8_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, uint8_t> {
public:
  static const char* getName() { return "uint8_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, int8_t>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, int8_t> {
public:
  static const char* getName() { return "int8_t"; }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, char>
  : public RPCByteChannelPrimitiveSerialization<ChannelT, uint8_t> {
public:
  static const char* getName() { return "char"; }

  static Error serialize(RPCByteChannel &C, char V) {
    return serializeSeq(C, static_cast<uint8_t>(V));
  };

  static Error deserialize(RPCByteChannel &C, char &V) {
    uint8_t VV;
    if (auto Err = deserializeSeq(C, VV))
      return Err;
    V = static_cast<char>(V);
    return Error::success();
  };
};

template <typename ChannelT>
class SerializationTraits<ChannelT, bool,
                          typename std::enable_if<
                            std::is_base_of<RPCByteChannel, ChannelT>::value>::
                              type> {
public:
  static const char* getName() { return "bool"; }

  static Error serialize(ChannelT &C, bool V) {
    return C.appendBytes(reinterpret_cast<const char *>(&V), 1);
  }

  static Error deserialize(ChannelT &C, bool &V) {
    return C.readBytes(reinterpret_cast<char *>(&V), 1);
  }
};

template <typename ChannelT>
class SerializationTraits<ChannelT, std::string,
                          typename std::enable_if<
                            std::is_base_of<RPCByteChannel, ChannelT>::value>::
                              type> {
public:
  static const char* getName() { return "std::string"; }

  static Error serialize(RPCByteChannel &C, StringRef S) {
    if (auto Err = SerializationTraits<RPCByteChannel, uint64_t>::
                     serialize(C, static_cast<uint64_t>(S.size())))
      return Err;
    return C.appendBytes((const char *)S.bytes_begin(), S.size());
  }

  /// RPC channel serialization for std::strings.
  static Error serialize(RPCByteChannel &C, const std::string &S) {
    return serialize(C, StringRef(S));
  }

  /// RPC channel deserialization for std::strings.
  static Error deserialize(RPCByteChannel &C, std::string &S) {
    uint64_t Count = 0;
    if (auto Err = SerializationTraits<RPCByteChannel, uint64_t>::
                     deserialize(C, Count))
      return Err;
    S.resize(Count);
    return C.readBytes(&S[0], Count);
  }
};

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RPCBYTECHANNEL_H
