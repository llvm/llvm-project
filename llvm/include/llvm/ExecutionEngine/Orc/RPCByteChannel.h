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

/// RPC channel serialization for a variadic list of arguments.
template <typename T, typename... Ts>
Error serializeSeq(RPCByteChannel &C, const T &Arg, const Ts &... Args) {
  if (auto Err = serialize(C, Arg))
    return Err;
  return serializeSeq(C, Args...);
}

/// RPC channel serialization for an (empty) variadic list of arguments.
inline Error serializeSeq(RPCByteChannel &C) { return Error::success(); }

/// RPC channel deserialization for a variadic list of arguments.
template <typename T, typename... Ts>
Error deserializeSeq(RPCByteChannel &C, T &Arg, Ts &... Args) {
  if (auto Err = deserialize(C, Arg))
    return Err;
  return deserializeSeq(C, Args...);
}

/// RPC channel serialization for an (empty) variadic list of arguments.
inline Error deserializeSeq(RPCByteChannel &C) { return Error::success(); }

/// RPC channel serialization for integer primitives.
template <typename T>
typename std::enable_if<
    std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value ||
        std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
        std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value,
    Error>::type
serialize(RPCByteChannel &C, T V) {
  support::endian::byte_swap<T, support::big>(V);
  return C.appendBytes(reinterpret_cast<const char *>(&V), sizeof(T));
}

/// RPC channel deserialization for integer primitives.
template <typename T>
typename std::enable_if<
    std::is_same<T, uint64_t>::value || std::is_same<T, int64_t>::value ||
        std::is_same<T, uint32_t>::value || std::is_same<T, int32_t>::value ||
        std::is_same<T, uint16_t>::value || std::is_same<T, int16_t>::value ||
        std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value,
    Error>::type
deserialize(RPCByteChannel &C, T &V) {
  if (auto Err = C.readBytes(reinterpret_cast<char *>(&V), sizeof(T)))
    return Err;
  support::endian::byte_swap<T, support::big>(V);
  return Error::success();
}

/// RPC channel serialization for enums.
template <typename T>
typename std::enable_if<std::is_enum<T>::value, Error>::type
serialize(RPCByteChannel &C, T V) {
  return serialize(C, static_cast<typename std::underlying_type<T>::type>(V));
}

/// RPC channel deserialization for enums.
template <typename T>
typename std::enable_if<std::is_enum<T>::value, Error>::type
deserialize(RPCByteChannel &C, T &V) {
  typename std::underlying_type<T>::type Tmp;
  Error Err = deserialize(C, Tmp);
  V = static_cast<T>(Tmp);
  return Err;
}

/// RPC channel serialization for bools.
inline Error serialize(RPCByteChannel &C, bool V) {
  uint8_t VN = V ? 1 : 0;
  return C.appendBytes(reinterpret_cast<const char *>(&VN), 1);
}

/// RPC channel deserialization for bools.
inline Error deserialize(RPCByteChannel &C, bool &V) {
  uint8_t VN = 0;
  if (auto Err = C.readBytes(reinterpret_cast<char *>(&VN), 1))
    return Err;

  V = (VN != 0);
  return Error::success();
}

/// RPC channel serialization for StringRefs.
/// Note: There is no corresponding deseralization for this, as StringRef
/// doesn't own its memory and so can't hold the deserialized data.
inline Error serialize(RPCByteChannel &C, StringRef S) {
  if (auto Err = serialize(C, static_cast<uint64_t>(S.size())))
    return Err;
  return C.appendBytes((const char *)S.bytes_begin(), S.size());
}

/// RPC channel serialization for std::strings.
inline Error serialize(RPCByteChannel &C, const std::string &S) {
  return serialize(C, StringRef(S));
}

/// RPC channel deserialization for std::strings.
inline Error deserialize(RPCByteChannel &C, std::string &S) {
  uint64_t Count;
  if (auto Err = deserialize(C, Count))
    return Err;
  S.resize(Count);
  return C.readBytes(&S[0], Count);
}

// Serialization helper for std::tuple.
template <typename TupleT, size_t... Is>
inline Error serializeTupleHelper(RPCByteChannel &C, const TupleT &V,
                                  llvm::index_sequence<Is...> _) {
  return serializeSeq(C, std::get<Is>(V)...);
}

/// RPC channel serialization for std::tuple.
template <typename... ArgTs>
inline Error serialize(RPCByteChannel &C, const std::tuple<ArgTs...> &V) {
  return serializeTupleHelper(C, V, llvm::index_sequence_for<ArgTs...>());
}

// Serialization helper for std::tuple.
template <typename TupleT, size_t... Is>
inline Error deserializeTupleHelper(RPCByteChannel &C, TupleT &V,
                                    llvm::index_sequence<Is...> _) {
  return deserializeSeq(C, std::get<Is>(V)...);
}

/// RPC channel deserialization for std::tuple.
template <typename... ArgTs>
inline Error deserialize(RPCByteChannel &C, std::tuple<ArgTs...> &V) {
  return deserializeTupleHelper(C, V, llvm::index_sequence_for<ArgTs...>());
}

/// RPC channel serialization for ArrayRef<T>.
template <typename T> Error serialize(RPCByteChannel &C, const ArrayRef<T> &A) {
  if (auto Err = serialize(C, static_cast<uint64_t>(A.size())))
    return Err;

  for (const auto &E : A)
    if (auto Err = serialize(C, E))
      return Err;

  return Error::success();
}

/// RPC channel serialization for std::array<T>.
template <typename T> Error serialize(RPCByteChannel &C,
                                      const std::vector<T> &V) {
  return serialize(C, ArrayRef<T>(V));
}

/// RPC channel deserialization for std::array<T>.
template <typename T> Error deserialize(RPCByteChannel &C, std::vector<T> &V) {
  uint64_t Count = 0;
  if (auto Err = deserialize(C, Count))
    return Err;

  V.resize(Count);
  for (auto &E : V)
    if (auto Err = deserialize(C, E))
      return Err;

  return Error::success();
}

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RPCBYTECHANNEL_H
