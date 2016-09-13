//===- llvm/ExecutionEngine/Orc/RPCSerialization.h --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_RPCSERIALIZATION_H
#define LLVM_EXECUTIONENGINE_ORC_RPCSERIALIZATION_H

#include "OrcError.h"
#include <mutex>
#include <sstream>
#include <thread>

namespace llvm {
namespace orc {
namespace remote {

/// The SerializationTraits<ChannelT, T> class describes how to serialize and
/// deserialize an instance of type T to/from an abstract channel of type
/// ChannelT. It also provides a representation of the type's name via the
/// getName method.
///
/// Specializations of this class should provide the following functions:
///
///   @code{.cpp}
///
///   static const char* getName();
///   static Error serialize(ChannelT&, const T&);
///   static Error deserialize(ChannelT&, T&);
///
///   @endcode
///
/// The third argument of SerializationTraits is intended to support SFINAE.
/// E.g.:
///
///   @code{.cpp}
///
///   class MyVirtualChannel { ... };
///
///   template <DerivedChannelT>
///   class SerializationTraits<DerivedChannelT, bool,
///         typename std::enable_if<
///           std::is_base_of<VirtChannel, DerivedChannel>::value
///         >::type> {
///   public:
///     static const char* getName() { ... };
///   }
///
///   @endcode
template <typename ChannelT, typename T, typename = void>
class SerializationTraits {};

/// TypeNameSequence is a utility for rendering sequences of types to a string
/// by rendering each type, separated by ", ".
template <typename ChannelT, typename... ArgTs> class TypeNameSequence {};

/// Render a TypeNameSequence of a single type to an ostream.
template <typename OStream, typename ChannelT, typename ArgT>
OStream &operator<<(OStream &OS, const TypeNameSequence<ChannelT, ArgT> &V) {
  OS << SerializationTraits<ChannelT, ArgT>::getName();
  return OS;
}

/// Render a TypeNameSequence of more than one type to an ostream.
template <typename OStream, typename ChannelT, typename ArgT1, typename ArgT2,
          typename... ArgTs>
OStream &
operator<<(OStream &OS,
           const TypeNameSequence<ChannelT, ArgT1, ArgT2, ArgTs...> &V) {
  OS << SerializationTraits<ChannelT, ArgT1>::getName() << ", "
     << TypeNameSequence<ChannelT, ArgT2, ArgTs...>();
  return OS;
}

/// RPC channel serialization for a variadic list of arguments.
template <typename ChannelT, typename T, typename... Ts>
Error serializeSeq(ChannelT &C, const T &Arg, const Ts &... Args) {
  if (auto Err = SerializationTraits<ChannelT, T>::serialize(C, Arg))
    return Err;
  return serializeSeq(C, Args...);
}

/// RPC channel serialization for an (empty) variadic list of arguments.
template <typename ChannelT> Error serializeSeq(ChannelT &C) {
  return Error::success();
}

/// RPC channel deserialization for a variadic list of arguments.
template <typename ChannelT, typename T, typename... Ts>
Error deserializeSeq(ChannelT &C, T &Arg, Ts &... Args) {
  if (auto Err = SerializationTraits<ChannelT, T>::deserialize(C, Arg))
    return Err;
  return deserializeSeq(C, Args...);
}

/// RPC channel serialization for an (empty) variadic list of arguments.
template <typename ChannelT> Error deserializeSeq(ChannelT &C) {
  return Error::success();
}

/// SerializationTraits default specialization for std::pair.
template <typename ChannelT, typename T1, typename T2>
class SerializationTraits<ChannelT, std::pair<T1, T2>> {
public:
  static const char *getName() {
    std::lock_guard<std::mutex> Lock(NameMutex);
    if (Name.empty())
      Name = (std::ostringstream()
              << "std::pair<" << TypeNameSequence<ChannelT, T1, T2>() << ">")
                 .str();

    return Name.data();
  }

  static Error serialize(ChannelT &C, const std::pair<T1, T2> &V) {
    return serializeSeq(C, V.first, V.second);
  }

  static Error deserialize(ChannelT &C, std::pair<T1, T2> &V) {
    return deserializeSeq(C, V.first, V.second);
  }

private:
  static std::mutex NameMutex;
  static std::string Name;
};

template <typename ChannelT, typename T1, typename T2>
std::mutex SerializationTraits<ChannelT, std::pair<T1, T2>>::NameMutex;

template <typename ChannelT, typename T1, typename T2>
std::string SerializationTraits<ChannelT, std::pair<T1, T2>>::Name;

/// SerializationTraits default specialization for std::tuple.
template <typename ChannelT, typename... ArgTs>
class SerializationTraits<ChannelT, std::tuple<ArgTs...>> {
public:
  static const char *getName() {
    std::lock_guard<std::mutex> Lock(NameMutex);
    if (Name.empty())
      Name = (std::ostringstream()
              << "std::tuple<" << TypeNameSequence<ChannelT, ArgTs...>() << ">")
                 .str();

    return Name.data();
  }

  /// RPC channel serialization for std::tuple.
  static Error serialize(ChannelT &C, const std::tuple<ArgTs...> &V) {
    return serializeTupleHelper(C, V, llvm::index_sequence_for<ArgTs...>());
  }

  /// RPC channel deserialization for std::tuple.
  static Error deserialize(ChannelT &C, std::tuple<ArgTs...> &V) {
    return deserializeTupleHelper(C, V, llvm::index_sequence_for<ArgTs...>());
  }

private:
  // Serialization helper for std::tuple.
  template <size_t... Is>
  static Error serializeTupleHelper(ChannelT &C, const std::tuple<ArgTs...> &V,
                                    llvm::index_sequence<Is...> _) {
    return serializeSeq(C, std::get<Is>(V)...);
  }

  // Serialization helper for std::tuple.
  template <size_t... Is>
  static Error deserializeTupleHelper(ChannelT &C, std::tuple<ArgTs...> &V,
                                      llvm::index_sequence<Is...> _) {
    return deserializeSeq(C, std::get<Is>(V)...);
  }

  static std::mutex NameMutex;
  static std::string Name;
};

template <typename ChannelT, typename... ArgTs>
std::mutex SerializationTraits<ChannelT, std::tuple<ArgTs...>>::NameMutex;

template <typename ChannelT, typename... ArgTs>
std::string SerializationTraits<ChannelT, std::tuple<ArgTs...>>::Name;

/// SerializationTraits default specialization for std::vector.
template <typename ChannelT, typename T>
class SerializationTraits<ChannelT, std::vector<T>> {
public:
  static const char *getName() {
    std::lock_guard<std::mutex> Lock(NameMutex);
    if (Name.empty())
      Name = (std::ostringstream() << "std::vector<"
                                   << TypeNameSequence<ChannelT, T>() << ">")
                 .str();
    return Name.data();
  }

  static Error serialize(ChannelT &C, const std::vector<T> &V) {
    if (auto Err = SerializationTraits<ChannelT, uint64_t>::serialize(
            C, static_cast<uint64_t>(V.size())))
      return Err;

    for (const auto &E : V)
      if (auto Err = SerializationTraits<ChannelT, T>::serialize(C, E))
        return Err;

    return Error::success();
  }

  static Error deserialize(ChannelT &C, std::vector<T> &V) {
    uint64_t Count = 0;
    if (auto Err =
            SerializationTraits<ChannelT, uint64_t>::deserialize(C, Count))
      return Err;

    V.resize(Count);
    for (auto &E : V)
      if (auto Err = SerializationTraits<ChannelT, T>::deserialize(C, E))
        return Err;

    return Error::success();
  }

private:
  static std::mutex NameMutex;
  static std::string Name;
};

template <typename ChannelT, typename T>
std::mutex SerializationTraits<ChannelT, std::vector<T>>::NameMutex;

template <typename ChannelT, typename T>
std::string SerializationTraits<ChannelT, std::vector<T>>::Name;

} // end namespace remote
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RPCSERIALIZATION_H
