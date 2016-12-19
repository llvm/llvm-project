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
#include "llvm/Support/thread.h"
#include <mutex>
#include <sstream>

namespace llvm {
namespace orc {
namespace rpc {

template <typename T>
class RPCTypeName;

/// TypeNameSequence is a utility for rendering sequences of types to a string
/// by rendering each type, separated by ", ".
template <typename... ArgTs> class RPCTypeNameSequence {};

/// Render an empty TypeNameSequence to an ostream.
template <typename OStream>
OStream &operator<<(OStream &OS, const RPCTypeNameSequence<> &V) {
  return OS;
}

/// Render a TypeNameSequence of a single type to an ostream.
template <typename OStream, typename ArgT>
OStream &operator<<(OStream &OS, const RPCTypeNameSequence<ArgT> &V) {
  OS << RPCTypeName<ArgT>::getName();
  return OS;
}

/// Render a TypeNameSequence of more than one type to an ostream.
template <typename OStream, typename ArgT1, typename ArgT2, typename... ArgTs>
OStream&
operator<<(OStream &OS, const RPCTypeNameSequence<ArgT1, ArgT2, ArgTs...> &V) {
  OS << RPCTypeName<ArgT1>::getName() << ", "
     << RPCTypeNameSequence<ArgT2, ArgTs...>();
  return OS;
}

template <>
class RPCTypeName<void> {
public:
  static const char* getName() { return "void"; }
};

template <>
class RPCTypeName<int8_t> {
public:
  static const char* getName() { return "int8_t"; }
};

template <>
class RPCTypeName<uint8_t> {
public:
  static const char* getName() { return "uint8_t"; }
};

template <>
class RPCTypeName<int16_t> {
public:
  static const char* getName() { return "int16_t"; }
};

template <>
class RPCTypeName<uint16_t> {
public:
  static const char* getName() { return "uint16_t"; }
};

template <>
class RPCTypeName<int32_t> {
public:
  static const char* getName() { return "int32_t"; }
};

template <>
class RPCTypeName<uint32_t> {
public:
  static const char* getName() { return "uint32_t"; }
};

template <>
class RPCTypeName<int64_t> {
public:
  static const char* getName() { return "int64_t"; }
};

template <>
class RPCTypeName<uint64_t> {
public:
  static const char* getName() { return "uint64_t"; }
};

template <>
class RPCTypeName<bool> {
public:
  static const char* getName() { return "bool"; }
};

template <>
class RPCTypeName<std::string> {
public:
  static const char* getName() { return "std::string"; }
};

template <typename T1, typename T2>
class RPCTypeName<std::pair<T1, T2>> {
public:
  static const char* getName() {
    std::lock_guard<std::mutex> Lock(NameMutex);
    if (Name.empty())
      raw_string_ostream(Name) << "std::pair<" << RPCTypeNameSequence<T1, T2>()
                               << ">";
    return Name.data();
  }
private:
  static std::mutex NameMutex;
  static std::string Name;
};

template <typename T1, typename T2>
std::mutex RPCTypeName<std::pair<T1, T2>>::NameMutex;
template <typename T1, typename T2>
std::string RPCTypeName<std::pair<T1, T2>>::Name;

template <typename... ArgTs>
class RPCTypeName<std::tuple<ArgTs...>> {
public:
  static const char* getName() {
    std::lock_guard<std::mutex> Lock(NameMutex);
    if (Name.empty())
      raw_string_ostream(Name) << "std::tuple<"
                               << RPCTypeNameSequence<ArgTs...>() << ">";
    return Name.data();
  }
private:
  static std::mutex NameMutex;
  static std::string Name;
};

template <typename... ArgTs>
std::mutex RPCTypeName<std::tuple<ArgTs...>>::NameMutex;
template <typename... ArgTs>
std::string RPCTypeName<std::tuple<ArgTs...>>::Name;

template <typename T>
class RPCTypeName<std::vector<T>> {
public:
  static const char*getName() {
    std::lock_guard<std::mutex> Lock(NameMutex);
    if (Name.empty())
      raw_string_ostream(Name) << "std::vector<" << RPCTypeName<T>::getName()
                               << ">";
    return Name.data();
  }

private:
  static std::mutex NameMutex;
  static std::string Name;
};

template <typename T>
std::mutex RPCTypeName<std::vector<T>>::NameMutex;
template <typename T>
std::string RPCTypeName<std::vector<T>>::Name;


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
template <typename ChannelT, typename WireType,
          typename ConcreteType = WireType, typename = void>
class SerializationTraits;

template <typename ChannelT>
class SequenceTraits {
public:
  static Error emitSeparator(ChannelT &C) { return Error::success(); }
  static Error consumeSeparator(ChannelT &C) { return Error::success(); }
};

/// Utility class for serializing sequences of values of varying types.
/// Specializations of this class contain 'serialize' and 'deserialize' methods
/// for the given channel. The ArgTs... list will determine the "over-the-wire"
/// types to be serialized. The serialize and deserialize methods take a list
/// CArgTs... ("caller arg types") which must be the same length as ArgTs...,
/// but may be different types from ArgTs, provided that for each CArgT there
/// is a SerializationTraits specialization
/// SerializeTraits<ChannelT, ArgT, CArgT> with methods that can serialize the
/// caller argument to over-the-wire value.
template <typename ChannelT, typename... ArgTs>
class SequenceSerialization;

template <typename ChannelT>
class SequenceSerialization<ChannelT> {
public:
  static Error serialize(ChannelT &C) { return Error::success(); }
  static Error deserialize(ChannelT &C) { return Error::success(); }
};

template <typename ChannelT, typename ArgT>
class SequenceSerialization<ChannelT, ArgT> {
public:

  template <typename CArgT>
  static Error serialize(ChannelT &C, const CArgT &CArg) {
    return SerializationTraits<ChannelT, ArgT, CArgT>::serialize(C, CArg);
  }

  template <typename CArgT>
  static Error deserialize(ChannelT &C, CArgT &CArg) {
    return SerializationTraits<ChannelT, ArgT, CArgT>::deserialize(C, CArg);
  }
};

template <typename ChannelT, typename ArgT, typename... ArgTs>
class SequenceSerialization<ChannelT, ArgT, ArgTs...> {
public:

  template <typename CArgT, typename... CArgTs>
  static Error serialize(ChannelT &C, const CArgT &CArg,
                         const CArgTs&... CArgs) {
    if (auto Err =
        SerializationTraits<ChannelT, ArgT, CArgT>::serialize(C, CArg))
      return Err;
    if (auto Err = SequenceTraits<ChannelT>::emitSeparator(C))
      return Err;
    return SequenceSerialization<ChannelT, ArgTs...>::serialize(C, CArgs...);
  }

  template <typename CArgT, typename... CArgTs>
  static Error deserialize(ChannelT &C, CArgT &CArg,
                           CArgTs&... CArgs) {
    if (auto Err =
        SerializationTraits<ChannelT, ArgT, CArgT>::deserialize(C, CArg))
      return Err;
    if (auto Err = SequenceTraits<ChannelT>::consumeSeparator(C))
      return Err;
    return SequenceSerialization<ChannelT, ArgTs...>::deserialize(C, CArgs...);
  }
};

template <typename ChannelT, typename... ArgTs>
Error serializeSeq(ChannelT &C, const ArgTs &... Args) {
  return SequenceSerialization<ChannelT, ArgTs...>::serialize(C, Args...);
}

template <typename ChannelT, typename... ArgTs>
Error deserializeSeq(ChannelT &C, ArgTs &... Args) {
  return SequenceSerialization<ChannelT, ArgTs...>::deserialize(C, Args...);
}

/// SerializationTraits default specialization for std::pair.
template <typename ChannelT, typename T1, typename T2>
class SerializationTraits<ChannelT, std::pair<T1, T2>> {
public:
  static Error serialize(ChannelT &C, const std::pair<T1, T2> &V) {
    return serializeSeq(C, V.first, V.second);
  }

  static Error deserialize(ChannelT &C, std::pair<T1, T2> &V) {
    return deserializeSeq(C, V.first, V.second);
  }
};

/// SerializationTraits default specialization for std::tuple.
template <typename ChannelT, typename... ArgTs>
class SerializationTraits<ChannelT, std::tuple<ArgTs...>> {
public:

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
};

/// SerializationTraits default specialization for std::vector.
template <typename ChannelT, typename T>
class SerializationTraits<ChannelT, std::vector<T>> {
public:

  /// Serialize a std::vector<T> from std::vector<T>.
  static Error serialize(ChannelT &C, const std::vector<T> &V) {
    if (auto Err = serializeSeq(C, static_cast<uint64_t>(V.size())))
      return Err;

    for (const auto &E : V)
      if (auto Err = serializeSeq(C, E))
        return Err;

    return Error::success();
  }

  /// Deserialize a std::vector<T> to a std::vector<T>.
  static Error deserialize(ChannelT &C, std::vector<T> &V) {
    uint64_t Count = 0;
    if (auto Err = deserializeSeq(C, Count))
      return Err;

    V.resize(Count);
    for (auto &E : V)
      if (auto Err = deserializeSeq(C, E))
        return Err;

    return Error::success();
  }
};

} // end namespace rpc
} // end namespace orc
} // end namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_RPCSERIALIZATION_H
