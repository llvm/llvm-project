//==- Serialization.h - Generic Object Serialization to Bitcode ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines traits for primitive types used for both object
// serialization and deserialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_SERIALIZE
#define LLVM_BITCODE_SERIALIZE

#include "llvm/Bitcode/SerializationFwd.h"

namespace llvm {
  
/// SerializeTrait - SerializeTrait bridges between the Serializer/Deserializer
///  and the functions that serialize objects of specific types.  The default
///  behavior is to call static methods of the class for the object being
///  serialized, but this behavior can be changed by specializing this
///  template.  Classes only need to implement the methods corresponding
///  to the serialization scheme they want to support.  For example, "Read"
///  and "ReadVal" correspond to different deserialization schemes which make
///  sense for different types; a class need only implement one of them.
///  Serialization and deserialization of pointers are specially handled
///  by the Serializer and Deserializer using the EmitOwnedPtr, etc. methods.
///  To serialize the actual object referred to by a pointer, the class
///  of the object either must implement the methods called by the default
///  behavior of SerializeTrait, or specialize SerializeTrait.  This latter
///  is useful when one cannot add methods to an existing class (for example).
template <typename T>
struct SerializeTrait {
  static inline void Emit(Serializer& S, const T& X) { X.Emit(S); }
  static inline void Read(Deserializer& D, T& X) { X.Read(D); }
  static inline T* Create(Deserializer& D) { return T::Create(D); }
  
  template <typename Arg1>
  static inline T* Create(Deserializer& D, Arg1& arg1) {
    return T::Create(D, arg1);
  }
};

#define SERIALIZE_INT_TRAIT(TYPE)\
template <> struct SerializeTrait<TYPE> {\
  static void Emit(Serializer& S, TYPE X);\
  static void Read(Deserializer& S, TYPE& X); };

SERIALIZE_INT_TRAIT(bool)
SERIALIZE_INT_TRAIT(unsigned char)
SERIALIZE_INT_TRAIT(unsigned short)
SERIALIZE_INT_TRAIT(unsigned int)
SERIALIZE_INT_TRAIT(unsigned long)
  
SERIALIZE_INT_TRAIT(signed char)
SERIALIZE_INT_TRAIT(signed short)
SERIALIZE_INT_TRAIT(signed int)
SERIALIZE_INT_TRAIT(signed long)

#undef SERIALIZE_INT_TRAIT
  
} // end namespace llvm

#endif
