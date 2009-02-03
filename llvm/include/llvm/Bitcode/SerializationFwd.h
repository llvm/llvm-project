//==- SerializationFwd.h - Forward references for Serialization ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides forward references for bitcode object serialization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BITCODE_SERIALIZE_FWD
#define LLVM_BITCODE_SERIALIZE_FWD

namespace llvm {

class Serializer;
class Deserializer;  
template <typename T> struct SerializeTrait;  

typedef unsigned SerializedPtrID;

} // end namespace llvm

#endif
