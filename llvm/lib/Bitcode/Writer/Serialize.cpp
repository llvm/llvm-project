//==- Serialize.cpp - Generic Object Serialization to Bitcode ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the internal methods used for object serialization.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/Serialize.h"
#include "string.h"

#ifdef DEBUG_BACKPATCH
#include "llvm/Support/Streams.h"
#endif

using namespace llvm;

Serializer::Serializer(BitstreamWriter& stream)
  : Stream(stream), BlockLevel(0) {}

Serializer::~Serializer() {
  if (inRecord())
    EmitRecord();

  while (BlockLevel > 0)
    Stream.ExitBlock();
   
  Stream.FlushToWord();
}

void Serializer::EmitRecord() {
  assert(Record.size() > 0 && "Cannot emit empty record.");
  Stream.EmitRecord(8,Record);
  Record.clear();
}

void Serializer::EnterBlock(unsigned BlockID,unsigned CodeLen) {
  FlushRecord();
  Stream.EnterSubblock(BlockID,CodeLen);
  ++BlockLevel;
}

void Serializer::ExitBlock() {
  assert (BlockLevel > 0);
  --BlockLevel;
  FlushRecord();
  Stream.ExitBlock();
}

void Serializer::EmitInt(uint64_t X) {
  assert (BlockLevel > 0);
  Record.push_back(X);
}

void Serializer::EmitSInt(int64_t X) {
  if (X >= 0)
    EmitInt(X << 1);
  else
    EmitInt((-X << 1) | 1);
}

void Serializer::EmitCStr(const char* s, const char* end) {
  Record.push_back(end - s);
  
  while(s != end) {
    Record.push_back(*s);
    ++s;
  }
}

void Serializer::EmitCStr(const char* s) {
  EmitCStr(s,s+strlen(s));
}

SerializedPtrID Serializer::getPtrId(const void* ptr) {
  if (!ptr)
    return 0;
  
  MapTy::iterator I = PtrMap.find(ptr);
  
  if (I == PtrMap.end()) {
    unsigned id = PtrMap.size()+1;
#ifdef DEBUG_BACKPATCH
    llvm::cerr << "Registered PTR: " << ptr << " => " << id << "\n";
#endif
    PtrMap[ptr] = id;
    return id;
  }
  else return I->second;
}

bool Serializer::isRegistered(const void* ptr) const {
  MapTy::const_iterator I = PtrMap.find(ptr);
  return I != PtrMap.end();
}


#define INT_EMIT(TYPE)\
void SerializeTrait<TYPE>::Emit(Serializer&S, TYPE X) { S.EmitInt(X); }

INT_EMIT(bool)
INT_EMIT(unsigned char)
INT_EMIT(unsigned short)
INT_EMIT(unsigned int)
INT_EMIT(unsigned long)

#define SINT_EMIT(TYPE)\
void SerializeTrait<TYPE>::Emit(Serializer&S, TYPE X) { S.EmitSInt(X); }

SINT_EMIT(signed char)
SINT_EMIT(signed short)
SINT_EMIT(signed int)
SINT_EMIT(signed long)
