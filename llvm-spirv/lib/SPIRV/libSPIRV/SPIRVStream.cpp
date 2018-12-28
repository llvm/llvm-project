//===- SPIRVStream.cpp - Class to represent a SPIR-V Stream -----*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements SPIR-V stream class.
///
//===----------------------------------------------------------------------===//
#include "SPIRVStream.h"
#include "SPIRVDebug.h"
#include "SPIRVFunction.h"
#include "SPIRVNameMapEnum.h"
#include "SPIRVOpCode.h"

namespace SPIRV {

/// Write string with quote. Replace " with \".
static void writeQuotedString(spv_ostream &O, const std::string &Str) {
  O << '"';
  for (auto I : Str) {
    if (I == '"')
      O << '\\';
    O << I;
  }
  O << '"';
}

/// Read quoted string. Replace \" with ".
static void readQuotedString(std::istream &IS, std::string &Str) {
  char Ch = ' ';
  char PreCh = ' ';
  while (IS >> Ch && Ch != '"')
    ;

  if (IS >> PreCh && PreCh != '"') {
    while (IS >> Ch) {
      if (Ch == '"') {
        if (PreCh != '\\') {
          Str += PreCh;
          break;
        } else
          PreCh = Ch;
      } else {
        Str += PreCh;
        PreCh = Ch;
      }
    }
  }
}

#ifdef _SPIRV_SUPPORT_TEXT_FMT
bool SPIRVUseTextFormat = false;
#endif

SPIRVDecoder::SPIRVDecoder(std::istream &InputStream, SPIRVFunction &F)
    : IS(InputStream), M(*F.getModule()), WordCount(0), OpCode(OpNop),
      Scope(&F) {}

SPIRVDecoder::SPIRVDecoder(std::istream &InputStream, SPIRVBasicBlock &BB)
    : IS(InputStream), M(*BB.getModule()), WordCount(0), OpCode(OpNop),
      Scope(&BB) {}

void SPIRVDecoder::setScope(SPIRVEntry *TheScope) {
  assert(TheScope && (TheScope->getOpCode() == OpFunction ||
                      TheScope->getOpCode() == OpLabel));
  Scope = TheScope;
}

template <class T> const SPIRVDecoder &decode(const SPIRVDecoder &I, T &V) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    std::string W;
    I.IS >> W;
    V = getNameMap(V).rmap(W);
    SPIRVDBG(spvdbgs() << "Read word: W = " << W << " V = " << V << '\n');
    return I;
  }
#endif
  return decodeBinary(I, V);
}

template <class T> const SPIRVEncoder &encode(const SPIRVEncoder &O, T V) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    O.OS << getNameMap(V).map(V) << " ";
    return O;
  }
#endif
  return O << static_cast<SPIRVWord>(V);
}

#define SPIRV_DEF_ENCDEC(Type)                                                 \
  const SPIRVDecoder &operator>>(const SPIRVDecoder &I, Type &V) {             \
    return decode(I, V);                                                       \
  }                                                                            \
  const SPIRVEncoder &operator<<(const SPIRVEncoder &O, Type V) {              \
    return encode(O, V);                                                       \
  }

SPIRV_DEF_ENCDEC(Op)
SPIRV_DEF_ENCDEC(Capability)
SPIRV_DEF_ENCDEC(Decoration)
SPIRV_DEF_ENCDEC(OCLExtOpKind)
SPIRV_DEF_ENCDEC(SPIRVDebugExtOpKind)
SPIRV_DEF_ENCDEC(LinkageType)

// Read a string with padded 0's at the end so that they form a stream of
// words.
const SPIRVDecoder &operator>>(const SPIRVDecoder &I, std::string &Str) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    readQuotedString(I.IS, Str);
    SPIRVDBG(spvdbgs() << "Read string: \"" << Str << "\"\n");
    return I;
  }
#endif

  uint64_t Count = 0;
  char Ch;
  while (I.IS.get(Ch) && Ch != '\0') {
    Str += Ch;
    ++Count;
  }
  Count = (Count + 1) % 4;
  Count = Count ? 4 - Count : 0;
  for (; Count; --Count) {
    I.IS >> Ch;
    assert(Ch == '\0' && "Invalid string in SPIRV");
  }
  SPIRVDBG(spvdbgs() << "Read string: \"" << Str << "\"\n");
  return I;
}

// Write a string with padded 0's at the end so that they form a stream of
// words.
const SPIRVEncoder &operator<<(const SPIRVEncoder &O, const std::string &Str) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    writeQuotedString(O.OS, Str);
    return O;
  }
#endif

  size_t L = Str.length();
  O.OS.write(Str.c_str(), L);
  char Zeros[4] = {0, 0, 0, 0};
  O.OS.write(Zeros, 4 - L % 4);
  return O;
}

bool SPIRVDecoder::getWordCountAndOpCode() {
  if (IS.eof()) {
    WordCount = 0;
    OpCode = OpNop;
    SPIRVDBG(spvdbgs() << "[SPIRVDecoder] getWordCountAndOpCode EOF "
                       << WordCount << " " << OpCode << '\n');
    return false;
  }
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    *this >> WordCount;
    assert(!IS.bad() && "SPIRV stream is bad");
    if (IS.fail()) {
      WordCount = 0;
      OpCode = OpNop;
      SPIRVDBG(spvdbgs() << "[SPIRVDecoder] getWordCountAndOpCode FAIL "
                         << WordCount << " " << OpCode << '\n');
      return false;
    }
    *this >> OpCode;
  } else {
#endif
    SPIRVWord WordCountAndOpCode;
    *this >> WordCountAndOpCode;
    WordCount = WordCountAndOpCode >> 16;
    OpCode = static_cast<Op>(WordCountAndOpCode & 0xFFFF);
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  }
#endif
  assert(!IS.bad() && "SPIRV stream is bad");
  if (IS.fail()) {
    WordCount = 0;
    OpCode = OpNop;
    SPIRVDBG(spvdbgs() << "[SPIRVDecoder] getWordCountAndOpCode FAIL "
                       << WordCount << " " << OpCode << '\n');
    return false;
  }
  SPIRVDBG(spvdbgs() << "[SPIRVDecoder] getWordCountAndOpCode " << WordCount
                     << " " << OpCodeNameMap::map(OpCode) << '\n');
  return true;
}

SPIRVEntry *SPIRVDecoder::getEntry() {
  if (WordCount == 0 || OpCode == OpNop)
    return nullptr;
  SPIRVEntry *Entry = SPIRVEntry::create(OpCode);
  assert(Entry);
  Entry->setModule(&M);
  if (isModuleScopeAllowedOpCode(OpCode) && !Scope) {
  } else
    Entry->setScope(Scope);
  Entry->setWordCount(WordCount);
  if (OpCode != OpLine)
    Entry->setLine(M.getCurrentLine());
  IS >> *Entry;
  if (Entry->isEndOfBlock() || OpCode == OpNoLine)
    M.setCurrentLine(nullptr);
  assert(!IS.bad() && !IS.fail() && "SPIRV stream fails");
  return Entry;
}

void SPIRVDecoder::validate() const {
  assert(OpCode != OpNop && "Invalid op code");
  assert(WordCount && "Invalid word count");
  assert(!IS.bad() && "Bad iInput stream");
}

spv_ostream &operator<<(spv_ostream &O, const SPIRVNL &E) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat)
    O << '\n';
#endif
  return O;
}

} // namespace SPIRV
