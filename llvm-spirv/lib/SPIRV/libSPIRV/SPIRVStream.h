//===- SPIRVStream.h - Class to represent a SPIR-V Stream -------*- C++ -*-===//
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
/// This file defines Word class for SPIR-V.
///
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVSTREAM_H
#define SPIRV_LIBSPIRV_SPIRVSTREAM_H

#include "SPIRVDebug.h"
#include "SPIRVExtInst.h"
#include "SPIRVModule.h"
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace SPIRV {

#ifndef _SPIRV_SUPPORT_TEXT_FMT
#define _SPIRV_SUPPORT_TEXT_FMT
#endif

#ifdef _SPIRV_SUPPORT_TEXT_FMT
// Use textual format for SPIRV.
extern bool SPIRVUseTextFormat;
#endif

class SPIRVFunction;
class SPIRVBasicBlock;

class SPIRVDecoder {
public:
  SPIRVDecoder(std::istream &InputStream, SPIRVModule &Module)
      : IS(InputStream), M(Module), WordCount(0), OpCode(OpNop), Scope(NULL) {}
  SPIRVDecoder(std::istream &InputStream, SPIRVFunction &F);
  SPIRVDecoder(std::istream &InputStream, SPIRVBasicBlock &BB);

  void setScope(SPIRVEntry *);
  bool getWordCountAndOpCode();
  SPIRVEntry *getEntry();
  void validate() const;

  std::istream &IS;
  SPIRVModule &M;
  SPIRVWord WordCount;
  Op OpCode;
  SPIRVEntry *Scope; // A function or basic block
};

class SPIRVEncoder {
public:
  explicit SPIRVEncoder(spv_ostream &OutputStream) : OS(OutputStream) {}
  spv_ostream &OS;
};

/// Output a new line in text mode. Do nothing in binary mode.
class SPIRVNL {
  friend spv_ostream &operator<<(spv_ostream &O, const SPIRVNL &E);
};

template <typename T>
const SPIRVDecoder &decodeBinary(const SPIRVDecoder &I, T &V) {
  uint32_t W;
  I.IS.read(reinterpret_cast<char *>(&W), sizeof(W));
  V = static_cast<T>(W);
  SPIRVDBG(spvdbgs() << "Read word: W = " << W << " V = " << V << '\n');
  return I;
}

template <typename T>
const SPIRVDecoder &operator>>(const SPIRVDecoder &I, T &V) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    uint32_t W;
    I.IS >> W;
    V = static_cast<T>(W);
    SPIRVDBG(spvdbgs() << "Read word: W = " << W << " V = " << V << '\n');
    return I;
  }
#endif
  return decodeBinary(I, V);
}

template <typename T>
const SPIRVDecoder &operator>>(const SPIRVDecoder &I, T *&P) {
  SPIRVId Id;
  I >> Id;
  P = static_cast<T *>(I.M.getEntry(Id));
  return I;
}

template <typename IterTy>
const SPIRVDecoder &operator>>(const SPIRVDecoder &Decoder,
                               const std::pair<IterTy, IterTy> &Range) {
  for (IterTy I = Range.first, E = Range.second; I != E; ++I)
    Decoder >> *I;
  return Decoder;
}

template <typename T>
const SPIRVDecoder &operator>>(const SPIRVDecoder &I, std::vector<T> &V) {
  for (size_t J = 0, E = V.size(); J != E; ++J)
    I >> V[J];
  return I;
}

template <typename T>
const SPIRVEncoder &operator<<(const SPIRVEncoder &O, T V) {
#ifdef _SPIRV_SUPPORT_TEXT_FMT
  if (SPIRVUseTextFormat) {
    O.OS << V << " ";
    return O;
  }
#endif
  uint32_t W = static_cast<uint32_t>(V);
  O.OS.write(reinterpret_cast<char *>(&W), sizeof(W));
  return O;
}

template <typename T>
const SPIRVEncoder &operator<<(const SPIRVEncoder &O, T *P) {
  return O << P->getId();
}

template <typename T>
const SPIRVEncoder &operator<<(const SPIRVEncoder &O, const std::vector<T> &V) {
  for (size_t I = 0, E = V.size(); I != E; ++I)
    O << V[I];
  return O;
}

template <typename IterTy>
const SPIRVEncoder &operator<<(const SPIRVEncoder &Encoder,
                               const std::pair<IterTy, IterTy> &Range) {
  for (IterTy I = Range.first, E = Range.second; I != E; ++I)
    Encoder << *I;
  return Encoder;
}

#define SPIRV_DEC_ENCDEC(Type)                                                 \
  const SPIRVEncoder &operator<<(const SPIRVEncoder &O, Type V);               \
  const SPIRVDecoder &operator>>(const SPIRVDecoder &I, Type &V);

SPIRV_DEC_ENCDEC(Op)
SPIRV_DEC_ENCDEC(Capability)
SPIRV_DEC_ENCDEC(Decoration)
SPIRV_DEC_ENCDEC(OCLExtOpKind)
SPIRV_DEC_ENCDEC(SPIRVDebugExtOpKind)
SPIRV_DEC_ENCDEC(LinkageType)

const SPIRVEncoder &operator<<(const SPIRVEncoder &O, const std::string &Str);
const SPIRVDecoder &operator>>(const SPIRVDecoder &I, std::string &Str);

} // namespace SPIRV
#endif // SPIRV_LIBSPIRV_SPIRVSTREAM_H
