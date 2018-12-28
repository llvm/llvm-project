//===- SPIRVFunction.cpp - Class to represent a SPIR-V Function --- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Function class for SPIRV.
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

#include "SPIRVFunction.h"
#include "SPIRVBasicBlock.h"
#include "SPIRVEntry.h"
#include "SPIRVInstruction.h"
#include "SPIRVStream.h"

#include <algorithm>
#include <functional>
using namespace SPIRV;

SPIRVFunctionParameter::SPIRVFunctionParameter(SPIRVType *TheType,
                                               SPIRVId TheId,
                                               SPIRVFunction *TheParent,
                                               unsigned TheArgNo)
    : SPIRVValue(TheParent->getModule(), 3, OpFunctionParameter, TheType,
                 TheId),
      ParentFunc(TheParent), ArgNo(TheArgNo) {
  validate();
}

void SPIRVFunctionParameter::foreachAttr(
    std::function<void(SPIRVFuncParamAttrKind)> Func) {
  auto Locs = Decorates.equal_range(DecorationFuncParamAttr);
  for (auto I = Locs.first, E = Locs.second; I != E; ++I) {
    auto Attr = static_cast<SPIRVFuncParamAttrKind>(I->second->getLiteral(0));
    assert(isValid(Attr));
    Func(Attr);
  }
}

SPIRVDecoder SPIRVFunction::getDecoder(std::istream &IS) {
  return SPIRVDecoder(IS, *this);
}

void SPIRVFunction::encode(spv_ostream &O) const {
  getEncoder(O) << Type << Id << FCtrlMask << FuncType;
}

void SPIRVFunction::encodeChildren(spv_ostream &O) const {
  O << SPIRVNL();
  for (auto &I : Parameters)
    O << *I;
  O << SPIRVNL();
  for (auto &I : BBVec)
    O << *I;
  O << SPIRVFunctionEnd();
}

void SPIRVFunction::encodeExecutionModes(spv_ostream &O) const {
  for (auto &I : ExecModes)
    O << *I.second;
}

void SPIRVFunction::decode(std::istream &I) {
  SPIRVDecoder Decoder = getDecoder(I);
  Decoder >> Type >> Id >> FCtrlMask >> FuncType;
  Module->addFunction(this);
  SPIRVDBG(spvdbgs() << "Decode function: " << Id << '\n');

  Decoder.getWordCountAndOpCode();
  while (!I.eof()) {
    if (Decoder.OpCode == OpFunctionEnd)
      break;

    switch (Decoder.OpCode) {
    case OpFunctionParameter: {
      auto Param = static_cast<SPIRVFunctionParameter *>(Decoder.getEntry());
      assert(Param);
      Module->add(Param);
      Param->setParent(this);
      Parameters.push_back(Param);
      Decoder.getWordCountAndOpCode();
      break;
    }
    case OpLabel: {
      decodeBB(Decoder);
      break;
    }
    default:
      assert(0 && "Invalid SPIRV format");
    }
  }
}

/// Decode basic block and contained instructions.
/// Do it here instead of in BB:decode to avoid back track in input stream.
void SPIRVFunction::decodeBB(SPIRVDecoder &Decoder) {
  SPIRVBasicBlock *BB = static_cast<SPIRVBasicBlock *>(Decoder.getEntry());
  assert(BB);
  addBasicBlock(BB);
  SPIRVDBG(spvdbgs() << "Decode BB: " << BB->getId() << '\n');

  Decoder.setScope(BB);
  SPIRVEntry *DebugScope = nullptr;
  while (Decoder.getWordCountAndOpCode()) {
    if (Decoder.OpCode == OpFunctionEnd || Decoder.OpCode == OpLabel) {
      break;
    }

    if (Decoder.OpCode == OpLine) {
      Module->add(Decoder.getEntry());
      continue;
    }

    auto *Inst = static_cast<SPIRVInstruction *>(Decoder.getEntry());
    assert(Inst);
    if (Inst->getOpCode() == OpUndef) {
      Module->add(Inst);
    } else {
      if (Inst->isExtInst(SPIRVEIS_Debug, SPIRVDebug::Scope)) {
        DebugScope = Inst;
      } else if (Inst->isExtInst(SPIRVEIS_Debug, SPIRVDebug::NoScope)) {
        DebugScope = nullptr;
      } else {
        Inst->setDebugScope(DebugScope);
      }
      BB->addInstruction(Inst);
    }
  }
  Decoder.setScope(this);
}

void SPIRVFunction::foreachReturnValueAttr(
    std::function<void(SPIRVFuncParamAttrKind)> Func) {
  auto Locs = Decorates.equal_range(DecorationFuncParamAttr);
  for (auto I = Locs.first, E = Locs.second; I != E; ++I) {
    auto Attr = static_cast<SPIRVFuncParamAttrKind>(I->second->getLiteral(0));
    assert(isValid(Attr));
    Func(Attr);
  }
}
