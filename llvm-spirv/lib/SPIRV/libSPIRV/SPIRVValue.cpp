//===- SPIRVValue.cpp - Class to represent a SPIR-V Value -------*- C++ -*-===//
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
/// This file defines the values defined in SPIR-V spec with op codes.
///
/// The name of the SPIR-V values follow the op code name in the spec.
/// This is for readability and ease of using macro to handle types.
//
//===----------------------------------------------------------------------===//

#include "SPIRVValue.h"
#include "SPIRVEnum.h"
namespace SPIRV {
void SPIRVValue::setAlignment(SPIRVWord A) {
  if (A == 0) {
    eraseDecorate(DecorationAlignment);
    return;
  }
  addDecorate(new SPIRVDecorate(DecorationAlignment, this, A));
  SPIRVDBG(spvdbgs() << "Set alignment " << A << " for obj " << Id << "\n")
}

bool SPIRVValue::hasAlignment(SPIRVWord *Result) const {
  return hasDecorate(DecorationAlignment, 0, Result);
}

bool SPIRVValue::isVolatile() const { return hasDecorate(DecorationVolatile); }

void SPIRVValue::setVolatile(bool IsVolatile) {
  if (!IsVolatile) {
    eraseDecorate(DecorationVolatile);
    return;
  }
  addDecorate(new SPIRVDecorate(DecorationVolatile, this));
  SPIRVDBG(spvdbgs() << "Set volatile "
                     << " for obj " << Id << "\n")
}

bool SPIRVValue::hasNoSignedWrap() const {
  return hasDecorate(DecorationNoSignedWrap);
}

void SPIRVValue::setNoSignedWrap(bool HasNoSignedWrap) {
  if (!HasNoSignedWrap) {
    eraseDecorate(DecorationNoSignedWrap);
    return;
  }
  addDecorate(new SPIRVDecorate(DecorationNoSignedWrap, this));
  SPIRVDBG(spvdbgs() << "Set nsw "
                     << " for obj " << Id << "\n")
}

bool SPIRVValue::hasNoUnsignedWrap() const {
  return hasDecorate(DecorationNoUnsignedWrap);
}

void SPIRVValue::setNoUnsignedWrap(bool HasNoUnsignedWrap) {
  if (!HasNoUnsignedWrap) {
    eraseDecorate(DecorationNoUnsignedWrap);
    return;
  }
  addDecorate(new SPIRVDecorate(DecorationNoUnsignedWrap, this));
  SPIRVDBG(spvdbgs() << "Set nuw "
                     << " for obj " << Id << "\n")
}

} // namespace SPIRV
