//===- SPIRVError.h - SPIR-V error code and checking ------------*- C++ -*-===//
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
//
// This file defines SPIRV error code and checking utility.
//
//===----------------------------------------------------------------------===//

#ifndef SPIRV_LIBSPIRV_SPIRVERROR_H
#define SPIRV_LIBSPIRV_SPIRVERROR_H

#include "SPIRVDebug.h"
#include "SPIRVUtil.h"
#include <sstream>
#include <string>

namespace SPIRV {

// Check condition and set error code and error msg.
// To use this macro, function checkError must be defined in the scope.
#define SPIRVCK(Condition, ErrCode, ErrMsg)                                    \
  getErrorLog().checkError(Condition, SPIRVEC_##ErrCode,                       \
                           std::string() + (ErrMsg), #Condition, __FILE__,     \
                           __LINE__)

// Check condition and set error code and error msg. If fail returns false.
#define SPIRVCKRT(Condition, ErrCode, ErrMsg)                                  \
  if (!getErrorLog().checkError(Condition, SPIRVEC_##ErrCode,                  \
                                std::string() + (ErrMsg), #Condition,          \
                                __FILE__, __LINE__))                           \
    return false;

// Defines error code enum type SPIRVErrorCode.
enum SPIRVErrorCode {
#define _SPIRV_OP(x, y) SPIRVEC_##x,
#include "SPIRVErrorEnum.h"
#undef _SPIRV_OP
};

// Defines OpErorMap which maps error code to a string describing the error.
template <> inline void SPIRVMap<SPIRVErrorCode, std::string>::init() {
#define _SPIRV_OP(x, y) add(SPIRVEC_##x, std::string(#x) + ": " + (y));
#include "SPIRVErrorEnum.h"
#undef _SPIRV_OP
}

typedef SPIRVMap<SPIRVErrorCode, std::string> SPIRVErrorMap;

class SPIRVErrorLog {
public:
  SPIRVErrorLog() : ErrorCode(SPIRVEC_Success) {}
  SPIRVErrorCode getError(std::string &ErrMsg) {
    ErrMsg = ErrorMsg;
    return ErrorCode;
  }
  void setError(SPIRVErrorCode ErrCode, const std::string &ErrMsg) {
    ErrorCode = ErrCode;
    ErrorMsg = ErrMsg;
  }
  // Check if Condition is satisfied and set ErrCode and DetailedMsg
  // if not. Returns true if no error.
  bool checkError(bool Condition, SPIRVErrorCode ErrCode,
                  const std::string &DetailedMsg = "",
                  const char *CondString = nullptr,
                  const char *FileName = nullptr, unsigned LineNumber = 0);

protected:
  SPIRVErrorCode ErrorCode;
  std::string ErrorMsg;
};

inline bool SPIRVErrorLog::checkError(bool Cond, SPIRVErrorCode ErrCode,
                                      const std::string &Msg,
                                      const char *CondString,
                                      const char *FileName, unsigned LineNo) {
  std::stringstream SS;
  if (Cond)
    return Cond;
  // Do not overwrite previous failure.
  if (ErrorCode != SPIRVEC_Success)
    return Cond;
  SS << SPIRVErrorMap::map(ErrCode) << " " << Msg;
  if (SPIRVDbgErrorMsgIncludesSourceInfo)
    SS << " [Src: " << FileName << ":" << LineNo << " " << CondString << " ]";
  setError(ErrCode, SS.str());
  if (SPIRVDbgAssertOnError) {
    spvdbgs() << SS.str() << '\n';
    spvdbgs().flush();
    assert(0);
  }
  return Cond;
}

} // namespace SPIRV

#endif // SPIRV_LIBSPIRV_SPIRVERROR_H
