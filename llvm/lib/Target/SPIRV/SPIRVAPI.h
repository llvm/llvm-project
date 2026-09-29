//===-- SPIRVAPI.h - SPIR-V Backend API interface ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVAPI_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVAPI_H

#include <string>
#include <vector>

namespace llvm {
class Module;

extern "C" bool
SPIRVTranslateModule(Module *M, std::string &SpirvObj, std::string &ErrMsg,
                     const std::vector<std::string> &AllowExtNames,
                     const std::vector<std::string> &Opts);
} // namespace llvm

#endif // LLVM_LIB_TARGET_SPIRV_SPIRVAPI_H
