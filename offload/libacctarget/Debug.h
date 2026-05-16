//===- Debug.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBACCTARGET_DEBUG_H_
#define _LIBACCTARGET_DEBUG_H_

namespace llvm::acc::target::debug {

// Debug types to use in libacctarget
constexpr const char *ADT_Init = "ACCInit";
constexpr const char *ADT_Mapping = "ACCMapping";
constexpr const char *ADT_Descriptor = "ACCDescriptor";
constexpr const char *ADT_Queue = "ACCQueue";
constexpr const char *ADT_Interface = "ACCInterface";
constexpr const char *ADT_Kernel = "ACCKernel";

} // namespace llvm::acc::target::debug

#endif // _LIBACCTARGET_DEBUG_H_
