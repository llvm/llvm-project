//===- Private.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _OPENACC_PRIVATE_H_
#define _OPENACC_PRIVATE_H_

#include <cstddef>
#include <cstdint>

namespace llvm::acc::target {
int accIsPresent(void *);
void *accAlloc(size_t);
void accFree(void *);
void accMemcpyFromDevice(void *, void *, size_t);
void accMemcpyToDevice(void *, void *, size_t);
void accMemcpyD2D(void *, void *, size_t, int, int);
void accMapData(void *, void *, size_t);
void accUnmapData(void *);

void *accDataEnter(void *ArgBasePtr, void *ArgPtr, int64_t ArgSize,
                   int64_t ArgType, int64_t Async);
} // namespace llvm::acc::target

#endif // _OPENACC_PRIVATE_H_
