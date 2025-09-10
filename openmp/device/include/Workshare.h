//===-------- Workshare.h - OpenMP Workshare interface ------------ C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_WORKSHARE_H
#define OMPTARGET_WORKSHARE_H

namespace ompx {

namespace workshare {

/// Initialize the worksharing machinery.
void init(bool IsSPMD);

} // namespace workshare

} // namespace ompx

#endif
