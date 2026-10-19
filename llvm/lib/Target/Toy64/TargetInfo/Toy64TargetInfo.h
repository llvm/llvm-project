//===-- Toy64TargetInfo.h - Toy64 Target Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_TOY64_TARGETINFO_TOY64TARGETINFO_H
#define LLVM_LIB_TARGET_TOY64_TARGETINFO_TOY64TARGETINFO_H

namespace llvm {

    class Target;

    Target &getTheToy64Target();

}

#endif