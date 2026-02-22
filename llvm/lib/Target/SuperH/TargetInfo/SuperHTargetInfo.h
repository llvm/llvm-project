//===-- SuperHTargetInfo.h - SuperH Target Implementation ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_TARGETINFO_SUPERHTARGETINFO_H
#define LLVM_LIB_TARGET_SUPERH_TARGETINFO_SUPERHTARGETINFO_H

namespace llvm {

class Target;

Target &getTheSuperHTarget();

} // namespace llvm

#endif // LLVM_LIB_TARGET_SUPERH_TARGETINFO_SUPERHTARGETINFO_H
