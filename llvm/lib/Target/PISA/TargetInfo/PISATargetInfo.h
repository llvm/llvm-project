//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the PISA target registration interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_TARGETINFO_PISATARGETINFO_H
#define LLVM_LIB_TARGET_PISA_TARGETINFO_PISATARGETINFO_H

namespace llvm {

class Target;

Target &getThePISATarget();

} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_TARGETINFO_PISATARGETINFO_H
