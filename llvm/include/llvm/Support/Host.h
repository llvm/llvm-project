//===-- llvm/Support/Host.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header is deprecated in favour of `llvm/TargetParser/Host.h`.
///
//===----------------------------------------------------------------------===//

#ifdef __GNUC__
#pragma GCC warning                                                            \
    "This header is deprecated, please use llvm/TargetParser/Host.h"
#endif
#include "llvm/TargetParser/Host.h"
