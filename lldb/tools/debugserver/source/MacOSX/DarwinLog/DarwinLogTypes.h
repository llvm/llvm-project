//===-- DarwinLogTypes.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DarwinLogTypes_h
#define DarwinLogTypes_h

enum FilterTarget {
  eFilterTargetInvalid,
  eFilterTargetActivity,
  eFilterTargetActivityChain,
  eFilterTargetCategory,
  eFilterTargetMessage,
  eFilterTargetSubsystem
};

#endif /* DarwinLogTypes_h */
