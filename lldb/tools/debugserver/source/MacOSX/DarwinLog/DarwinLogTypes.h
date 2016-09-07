//===-- DarwinLogTypes.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
