//===-- LaunchFlavor.h ---------------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LaunchFlavor_h
#define LaunchFlavor_h

namespace lldb_private {
namespace process_darwin {

enum class LaunchFlavor {
  Default = 0,
  PosixSpawn = 1,
  ForkExec = 2,
#ifdef WITH_SPRINGBOARD
  SpringBoard = 3,
#endif
#ifdef WITH_BKS
  BKS = 4,
#endif
#ifdef WITH_FBS
  FBS = 5
#endif
};
}
} // namespaces

#endif /* LaunchFlavor_h */
