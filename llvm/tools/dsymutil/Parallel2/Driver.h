//===- Driver.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_DSYMUTIL_PARALLEL2_DRIVER_H
#define LLVM_TOOLS_DSYMUTIL_PARALLEL2_DRIVER_H

namespace llvm {
namespace opt {
class InputArgList;
} // namespace opt

namespace dsymutil {
namespace parallel2 {

/// dsymutil --linker=parallel2 entry point.
int link(const opt::InputArgList &Args);

} // namespace parallel2
} // namespace dsymutil
} // namespace llvm

#endif
