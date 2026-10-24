//===-- MCTargetPlugin.h - Example llvm-mc target plugin --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXAMPLES_MCTARGETPLUGIN_MCTARGETPLUGIN_H
#define LLVM_EXAMPLES_MCTARGETPLUGIN_MCTARGETPLUGIN_H

namespace llvm {
class Target;

/// The one Target this plugin registers, shared by the pieces of the example
/// that live in different files.
Target &getTheMCPluginTarget();
} // namespace llvm

#endif
