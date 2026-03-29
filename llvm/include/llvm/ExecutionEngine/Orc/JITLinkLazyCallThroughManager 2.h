//===- JITLinkLazyCallThroughManager.h - JITLink based laziness -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redirectable Symbol Manager implementation using JITLink
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_JITLINKLAZYCALLTHROUGHMANAGER_H
#define LLVM_EXECUTIONENGINE_ORC_JITLINKLAZYCALLTHROUGHMANAGER_H

#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/RedirectionManager.h"
#include "llvm/Support/StringSaver.h"

#include <atomic>

namespace llvm {
namespace orc {} // namespace orc
} // namespace llvm

#endif // LLVM_EXECUTIONENGINE_ORC_JITLINKLAZYCALLTHROUGHMANAGER_H
