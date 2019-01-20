//===-- WebAssemblySelectionDAGInfo.cpp - WebAssembly SelectionDAG Info ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the WebAssemblySelectionDAGInfo class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyTargetMachine.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-selectiondag-info"

WebAssemblySelectionDAGInfo::~WebAssemblySelectionDAGInfo() {}
