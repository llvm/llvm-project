//===- llvm/unittest/ADT/CountCopyAndMove.cpp - Optional unit tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CountCopyAndMove.h"

using namespace llvm;

int CountCopyAndMove::CopyConstructions = 0;
int CountCopyAndMove::CopyAssignments = 0;
int CountCopyAndMove::MoveConstructions = 0;
int CountCopyAndMove::MoveAssignments = 0;
int CountCopyAndMove::Destructions = 0;
