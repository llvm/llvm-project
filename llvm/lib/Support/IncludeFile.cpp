//===- lib/Support/IncludeFile.cpp - Ensure Linking Of Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IncludeFile constructor.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/IncludeFile.h"

using namespace llvm;

// This constructor is used to ensure linking of other modules. See the
// llvm/Support/IncludeFile.h header for details.
IncludeFile::IncludeFile(const void*) {}
