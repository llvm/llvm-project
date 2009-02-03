//===- lib/System/IncludeFile.cpp - Ensure Linking Of Implementation -----===//
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

#include "llvm/System/IncludeFile.h"

using namespace llvm;

// This constructor is used to ensure linking of other modules. See the
// llvm/System/IncludeFile.h header for details. 
IncludeFile::IncludeFile(const void*) {}
