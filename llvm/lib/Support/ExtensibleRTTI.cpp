//===----- lib/Support/ExtensibleRTTI.cpp - ExtensibleRTTI utilities ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ExtensibleRTTI.h"

void llvm::RTTIRoot::anchor() {}
char llvm::RTTIRoot::ID = 0;
