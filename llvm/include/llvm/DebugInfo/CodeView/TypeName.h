//===- TypeName.h --------------------------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_TYPENAME_H
#define LLVM_DEBUGINFO_CODEVIEW_TYPENAME_H

#include "llvm/DebugInfo/CodeView/TypeCollection.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"

namespace llvm {
namespace codeview {
std::string computeTypeName(TypeCollection &Types, TypeIndex Index);
}
} // namespace llvm

#endif
