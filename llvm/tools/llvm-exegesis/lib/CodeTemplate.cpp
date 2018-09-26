//===-- CodeTemplate.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "CodeTemplate.h"

namespace exegesis {

CodeTemplate::CodeTemplate(CodeTemplate &&) = default;

CodeTemplate &CodeTemplate::operator=(CodeTemplate &&) = default;

} // namespace exegesis
