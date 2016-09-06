//===-- CodeView.h - On-disk record types for CodeView ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides data structures useful for consuming on-disk
/// CodeView. It is based on information published by Microsoft at
/// https://github.com/Microsoft/microsoft-pdb/.
///
//===----------------------------------------------------------------------===//

// FIXME: Find a home for this in include/llvm/DebugInfo/CodeView/.

#ifndef LLVM_READOBJ_CODEVIEW_H
#define LLVM_READOBJ_CODEVIEW_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeIndex.h"
#include "llvm/Support/Endian.h"

namespace llvm {
namespace codeview {

using llvm::support::ulittle16_t;
using llvm::support::ulittle32_t;

/// Data in the the SUBSEC_FRAMEDATA subection.
struct FrameData {
  ulittle32_t RvaStart;
  ulittle32_t CodeSize;
  ulittle32_t LocalSize;
  ulittle32_t ParamsSize;
  ulittle32_t MaxStackSize;
  ulittle32_t FrameFunc;
  ulittle16_t PrologSize;
  ulittle16_t SavedRegsSize;
  ulittle32_t Flags;
  enum : uint32_t {
    HasSEH = 1 << 0,
    HasEH = 1 << 1,
    IsFunctionStart = 1 << 2,
  };
};


} // namespace codeview
} // namespace llvm

#endif // LLVM_READOBJ_CODEVIEW_H
