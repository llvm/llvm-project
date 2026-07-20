//===--- DXILAttributes.cpp - attribute helper function -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILAttributes.h"
#include "llvm/IR/AttributeMask.h"

using namespace llvm;
using namespace llvm::dxil;

namespace llvm {
namespace dxil {
const AttributeMask &getNonDXILAttributeMask() {
  static const AttributeMask Result = [] {
    AttributeMask DXILAttributeMask;
    for (Attribute::AttrKind Kind : {Attribute::Alignment,
                                     Attribute::AlwaysInline,
                                     Attribute::Builtin,
                                     Attribute::ByVal,
                                     Attribute::InAlloca,
                                     Attribute::Cold,
                                     Attribute::Convergent,
                                     Attribute::InlineHint,
                                     Attribute::InReg,
                                     Attribute::JumpTable,
                                     Attribute::MinSize,
                                     Attribute::Naked,
                                     Attribute::Nest,
                                     Attribute::NoAlias,
                                     Attribute::NoBuiltin,
                                     Attribute::NoDuplicate,
                                     Attribute::NoImplicitFloat,
                                     Attribute::NoInline,
                                     Attribute::NonLazyBind,
                                     Attribute::NonNull,
                                     Attribute::Dereferenceable,
                                     Attribute::DereferenceableOrNull,
                                     Attribute::Memory,
                                     Attribute::NoRedZone,
                                     Attribute::NoReturn,
                                     Attribute::NoUnwind,
                                     Attribute::OptimizeForSize,
                                     Attribute::OptimizeNone,
                                     Attribute::ReadNone,
                                     Attribute::ReadOnly,
                                     Attribute::Returned,
                                     Attribute::ReturnsTwice,
                                     Attribute::SExt,
                                     Attribute::StackAlignment,
                                     Attribute::StackProtect,
                                     Attribute::StackProtectReq,
                                     Attribute::StackProtectStrong,
                                     Attribute::SafeStack,
                                     Attribute::StructRet,
                                     Attribute::SanitizeAddress,
                                     Attribute::SanitizeThread,
                                     Attribute::SanitizeMemory,
                                     Attribute::UWTable,
                                     Attribute::ZExt})
      DXILAttributeMask.addAttribute(Kind);
    AttributeMask Result;
    for (Attribute::AttrKind Kind = Attribute::None;
         Kind != Attribute::EndAttrKinds;
         Kind = Attribute::AttrKind(Kind + 1)) {
      if (!DXILAttributeMask.contains(Kind))
        Result.addAttribute(Kind);
    }
    return Result;
  }();
  return Result;
}
} // namespace dxil
} // namespace llvm
