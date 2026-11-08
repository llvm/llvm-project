//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_REGISTER_COMMON_REGISTERINFO_POWERPC_INTERNAL_H
#define LLDB_SOURCE_REGISTER_COMMON_REGISTERINFO_POWERPC_INTERNAL_H
#include "lldb/Register/RegisterDescription.h"
namespace lldb_private {
RegisterDescription
MakePowerPCDescription(llvm::ArrayRef<RegisterDescriptor> values,
                       bool normalize_identities);
}
#endif
