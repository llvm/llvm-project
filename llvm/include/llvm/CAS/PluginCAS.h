//===- llvm/CAS/PluginCAS.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"

#ifndef LLVM_CAS_PLUGINCAS_H
#define LLVM_CAS_PLUGINCAS_H

namespace llvm::cas {

/// Create \c ObjectStore and \c ActionCache instances using the plugin
/// interface.
Expected<std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
createPluginCASDatabases(
    StringRef PluginPath, StringRef OnDiskPath,
    ArrayRef<std::pair<std::string, std::string>> PluginArgs);

} // namespace llvm::cas

#endif
