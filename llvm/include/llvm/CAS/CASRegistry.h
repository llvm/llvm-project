//===- llvm/CAS/CASRegistry.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/CAS/ActionCache.h"
#include "llvm/CAS/CASID.h"
#include "llvm/CAS/CASReference.h"
#include "llvm/CAS/ObjectStore.h"
#include "llvm/Support/Error.h"

#ifndef LLVM_CAS_CASREGISTRY_H
#define LLVM_CAS_CASREGISTRY_H

namespace llvm::cas {

/// Create ObjectStore from a string identifier.
/// Currently the string identifier is using URL scheme with following supported
/// schemes:
///  * InMemory CAS: mem://
///  * OnDisk CAS: file://${PATH_TO_ONDISK_CAS}
///  * PlugIn CAS: plugin://${PATH_TO_PLUGIN}?${OPT1}=${VAL1}&${OPT2}=${VAL2}..
/// If no URL scheme is used, it defaults to following (but might change in
/// future)
/// For the plugin scheme, use argument "ondisk-path=${PATH}" to choose the
/// on-disk directory that the plugin should use, otherwise the default
/// OnDiskCAS location will be used.
/// FIXME: Need to implement proper URL encoding scheme that allows "%".
Expected<std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>
createCASFromIdentifier(StringRef Id);

/// Check if a string is a CAS identifier.
bool isRegisteredCASIdentifier(StringRef Config);

/// Register a URL scheme to CAS Identifier.
using ObjectStoreCreateFuncTy = Expected<
    std::pair<std::shared_ptr<ObjectStore>, std::shared_ptr<ActionCache>>>(
    const Twine &);
void registerCASURLScheme(StringRef Prefix, ObjectStoreCreateFuncTy *Func);

} // namespace llvm::cas

#endif
