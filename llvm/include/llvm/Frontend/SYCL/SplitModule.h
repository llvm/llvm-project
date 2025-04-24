//===-------- SplitModule.h - module split ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module for SYCL Offloading Kind.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FRONTEND_SYCL_SPLIT_MODULE_H
#define LLVM_FRONTEND_SYCL_SPLIT_MODULE_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Frontend/SYCL/Utils.h"

#include <memory>
#include <optional>
#include <string>

namespace llvm {

class Module;
class Function;

namespace sycl {

using PostSplitCallbackType = function_ref<void(std::unique_ptr<Module> Part)>;

/// Splits the given module \p M.
/// Every split image is being passed to \p Callback for further possible
/// processing.
void splitModule(std::unique_ptr<Module> M, IRSplitMode Mode,
                 PostSplitCallbackType Callback);

} // namespace sycl
} // namespace llvm

#endif // LLVM_FRONTEND_SYCL_SPLIT_MODULE_H
