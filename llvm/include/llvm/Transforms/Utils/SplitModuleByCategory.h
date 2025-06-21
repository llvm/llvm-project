//===-------- SplitModuleByCategory.h - module split ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Functionality to split a module by categories.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORM_UTILS_SPLIT_MODULE_BY_CATEGORY_H
#define LLVM_TRANSFORM_UTILS_SPLIT_MODULE_BY_CATEGORY_H

#include "llvm/ADT/STLFunctionalExtras.h"

#include <memory>
#include <optional>
#include <string>

namespace llvm {

class Module;
class Function;

/// Splits the given module \p M using the given \p FunctionCategorizer.
/// \p FunctionCategorizer returns integer category for an input Function.
/// It may return std::nullopt if a function doesn't have a category.
/// Module's functions are being grouped by categories. Every such group
/// populates a call graph containing group's functions themselves and all
/// reachable functions and globals. Split outputs are populated from each call
/// graph associated with some category.
///
/// Every split output is being passed to \p Callback for further possible
/// processing.
///
/// Currently, the supported targets are SPIRV, AMDGPU and NVPTX.
void splitModuleByCategory(
    std::unique_ptr<Module> M,
    function_ref<std::optional<int>(const Function &F)> FunctionCategorizer,
    function_ref<void(std::unique_ptr<Module> Part)> Callback);

} // namespace llvm

#endif // LLVM_TRANSFORM_UTILS_SPLIT_MODULE_BY_CATEGORY_H
