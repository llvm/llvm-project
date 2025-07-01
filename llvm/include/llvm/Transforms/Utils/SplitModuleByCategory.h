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

/// Splits the given module \p M in parts. Every output part is being passed to
/// \p Callback for further possible processing. Every part corresponds to a
/// module's subset that is transitively reachable from some entry point group.
/// Every entry point group is defined by \p EntryPointCategorizer (EPC) as
/// follows: 1) If the function is not an entry point then Categorizer returns
/// std::nullopt. Therefore, the function doesn't belong to any group. However,
/// the function and global objects still can be associated with some output
/// parts if it is transitively used from some entry points. 2) If the function
/// belongs to some entry point group then EPC returns an integer which is an
/// identifier of the group. If two entry point belong to one group then EPC
/// returns exact identifiers for both of them.
///
/// Let A and B be global objects in the module. Transitive dependency relation
/// is defined such that: If global object A is used by global object B in any
/// way (e.g., store, bitcast, phi node, call), then "A" -> "B". Transitivity is
/// defined such that: If "A" -> "B" and "B" -> "C", then "A" -> "C". Examples
/// of dependencies:
/// - Function FA calls function FB
/// - Function FA uses global variable GA
/// - Global variable GA references (initialized with) function FB
/// - Function FA stores address of a function FB somewhere
///
/// The following cases are treated as dependencies between global objects:
/// 1. Global object A is used within by a global object B in any way (store,
///    bitcast, phi node, call, etc.): "A" -> "B" edge will be added to the
///    graph;
/// 2. function A performs an indirect call of a function with signature S and
///    there is a function B with signature S. "A" -> "B" edge will be added to
///    the graph;
///
/// FIXME: For now the algorithm supposes no recursion in the input Module. That
/// is going to be fixed in the near future.
void splitModuleTransitiveFromEntryPoints(
    std::unique_ptr<Module> M,
    function_ref<std::optional<int>(const Function &F)> EntryPointCategorizer,
    function_ref<void(std::unique_ptr<Module> Part)> Callback);

} // namespace llvm

#endif // LLVM_TRANSFORM_UTILS_SPLIT_MODULE_BY_CATEGORY_H
