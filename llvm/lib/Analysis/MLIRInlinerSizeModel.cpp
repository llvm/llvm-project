//===- MLIRInlinerSizeModel.cpp - MLIR inliner model wrapper ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements the wrapper around the MLIR-translated MLGO inliner
/// model.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MLIRInlinerSizeModel.h"

#include "llvm/Support/ErrorHandling.h"

#include <math.h>
#include <memory>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

// Generated models may name the wrapper class either `mainClass` or
// `actionClass`. Normalize both spellings to a TU-local name so the inliner and
// regalloc wrappers do not emit conflicting global symbols.
#define actionClass MLIRInlinerGeneratedModel
#define mainClass MLIRInlinerGeneratedModel
#include "llvm/Analysis/MLIRInlinerSizeModel.inc"
#undef actionClass
#undef mainClass

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace llvm {

namespace {

using GeneratedInlinerModel = MLIRInlinerGeneratedModel;

struct NamedArg {
  const char *FeedName;
  const char *ModelName;
};

constexpr NamedArg InlinerArgs[] = {
    {"feed_dead_blocks", "dead_blocks"},
    {"feed_case_cluster_penalty", "case_cluster_penalty"},
    {"feed_sroa_savings", "sroa_savings"},
    {"feed_jump_table_penalty", "jump_table_penalty"},
    {"feed_callsite_height", "callsite_height"},
    {"feed_callee_basic_block_count", "callee_basic_block_count"},
    {"feed_call_argument_setup", "call_argument_setup"},
    {"feed_lowered_call_arg_setup", "lowered_call_arg_setup"},
    {"feed_simplified_instructions", "simplified_instructions"},
    {"feed_nr_ctant_params", "nr_ctant_params"},
    {"feed_is_multiple_blocks", "is_multiple_blocks"},
    {"feed_load_elimination", "load_elimination"},
    {"feed_edge_count", "edge_count"},
    {"feed_caller_users", "caller_users"},
    {"feed_caller_conditionally_executed_blocks",
     "caller_conditionally_executed_blocks"},
    {"feed_constant_offset_ptr_args", "constant_offset_ptr_args"},
    {"feed_callsite_cost", "callsite_cost"},
    {"feed_caller_basic_block_count", "caller_basic_block_count"},
    {"feed_load_relative_intrinsic", "load_relative_intrinsic"},
    {"feed_indirect_call_penalty", "indirect_call_penalty"},
    {"feed_cost_estimate", "cost_estimate"},
    {"feed_threshold", "threshold"},
    {"feed_nested_inline_cost_estimate", "nested_inline_cost_estimate"},
    {"feed_unsimplified_common_instructions",
     "unsimplified_common_instructions"},
    {"feed_sroa_losses", "sroa_losses"},
    {"feed_num_loops", "num_loops"},
    {"feed_switch_penalty", "switch_penalty"},
    {"feed_callee_users", "callee_users"},
    {"feed_node_count", "node_count"},
    {"feed_constant_args", "constant_args"},
    {"feed_last_call_to_static_bonus", "last_call_to_static_bonus"},
    {"feed_cold_cc_penalty", "cold_cc_penalty"},
    {"feed_callee_conditionally_executed_blocks",
     "callee_conditionally_executed_blocks"},
    {"feed_call_penalty", "call_penalty"},
    {"feed_nested_inlines", "nested_inlines"},
};

constexpr size_t NumInlinerArgs = sizeof(InlinerArgs) / sizeof(*InlinerArgs);

static_assert(NumInlinerArgs == 35,
              "Unexpected number of inliner model inputs");

} // namespace

struct MLIRInlinerSizeModel::Impl {
  GeneratedInlinerModel Model{};

  bool hasBufferForName(const char *Name) const {
    return Model.reflectionMap.find(Name) != Model.reflectionMap.end();
  }

  void *getBufferForName(const char *Name) {
    return Model.getBufferForName(std::string(Name));
  }
};

MLIRInlinerSizeModel::MLIRInlinerSizeModel()
    : Model(std::make_unique<Impl>()) {}

MLIRInlinerSizeModel::~MLIRInlinerSizeModel() = default;

int MLIRInlinerSizeModel::LookupArgIndex(const std::string &Name) {
  for (size_t I = 0; I < NumInlinerArgs; ++I)
    if (Name == InlinerArgs[I].FeedName &&
        Model->hasBufferForName(InlinerArgs[I].ModelName))
      return static_cast<int>(I);
  return -1;
}

int MLIRInlinerSizeModel::LookupResultIndex(const std::string &Name) {
  return Name == "fetch_inlining_decision" ? 0 : -1;
}

void *MLIRInlinerSizeModel::arg_data(int Index) {
  if (Index < 0 || static_cast<size_t>(Index) >= NumInlinerArgs)
    llvm_unreachable("invalid MLIR inliner input index");
  return Model->getBufferForName(InlinerArgs[Index].ModelName);
}

void *MLIRInlinerSizeModel::result_data(int Index) {
  if (Index != 0)
    llvm_unreachable("invalid MLIR inliner result index");
  return Result.data();
}

void MLIRInlinerSizeModel::Run() { Result[0] = (Model->Model)(); }

} // namespace llvm
