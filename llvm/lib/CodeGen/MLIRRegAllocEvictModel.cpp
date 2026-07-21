//===- MLIRRegAllocEvictModel.cpp - MLIR regalloc model wrapper ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// This file implements the wrapper around the MLIR-translated MLGO
/// regalloc eviction model.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MLIRRegAllocEvictModel.h"

#include "llvm/Support/ErrorHandling.h"

#include <cstddef>
#include <memory>
#include <string>

// Generated models may name the wrapper class either `mainClass` or
// `actionClass`. Normalize both spellings to a TU-local name so the inliner and
// regalloc wrappers do not emit conflicting global symbols.
#define actionClass MLIRRegAllocGeneratedModel
#define mainClass MLIRRegAllocGeneratedModel
#include "llvm/CodeGen/MLIRRegAllocEvictModel.inc"
#undef actionClass
#undef mainClass

namespace llvm {

namespace {

using GeneratedRegAllocModel = MLIRRegAllocGeneratedModel;

struct NamedArg {
  const char *FeedName;
  const char *ModelName;
};

constexpr NamedArg RegAllocArgs[] = {
    {"feed_mask", "mask"},
    {"feed_is_free", "is_free"},
    {"feed_nr_urgent", "nr_urgent"},
    {"feed_nr_broken_hints", "nr_broken_hints"},
    {"feed_is_hint", "is_hint"},
    {"feed_is_local", "is_local"},
    {"feed_nr_rematerializable", "nr_rematerializable"},
    {"feed_nr_defs_and_uses", "nr_defs_and_uses"},
    {"feed_weighed_reads_by_max", "weighed_reads_by_max"},
    {"feed_weighed_writes_by_max", "weighed_writes_by_max"},
    {"feed_weighed_read_writes_by_max", "weighed_read_writes_by_max"},
    {"feed_weighed_indvars_by_max", "weighed_indvars_by_max"},
    {"feed_hint_weights_by_max", "hint_weights_by_max"},
    {"feed_start_bb_freq_by_max", "start_bb_freq_by_max"},
    {"feed_end_bb_freq_by_max", "end_bb_freq_by_max"},
    {"feed_hottest_bb_freq_by_max", "hottest_bb_freq_by_max"},
    {"feed_liverange_size", "liverange_size"},
    {"feed_use_def_density", "use_def_density"},
    {"feed_max_stage", "max_stage"},
    {"feed_min_stage", "min_stage"},
    {"feed_progress", "progress"},
};

constexpr size_t NumRegAllocArgs = sizeof(RegAllocArgs) / sizeof(*RegAllocArgs);

static_assert(NumRegAllocArgs == 21,
              "Unexpected number of regalloc model inputs");

} // namespace

struct MLIRRegAllocEvictModel::Impl {
  GeneratedRegAllocModel Model{};

  bool hasBufferForName(const char *Name) const {
    return Model.reflectionMap.find(Name) != Model.reflectionMap.end();
  }

  void *getBufferForName(const char *Name) {
    return Model.getBufferForName(std::string(Name));
  }
};

MLIRRegAllocEvictModel::MLIRRegAllocEvictModel()
    : Model(std::make_unique<Impl>()) {}

MLIRRegAllocEvictModel::~MLIRRegAllocEvictModel() = default;

int MLIRRegAllocEvictModel::LookupArgIndex(const std::string &Name) {
  for (size_t I = 0; I < NumRegAllocArgs; ++I)
    if (Name == RegAllocArgs[I].FeedName &&
        Model->hasBufferForName(RegAllocArgs[I].ModelName))
      return static_cast<int>(I);
  return -1;
}

int MLIRRegAllocEvictModel::LookupResultIndex(const std::string &Name) {
  return Name == "fetch_index_to_evict" ? 0 : -1;
}

void *MLIRRegAllocEvictModel::arg_data(int Index) {
  if (Index < 0 || static_cast<size_t>(Index) >= NumRegAllocArgs)
    llvm_unreachable("invalid MLIR regalloc eviction input index");
  return Model->getBufferForName(RegAllocArgs[Index].ModelName);
}

void *MLIRRegAllocEvictModel::result_data(int Index) {
  if (Index != 0)
    llvm_unreachable("invalid MLIR regalloc eviction result index");
  return Result.data();
}

void MLIRRegAllocEvictModel::Run() { Result[0] = (Model->Model)(); }

} // namespace llvm
