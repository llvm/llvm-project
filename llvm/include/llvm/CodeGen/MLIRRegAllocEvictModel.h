//===- MLIRRegAllocEvictModel.h - MLIR regalloc model wrapper -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Wraps the MLIR-translated MLGO regalloc eviction model in the interface
/// expected by ReleaseModeModelRunner.
///
/// The generated `.inc` file only contains the lowered model class. This
/// wrapper adapts that name-based surface to the index-based
/// ReleaseModeModelRunner contract that the rest of LLVM already uses for
/// release-mode MLGO models.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MLIRREGALLOCEVICTMODEL_H
#define LLVM_CODEGEN_MLIRREGALLOCEVICTMODEL_H

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace llvm {

class MLIRRegAllocEvictModel final {
public:
  MLIRRegAllocEvictModel();
  ~MLIRRegAllocEvictModel();

  int LookupArgIndex(const std::string &Name);
  int LookupResultIndex(const std::string &Name);
  void *arg_data(int Index);
  void *result_data(int Index);
  void Run();

private:
  struct Impl;
  std::unique_ptr<Impl> Model;
  std::array<int64_t, 1> Result{};
};

} // namespace llvm

#endif
