//===- Utils.h - C API General Utilities ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines general utilities for C API. This file should not be
// included from C++ code other than C API implementation nor from C code.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CAPI_UTILS_H
#define MLIR_CAPI_UTILS_H

#include "mlir-c/Support.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// Printing helper.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
/// A simple raw ostream subclass that forwards write_impl calls to the
/// user-supplied callback together with opaque user-supplied data.
class CallbackOstream : public llvm::raw_ostream {
public:
  CallbackOstream(std::function<void(MlirStringRef, void *)> callback,
                  void *opaqueData)
      : raw_ostream(/*unbuffered=*/true), callback(callback),
        opaqueData(opaqueData), pos(0u) {}

  void write_impl(const char *ptr, size_t size) override {
    MlirStringRef string = mlirStringRefCreate(ptr, size);
    callback(string, opaqueData);
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

private:
  std::function<void(MlirStringRef, void *)> callback;
  void *opaqueData;
  uint64_t pos;
};
} // namespace detail
} // namespace mlir

#endif // MLIR_CAPI_UTILS_H
