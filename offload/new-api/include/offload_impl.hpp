//===- offload_impl.hpp- Implementation helpers for the Offload library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <offload_api.h>
#include <optional>
#include <string>

#include "llvm/ADT/StringRef.h"

std::optional<std::string> &LastErrorDetails();

struct offload_impl_result_t {
  offload_impl_result_t() = delete;
  offload_impl_result_t(offload_result_t Result) : Result(Result) {
    LastErrorDetails() = std::nullopt;
  }

  offload_impl_result_t(offload_result_t Result, std::string Details)
      : Result(Result) {
    assert(Result != OFFLOAD_RESULT_SUCCESS);
    LastErrorDetails() = Details;
  }

  operator offload_result_t() { return Result; }

private:
  offload_result_t Result;
};
