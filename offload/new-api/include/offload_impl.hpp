//===- offload_impl.hpp- Implementation helpers for the Offload library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#pragma once

#include <iostream>
#include <memory>
#include <offload_api.h>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

// Use the StringSet container to efficiently deduplicate repeated error
// strings (e.g. if the same error is hit constantly in a long running program)
llvm::StringSet<> &ErrorStrs();

// Use an unordered_set to avoid duplicates of error structs themselves.
// We cannot store the structs directly as returned pointers to them must always
// be valid, and a rehash of the set may invalidate them. This requires
// custom hash and equal_to function objects.
using ErrPtrT = std::unique_ptr<offload_error_struct_t>;
struct ErrPtrEqual {
  bool operator()(const ErrPtrT &lhs, const ErrPtrT &rhs) const {
    if (!lhs && !rhs) {
      return true;
    }
    if (!lhs || !rhs) {
      return false;
    }

    bool StrsEqual = false;
    if (lhs->details == NULL && rhs->details == NULL) {
      StrsEqual = true;
    } else if (lhs->details != NULL && rhs->details != NULL) {
      StrsEqual = (std::strcmp(lhs->details, rhs->details) == 0);
    }
    return (lhs->code == rhs->code) && StrsEqual;
  }
};
struct ErrPtrHash {
  size_t operator()(const ErrPtrT &e) const {
    if (!e) {
      // We shouldn't store empty errors (i.e. success), but just in case
      return 0lu;
    } else {
      return std::hash<int>{}(e->code);
    }
  }
};
using ErrSetT = std::unordered_set<ErrPtrT, ErrPtrHash, ErrPtrEqual>;
ErrSetT &Errors();

struct offload_impl_result_t {
  offload_impl_result_t(std::nullptr_t) : Result(OFFLOAD_SUCCESS) {}
  offload_impl_result_t(offload_errc_t Code) {
    if (Code == OFFLOAD_ERRC_SUCCESS) {
      Result = nullptr;
    } else {
      auto Err = std::unique_ptr<offload_error_struct_t>(
          new offload_error_struct_t{Code, nullptr});
      Result = Errors().emplace(std::move(Err)).first->get();
    }
  }

  offload_impl_result_t(offload_errc_t Code, llvm::StringRef Details) {
    assert(Code != OFFLOAD_ERRC_SUCCESS);
    Result = nullptr;
    auto DetailsStr = ErrorStrs().insert(Details).first->getKeyData();
    auto Err = std::unique_ptr<offload_error_struct_t>(
        new offload_error_struct_t{Code, DetailsStr});
    Result = Errors().emplace(std::move(Err)).first->get();
  }

  operator offload_result_t() { return Result; }

private:
  offload_result_t Result;
};
