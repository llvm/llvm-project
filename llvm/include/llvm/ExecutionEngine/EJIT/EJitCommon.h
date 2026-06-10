//===-- EJitCommon.h - EmbeddedJIT Shared Constants -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Shared string constants and configuration values used across the
//  EmbeddedJIT AOT passes, runtime library, and Clang CodeGen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITCOMMON_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITCOMMON_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"

namespace llvm {
namespace ejit {

//===----------------------------------------------------------------------===//
// Metadata kind names
//===----------------------------------------------------------------------===//
constexpr const char *MD_EJIT_METADATA = "ejit.metadata";
constexpr const char *MD_EJIT_MAY_CONST = "ejit.may_const";

//===----------------------------------------------------------------------===//
// Metadata entry tags (first operand of sub-nodes in !ejit.metadata)
//===----------------------------------------------------------------------===//
constexpr const char *TAG_EJIT_ENTRY = "ejit_entry";
constexpr const char *TAG_EJIT_PERIOD_LC = "ejit_period_lc";
constexpr const char *TAG_EJIT_PERIOD_ARR_IND = "ejit_period_arr_ind";
constexpr const char *TAG_EJIT_PERIOD_ARR = "ejit_period_arr";
constexpr const char *TAG_EJIT_PERIOD = "ejit_period";
constexpr const char *TAG_EJIT_MAY_CONST_FIELD = "ejit_may_const_field";
constexpr const char *TAG_EJIT_FUNC_IDX = "ejit_func_idx";

//===----------------------------------------------------------------------===//
// Global variable and section names
//===----------------------------------------------------------------------===//
constexpr const char *GV_EJIT_BITCODE = "__ejit_bitcode";
constexpr const char *SECT_EJIT_BITCODE = ".ejit.bitcode";
constexpr const char *FN_AUTO_REGISTER = "ejit_auto_register";
constexpr const char *CTORS_GLOBAL = "llvm.global_ctors";

//===----------------------------------------------------------------------===//
// Runtime function names (extern symbols called by AOT-generated code)
//===----------------------------------------------------------------------===//
constexpr const char *FN_REGISTER_BITCODE = "ejit_register_bitcode";
constexpr const char *FN_REGISTER_PERIOD_ARRAY = "ejit_register_period_array";
constexpr const char *FN_REGISTER_STATIC_VAR = "ejit_register_static_var";
constexpr const char *FN_COMPILE_OR_GET = "ejit_compile_or_get";
constexpr const char *FN_COMPILE_OR_GET_V2 = "ejit_compile_or_get_v2";
constexpr const char *FN_DEACTIVATE_ARRAY = "ejit_deactivate_array";
constexpr const char *FN_ACTIVATE_ARRAY = "ejit_activate_array";

//===----------------------------------------------------------------------===//
// Constructor priority (lower = later; 65535 runs last)
//===----------------------------------------------------------------------===//
constexpr unsigned EJIT_CTOR_PRIORITY = 65535;

//===----------------------------------------------------------------------===//
// Limits
//===----------------------------------------------------------------------===//
constexpr unsigned MAX_PERIOD_ARR_IND_PARAMS = 4;
constexpr unsigned MAX_PERIOD_ARR_SIZE = 100;

//===----------------------------------------------------------------------===//
// Deterministic funcIdx generation
//===----------------------------------------------------------------------===//

/// FNV-1a 32-bit hash used to compute a compile-time funcIdx from the
/// function name. The same hash is recomputed at runtime for registration
/// lookup, eliminating the need for string→idx map lookups on the cache-hit
/// path. Collisions are detected at ejit_init time.
inline uint32_t hashFuncName(StringRef name) {
  uint32_t h = 2166136261u;
  for (char c : name)
    h = (h ^ static_cast<uint8_t>(c)) * 16777619u;
  return h;
}

//===----------------------------------------------------------------------===//
// Metadata utility functions (shared across AOT passes)
//===----------------------------------------------------------------------===//

inline bool hasMDStringEntry(const MDNode *Node, StringRef Name) {
  if (!Node)
    return false;
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (!Sub || Sub->getNumOperands() == 0)
      continue;
    if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
      if (S->getString() == Name)
        return true;
  }
  return false;
}

inline StringRef getMDStringValue(const MDNode *Node, StringRef Tag) {
  if (!Node)
    return {};
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (Sub && Sub->getNumOperands() >= 2) {
      if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
        if (S->getString() == Tag)
          if (auto *V = dyn_cast<MDString>(Sub->getOperand(1)))
            return V->getString();
    }
  }
  return {};
}

inline uint32_t getMDIntValue(const MDNode *Node, StringRef Tag) {
  if (!Node)
    return 0;
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (Sub && Sub->getNumOperands() >= 3) {
      if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
        if (S->getString() == Tag)
          if (auto *C = dyn_cast<ConstantAsMetadata>(Sub->getOperand(2)))
            if (auto *CI = dyn_cast<ConstantInt>(C->getValue()))
              return static_cast<uint32_t>(CI->getZExtValue());
    }
  }
  return 0;
}

} // namespace ejit
} // namespace llvm

#endif
