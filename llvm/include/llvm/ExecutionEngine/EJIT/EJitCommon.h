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
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include <cstdint>
#include <map>
#include <string>

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
constexpr const char *FN_REGISTER_LIFECYCLE = "ejit_register_lifecycle";
constexpr const char *FN_REGISTER_FUNCINDEX = "ejit_register_funcindex";
constexpr const char *FN_COMPILE_OR_GET = "ejit_compile_or_get";
constexpr const char *FN_TASKPOOL_COMPILE_OR_GET =
    "ejit_taskpool_compile_or_get";
constexpr const char *FN_TASKPOOL_RELEASE_READ = "ejit_taskpool_release_read";
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

//===----------------------------------------------------------------------===//
// Explicit, registration-time identity assignment for cross-module agreement.
//
// Neither funcIndex nor dimType can be derived independently per module: a
// modulo name hash collides (50 functions ~26%, 200 ~99% at 4096 slots; and
// fnv("cell")%8 == fnv("tenant")%8 for 8 dimType slots), and no AOT pass sees
// every final module. Both are therefore assigned ONCE, by name, in a
// process-global registry at registration time and read back by the wrapper
// through a per-function / per-lifecycle global the registration backfills:
//
//   * funcIndex: a dense index in [0, kEJitMaxFuncIndex) handed out by
//     EJitFuncRegistry. The wrapper loads @__ejit_funcidx_<name> (initialized
//     to kEJitInvalidFuncIndex) and falls back WITHOUT entering the taskpool
//     when it is still invalid (unregistered / capacity exhausted). The module
//     loader keys its table by the SAME registry index, so a distinct function
//     can never alias another's slot. See EJitFuncRegistry.h.
//   * dimType: a dense lifecycle slot in [0, kEJitMaxDimTypes) handed out by
//     EJitLifecycleRegistry, read back through @__ejit_dimtype_<name>. See
//     EJitLifecycleRegistry.h.
//
// Same name -> same index across every module and registration order; a new
// module never shifts an existing index; capacity exhaustion is a clean,
// propagated failure (ejit_init fails) — never a silent alias or hash collision
// on the correctness path.
//===----------------------------------------------------------------------===//

constexpr unsigned kEJitMaxDimTypes = 8;    // spec §5.1 MAX_DIM_TYPES
constexpr unsigned kEJitMaxInstances = 256; // spec §5.1 MAX_INSTANCES

// Flat-dedup capacity = max dense funcIndex (spec §3.5 inFlight_[]). Mirrors
// the runtime EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX; keep the two defaults in sync.
#ifndef EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX
#define EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX 4096u
#endif
constexpr uint32_t kEJitMaxFuncIndex = EJIT_SRE_TASKPOOL_MAX_FUNC_INDEX;

/// Sentinel for "no dimType" (unknown / unregistered lifecycle). Out of
/// [0, kEJitMaxDimTypes), so it can never be a valid dimType.
constexpr uint32_t kEJitInvalidDimType = 0xFFFFFFFFu;

/// Sentinel for "no funcIndex" (unregistered function or funcIndex capacity
/// exhausted). Out of [0, kEJitMaxFuncIndex); the wrapper treats it as
/// "fall back, never enter the taskpool".
constexpr uint32_t kEJitInvalidFuncIndex = 0xFFFFFFFFu;

} // namespace ejit
} // namespace llvm

#endif
