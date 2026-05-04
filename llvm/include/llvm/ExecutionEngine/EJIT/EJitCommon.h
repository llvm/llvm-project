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

} // namespace ejit
} // namespace llvm

#endif
