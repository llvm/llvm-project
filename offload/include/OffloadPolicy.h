//===-- OffloadPolicy.h - Configuration of offload behavior -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Configuration for offload behavior, e.g., if offload is disabled, can be
// disabled, is mandatory, etc.
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_OFFLOAD_POLICY_H
#define OMPTARGET_OFFLOAD_POLICY_H

#include "PluginManager.h"

enum kmp_target_offload_kind_t {
  tgt_disabled = 0,
  tgt_default = 1,
  tgt_mandatory = 2
};

extern "C" int __kmpc_get_target_offload(void) __attribute__((weak));

class OffloadPolicy {

  OffloadPolicy(PluginManager &PM) {
    // TODO: Check for OpenMP.
    switch ((kmp_target_offload_kind_t)__kmpc_get_target_offload()) {
    case tgt_disabled:
      Kind = DISABLED;
      return;
    case tgt_mandatory:
      Kind = MANDATORY;
      return;
    default:
      if (PM.getNumDevices()) {
        DP("Default TARGET OFFLOAD policy is now mandatory "
           "(devices were found)\n");
        Kind = MANDATORY;
      } else {
        DP("Default TARGET OFFLOAD policy is now disabled "
           "(no devices were found)\n");
        Kind = DISABLED;
      }
      return;
    };
  }

public:
  static const OffloadPolicy &get(PluginManager &PM) {
    static OffloadPolicy OP(PM);
    return OP;
  }

  enum OffloadPolicyKind { DISABLED, MANDATORY };

  OffloadPolicyKind Kind = MANDATORY;
};

#endif // OMPTARGET_OFFLOAD_POLICY_H
