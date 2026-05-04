//===-- EJitRegistration.h - AOT Registration Callbacks -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_EJIT_EJITREGISTRATION_H
#define LLVM_EXECUTIONENGINE_EJIT_EJITREGISTRATION_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Called by ejit_auto_register() during .ctor phase to register
/// extracted bitcode for an entry function.
void ejit_register_bitcode(const char *funcName,
                           const uint8_t *bitcodeData, uint64_t bitcodeSize);

/// Register a period array global variable.
void ejit_register_period_array(const char *periodName,
                                const char *varName,
                                void *baseAddr, uint64_t arraySize);

/// Register a static period variable.
void ejit_register_static_var(const char *varName, void *varAddr);

#ifdef __cplusplus
}
#endif

#endif
