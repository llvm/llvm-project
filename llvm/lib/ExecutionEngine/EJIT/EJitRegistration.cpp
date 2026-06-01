//===-- EJitRegistration.cpp - AOT Registration Callbacks -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration functions with dual-path support (manual + constructor).
// Actual implementations live in EJitRuntime.cpp alongside gEJIT.
//
//===----------------------------------------------------------------------===//

// This file intentionally left minimal — the registration callbacks
// (ejit_register_bitcode / ejit_register_period_array /
//  ejit_register_static_var / ejit_register_symbol)
// are implemented in EJitRuntime.cpp with dual-path logic:
//   if (gEJIT) → direct forwarding (manual, post-init)
//   else       → EJitRegistrationStore staging (constructor, pre-init)
