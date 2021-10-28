//===-- M88kRegisterBanks.td - Describe the M88k Banks -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Define the M88k register banks used for GlobalISel.
///
//===----------------------------------------------------------------------===//

/// Register banks.
def GRRegBank : RegisterBank<"GR", [GPRRC]>;
def XRRegBank : RegisterBank<"XR", [XRRC]>;
