//===-- OffloadAPI.td - Root tablegen file for Offload -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Always include this file first
include "APIDefs.td"

// Add API definition files here
include "Common.td"
include "Platform.td"
include "Device.td"
