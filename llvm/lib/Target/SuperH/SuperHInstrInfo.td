//===-- SuperHInstrInfo.td - Main SuperH Instruction Definition -*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes the SuperH instruction set, defining the instructions
/// and properties of the instructions which are needed for code generation,
/// machine code emission, and analysis.
///
//===----------------------------------------------------------------------===//

include "SuperHInstrFormats.td"

//===----------------------------------------------------------------------===//
// SuperH Type Profiles
//===----------------------------------------------------------------------===//

