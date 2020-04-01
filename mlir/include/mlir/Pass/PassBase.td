//===-- PassBase.td - Base pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for defining pass registration and other
// mechanisms.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PASS_PASSBASE
#define MLIR_PASS_PASSBASE

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class Pass<string passArg> {
  // The command line argument of the pass.
  string argument = passArg;

  // A short 1-line summary of the pass.
  string summary = "";

  // A human readable description of the pass.
  string description = "";

  // A C++ constructor call to create an instance of this pass.
  code constructor = [{}];
}

#endif // MLIR_PASS_PASSBASE
