//===- Combine.td - Combine rule definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declare GlobalISel combine rules and provide mechanisms to opt-out.
//
//===----------------------------------------------------------------------===//

// Declares a combiner helper class
class GICombinerHelper<string classname> {
  // The class name to use in the generated output.
  string Classname = classname;
}
