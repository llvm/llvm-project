//===-- RISCVInstrInfoXSpacemiT.td -------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by SpacemiT.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsSpacemitV.td"

let DecoderNamespace = "XSMT" in {
  let Predicates = [HasVendorXSMTVDot, IsRV64], ElementsDependOn = EltDepsNone,
      RVVConstraint = WidenVNoMask, DestEEW = EEWSEWx4 in {
    // Base vector dot product (no slide) instructions
    // NOTE: Destination registers (vd) MUST be even-numbered (v0, v2, ..., v30)
    //       due to the number of elements processed requires a pair of register operations.
    // Using odd registers may cause undefined behavior.

    foreach a = VTypeEncodes in
      def SMT_VMADOT#!toupper(a.Name) :
            SMTVMADot<a.Encoding, "smt.vmadot"#a.Name>, Sched<[]>;

    //===----------------------------------------------------------------------===//
    // Sliding-window Vector Dot Product Instructions
    //
    // The numeric suffix (1, 2, 3) specifies the stride of the sliding window:
    //   1: Window slides by 1 element per operation
    //   2: Window slides by 2 elements per operation
    //   3: Window slides by 3 elements per operation
    //
    // These instructions compute dot products with overlapping operand windows
    // where the window position increments by <N> elements between
    // computations.
    //===----------------------------------------------------------------------===//
    // NOTE: Destination registers (vd) and first source register (vs1) MUST be
    //       even-numbered (v0, v2, ..., v30) due to the number of elements processed
    //       requires a pair of register operations.
    //       Using odd registers may cause undefined behavior.

    foreach a = SlideEncodes in
      foreach b = VTypeEncodes in
        def SMT_VMADOT#!toupper(a.Name)#!toupper(b.Name) :
            SMTVMADotSlide<a.Encoding, b.Encoding, "smt.vmadot"#a.Name#b.Name>, Sched<[]>;
  }
}
