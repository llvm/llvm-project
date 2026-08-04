//===-- WebAssemblyInstrGISel.td - Wasm GISel target pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// WebAssembly GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//


class WebAssemblyGenericInstruction : GenericInstruction {
  let Namespace = "WebAssembly";
}

// non-standard casing to match the same for ISelDAG
def G_Wrapper : WebAssemblyGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = false;
}

// non-standard casing to match the same for ISelDAG
def G_WrapperREL : WebAssemblyGenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src);
  let hasSideEffects = false;
}

def : GINodeEquiv<G_Wrapper, WebAssemblyWrapper>;
def : GINodeEquiv<G_WrapperREL, WebAssemblyWrapperREL>;
