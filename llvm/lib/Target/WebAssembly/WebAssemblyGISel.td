//===-- WebAssemblyGISel.td - Wasm GlobalISel Patterns -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file contains patterns that are relevant to GlobalISel, including
/// GIComplexOperandMatcher definitions for equivalent SelectionDAG
/// ComplexPatterns.
//
//===----------------------------------------------------------------------===//

include "WebAssembly.td"
include "WebAssemblyCombine.td"

defvar ModeAddr32 = DefaultMode;
def ModeAddr64 : HwMode<[HasAddr64]>;

def p0i : ValueTypeByHwMode<[ModeAddr32, ModeAddr64],
                            [i32, i64]>;
def p0 : PtrValueTypeByHwMode<p0i, 0>;


def : Pat<(p0 (ptradd p0:$lhs, p0i:$rhs)),
          (ADD_I32 I32:$lhs, I32:$rhs)>, Requires<[HasAddr32]>;
def : Pat<(p0 (ptradd p0:$lhs, p0i:$rhs)),
          (ADD_I64 I64:$lhs, I64:$rhs)>, Requires<[HasAddr64]>;


def to_tframeindex : SDNodeXForm<frameindex, [{
  return CurDAG->getTargetFrameIndex(N->getIndex(), N->getValueType(0));
}]>;

def : GICustomOperandRenderer<"renderFrameIndex">,
      GISDNodeXFormEquiv<to_tframeindex>;

def : Pat<(frameindex:$fi),
          (COPY_I32 (to_tframeindex $fi))>, Requires<[HasAddr32]>;
def : Pat<(frameindex:$fi),
          (COPY_I64 (to_tframeindex $fi))>, Requires<[HasAddr64]>;


//===----------------------------------------------------------------------===//
// Complex pattern equivalents
//===----------------------------------------------------------------------===//

def gi_AddrOps32 : GIComplexOperandMatcher<s32, "selectAddrOperands32">,
                   GIComplexPatternEquiv<AddrOps32>;

def gi_AddrOps64 : GIComplexOperandMatcher<s64, "selectAddrOperands64">,
                   GIComplexPatternEquiv<AddrOps64>;
