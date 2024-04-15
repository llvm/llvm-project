//===-- GENAttrDefs.td - GEN dialect attributes def. file --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GEN_ATTRDEFS
#define GEN_ATTRDEFS

include "mlir/IR/EnumAttr.td"

/// Enum attribute of the different shuffle kinds.
/// Based on https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_non_uniform_instructions
def GEN_ShflKindAttr : I32EnumAttr<"ShflKind", "GEN shuffle kind",
  [
    I32EnumAttrCase<"XOR",  0, "xor">,
    I32EnumAttrCase<"UP",   1, "up">,
    I32EnumAttrCase<"DOWN", 2, "down">,
    I32EnumAttrCase<"IDX",  3, "idx">
  ]> {
  let cppNamespace = "::mlir::GEN";
}

#endif // GEN_ATTRDEFS
