//===- IndexEnums.td - Index enum definitions --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEX_ENUMS
#define INDEX_ENUMS

include "mlir/Dialect/Index/IR/IndexDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// IndexCmpPredicate
//===----------------------------------------------------------------------===//

def IndexCmpPredicate : I32EnumAttr<
    "IndexCmpPredicate", "index comparison predicate kind",
    [
      I32EnumAttrCase<"EQ",  0, "eq">,
      I32EnumAttrCase<"NE",  1, "ne">,
      I32EnumAttrCase<"SLT", 2, "slt">,
      I32EnumAttrCase<"SLE", 3, "sle">,
      I32EnumAttrCase<"SGT", 4, "sgt">,
      I32EnumAttrCase<"SGE", 5, "sge">,
      I32EnumAttrCase<"ULT", 6, "ult">,
      I32EnumAttrCase<"ULE", 7, "ule">,
      I32EnumAttrCase<"UGT", 8, "ugt">,
      I32EnumAttrCase<"UGE", 9, "uge">
    ]> {
  let cppNamespace = "::mlir::index";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// IndexCmpPredicateAttr
//===----------------------------------------------------------------------===//

def IndexCmpPredicateAttr : EnumAttr<
    IndexDialect, IndexCmpPredicate, "cmp_predicate">;

#endif // INDEX_ENUMS
