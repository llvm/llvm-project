//===- ArithPatterns.td - Arith dialect patterns -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_PATTERNS
#define ARITH_PATTERNS

include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

// Create zero attribute of type matching the argument's type.
def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr($0.getType())">;

// Add two integer attributes and create a new one with the result.
def AddIntAttrs : NativeCodeCall<"addIntegerAttrs($_builder, $0, $1, $2)">;

// Subtract two integer attributes and createa a new one with the result.
def SubIntAttrs : NativeCodeCall<"subIntegerAttrs($_builder, $0, $1, $2)">;

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

// addi is commutative and will be canonicalized to have its constants appear
// as the second operand.

// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
def AddIAddConstant :
    Pat<(Arith_AddIOp:$res
          (Arith_AddIOp $x, (Arith_ConstantOp APIntAttr:$c0)),
          (Arith_ConstantOp APIntAttr:$c1)),
        (Arith_AddIOp $x, (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)))>;

// addi(subi(x, c0), c1) -> addi(x, c1 - c0)
def AddISubConstantRHS :
    Pat<(Arith_AddIOp:$res
          (Arith_SubIOp $x, (Arith_ConstantOp APIntAttr:$c0)),
          (Arith_ConstantOp APIntAttr:$c1)),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)))>;

// addi(subi(c0, x), c1) -> subi(c0 + c1, x)
def AddISubConstantLHS :
    Pat<(Arith_AddIOp:$res
          (Arith_SubIOp (Arith_ConstantOp APIntAttr:$c0), $x),
          (Arith_ConstantOp APIntAttr:$c1)),
        (Arith_SubIOp (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x)>;

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

// subi(addi(x, c0), c1) -> addi(x, c0 - c1)
def SubIRHSAddConstant :
    Pat<(Arith_SubIOp:$res
          (Arith_AddIOp $x, (Arith_ConstantOp APIntAttr:$c0)),
          (Arith_ConstantOp APIntAttr:$c1)),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c0, $c1)))>;

// subi(c1, addi(x, c0)) -> subi(c1 - c0, x)
def SubILHSAddConstant :
    Pat<(Arith_SubIOp:$res
          (Arith_ConstantOp APIntAttr:$c1),
          (Arith_AddIOp $x, (Arith_ConstantOp APIntAttr:$c0))),
        (Arith_SubIOp (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)), $x)>;

// subi(subi(x, c0), c1) -> subi(x, c0 + c1)
def SubIRHSSubConstantRHS :
    Pat<(Arith_SubIOp:$res
          (Arith_SubIOp $x, (Arith_ConstantOp APIntAttr:$c0)),
          (Arith_ConstantOp APIntAttr:$c1)),
        (Arith_SubIOp $x, (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)))>;

// subi(subi(c0, x), c1) -> subi(c0 - c1, x)
def SubIRHSSubConstantLHS :
    Pat<(Arith_SubIOp:$res
          (Arith_SubIOp (Arith_ConstantOp APIntAttr:$c0), $x),
          (Arith_ConstantOp APIntAttr:$c1)),
        (Arith_SubIOp (Arith_ConstantOp (SubIntAttrs $res, $c0, $c1)), $x)>;

// subi(c1, subi(x, c0)) -> subi(c0 + c1, x)
def SubILHSSubConstantRHS :
    Pat<(Arith_SubIOp:$res
          (Arith_ConstantOp APIntAttr:$c1),
          (Arith_SubIOp $x, (Arith_ConstantOp APIntAttr:$c0))),
        (Arith_SubIOp (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x)>;

// subi(c1, subi(c0, x)) -> addi(x, c1 - c0)
def SubILHSSubConstantLHS :
    Pat<(Arith_SubIOp:$res
          (Arith_ConstantOp APIntAttr:$c1),
          (Arith_SubIOp (Arith_ConstantOp APIntAttr:$c0), $x)),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)))>;

// subi(subi(a, b), a) -> subi(0, b)
def SubISubILHSRHSLHS :
    Pat<(Arith_SubIOp:$res (Arith_SubIOp $x, $y), $x),
        (Arith_SubIOp (Arith_ConstantOp (GetZeroAttr $y)), $y)>;

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

// xori is commutative and will be canonicalized to have its constants appear
// as the second operand.

// not(cmpi(pred, a, b)) -> cmpi(~pred, a, b), where not(x) is xori(x, 1)
def InvertPredicate : NativeCodeCall<"invertPredicate($0)">;
def XOrINotCmpI :
    Pat<(Arith_XOrIOp
          (Arith_CmpIOp $pred, $a, $b),
          (Arith_ConstantOp ConstantAttr<I1Attr, "1">)),
        (Arith_CmpIOp (InvertPredicate $pred), $a, $b)>;

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

// cmpi(== or !=, a ext iNN, b ext iNN) == cmpi(== or !=, a, b)
def CmpIExtSI :
    Pat<(Arith_CmpIOp $pred,
          (Arith_ExtSIOp $a),
          (Arith_ExtSIOp $b)),
        (Arith_CmpIOp $pred, $a, $b),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $a, $b),
         (Constraint<CPred<"$0.getValue() == arith::CmpIPredicate::eq || $0.getValue() == arith::CmpIPredicate::ne">> $pred)]>;

// cmpi(== or !=, a ext iNN, b ext iNN) == cmpi(== or !=, a, b)
def CmpIExtUI :
    Pat<(Arith_CmpIOp $pred,
          (Arith_ExtUIOp $a),
          (Arith_ExtUIOp $b)),
        (Arith_CmpIOp $pred, $a, $b),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $a, $b),
         (Constraint<CPred<"$0.getValue() == arith::CmpIPredicate::eq || $0.getValue() == arith::CmpIPredicate::ne">> $pred)]>;

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// index_cast(index_cast(x)) -> x, if dstType == srcType.
def IndexCastOfIndexCast :
    Pat<(Arith_IndexCastOp:$res (Arith_IndexCastOp $x)),
        (replaceWithValue $x),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $res, $x)]>;

// index_cast(extsi(x)) -> index_cast(x)
def IndexCastOfExtSI :
    Pat<(Arith_IndexCastOp (Arith_ExtSIOp $x)), (Arith_IndexCastOp $x)>;

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

// bitcast(bitcast(x)) -> x
def BitcastOfBitcast :
    Pat<(Arith_BitcastOp (Arith_BitcastOp $x)), (replaceWithValue $x)>;

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

// extsi(extui(x iN : iM) : iL) -> extui(x : iL)
def ExtSIOfExtUI :
    Pat<(Arith_ExtSIOp (Arith_ExtUIOp $x)), (Arith_ExtUIOp $x)>;

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

// and extui(x), extui(y) -> extui(and(x,y))
def AndOfExtUI :
    Pat<(Arith_AndIOp (Arith_ExtUIOp $x), (Arith_ExtUIOp $y)), (Arith_ExtUIOp (Arith_AndIOp $x, $y)),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

// and extsi(x), extsi(y) -> extsi(and(x,y))
def AndOfExtSI :
    Pat<(Arith_AndIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)), (Arith_ExtSIOp (Arith_AndIOp $x, $y)),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

// or extui(x), extui(y) -> extui(or(x,y))
def OrOfExtUI :
    Pat<(Arith_OrIOp (Arith_ExtUIOp $x), (Arith_ExtUIOp $y)), (Arith_ExtUIOp (Arith_OrIOp $x, $y)),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

// or extsi(x), extsi(y) -> extsi(or(x,y))
def OrOfExtSI :
    Pat<(Arith_OrIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)), (Arith_ExtSIOp (Arith_OrIOp $x, $y)),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

// mulf(negf(x), negf(y)) -> mulf(x,y)
def MulFOfNegF :
    Pat<(Arith_MulFOp (Arith_NegFOp $x), (Arith_NegFOp $y)), (Arith_MulFOp $x, $y),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

// divf(negf(x), negf(y)) -> divf(x,y)
def DivFOfNegF :
    Pat<(Arith_DivFOp (Arith_NegFOp $x), (Arith_NegFOp $y)), (Arith_DivFOp $x, $y),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

#endif // ARITH_PATTERNS
