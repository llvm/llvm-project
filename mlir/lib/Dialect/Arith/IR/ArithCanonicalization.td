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

// Subtract two integer attributes and create a new one with the result.
def SubIntAttrs : NativeCodeCall<"subIntegerAttrs($_builder, $0, $1, $2)">;

// Multiply two integer attributes and create a new one with the result.
def MulIntAttrs : NativeCodeCall<"mulIntegerAttrs($_builder, $0, $1, $2)">;

// TODO: Canonicalizations currently doesn't take into account integer overflow
// flags and always reset them to default (wraparound) which is safe but can
// inhibit later optimizations. Individual patterns must be reviewed for
// better handling of overflow flags.
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

class cast<string type> : NativeCodeCall<"::mlir::cast<" # type # ">($0)">;

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

// addi is commutative and will be canonicalized to have its constants appear
// as the second operand.

// addi(addi(x, c0), c1) -> addi(x, c0 + c1)
def AddIAddConstant :
    Pat<(Arith_AddIOp:$res
          (Arith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)),
            DefOverflow)>;

// addi(subi(x, c0), c1) -> addi(x, c1 - c0)
def AddISubConstantRHS :
    Pat<(Arith_AddIOp:$res
          (Arith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)),
            DefOverflow)>;

// addi(subi(c0, x), c1) -> subi(c0 + c1, x)
def AddISubConstantLHS :
    Pat<(Arith_AddIOp:$res
          (Arith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x,
            DefOverflow)>;

def IsScalarOrSplatNegativeOne :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($0))">,
      CPred<"getIntOrSplatIntValue($0)->isAllOnes()">]>>;

// addi(x, muli(y, -1)) -> subi(x, y)
def AddIMulNegativeOneRhs :
    Pat<(Arith_AddIOp
           $x,
           (Arith_MulIOp $y, (ConstantLikeMatcher AnyAttr:$c0), $ovf1), $ovf2),
        (Arith_SubIOp $x, $y, DefOverflow),
        [(IsScalarOrSplatNegativeOne $c0)]>;

// addi(muli(x, -1), y) -> subi(y, x)
def AddIMulNegativeOneLhs :
    Pat<(Arith_AddIOp
           (Arith_MulIOp $x, (ConstantLikeMatcher AnyAttr:$c0), $ovf1),
           $y, $ovf2),
        (Arith_SubIOp $y, $x, DefOverflow),
        [(IsScalarOrSplatNegativeOne $c0)]>;

// muli(muli(x, c0), c1) -> muli(x, c0 * c1)
def MulIMulIConstant :
    Pat<(Arith_MulIOp:$res
          (Arith_MulIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_MulIOp $x, (Arith_ConstantOp (MulIntAttrs $res, $c0, $c1)),
            DefOverflow)>;

//===----------------------------------------------------------------------===//
// AddUIExtendedOp
//===----------------------------------------------------------------------===//

// addui_extended(x, y) -> [addi(x, y), x], when the `overflow` result has no
// uses. Since the 'overflow' result is unused, any replacement value will do.
def AddUIExtendedToAddI:
    Pattern<(Arith_AddUIExtendedOp:$res $x, $y),
             [(Arith_AddIOp $x, $y, DefOverflow), (replaceWithValue $x)],
             [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

// subi(addi(x, c0), c1) -> addi(x, c0 - c1)
def SubIRHSAddConstant :
    Pat<(Arith_SubIOp:$res
          (Arith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c0, $c1)),
            DefOverflow)>;

// subi(c1, addi(x, c0)) -> subi(c1 - c0, x)
def SubILHSAddConstant :
    Pat<(Arith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (Arith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)), $x,
            DefOverflow)>;

// subi(subi(x, c0), c1) -> subi(x, c0 + c1)
def SubIRHSSubConstantRHS :
    Pat<(Arith_SubIOp:$res
          (Arith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_SubIOp $x, (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)),
            DefOverflow)>;

// subi(subi(c0, x), c1) -> subi(c0 - c1, x)
def SubIRHSSubConstantLHS :
    Pat<(Arith_SubIOp:$res
          (Arith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (SubIntAttrs $res, $c0, $c1)), $x,
            DefOverflow)>;

// subi(c1, subi(x, c0)) -> subi(c0 + c1, x)
def SubILHSSubConstantRHS :
    Pat<(Arith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (Arith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x,
            DefOverflow)>;

// subi(c1, subi(c0, x)) -> addi(x, c1 - c0)
def SubILHSSubConstantLHS :
    Pat<(Arith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (Arith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)),
            DefOverflow)>;

// subi(subi(a, b), a) -> subi(0, b)
def SubISubILHSRHSLHS :
    Pat<(Arith_SubIOp:$res (Arith_SubIOp $x, $y, $ovf1), $x, $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (GetZeroAttr $y)), $y, DefOverflow)>;

//===----------------------------------------------------------------------===//
// MulSIExtendedOp
//===----------------------------------------------------------------------===//

// mulsi_extended(x, y) -> [muli(x, y), x], when the `high` result is unused.
// Since the `high` result it not used, any replacement value will do.
def MulSIExtendedToMulI :
    Pattern<(Arith_MulSIExtendedOp:$res $x, $y),
        [(Arith_MulIOp $x, $y, DefOverflow), (replaceWithValue $x)],
        [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;


def IsScalarOrSplatOne :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($0))">,
      CPred<"*getIntOrSplatIntValue($0) == 1">]>>;

// mulsi_extended(x, 1) -> [x, extsi(cmpi slt, x, 0)]
def MulSIExtendedRHSOne :
    Pattern<(Arith_MulSIExtendedOp $x, (ConstantLikeMatcher AnyAttr:$c1)),
            [(replaceWithValue $x),
             (Arith_ExtSIOp(Arith_CmpIOp
                ConstantEnumCase<Arith_CmpIPredicateAttr, "slt">,
                $x,
                (Arith_ConstantOp (GetZeroAttr $x))))],
            [(IsScalarOrSplatOne $c1)]>;

//===----------------------------------------------------------------------===//
// MulUIExtendedOp
//===----------------------------------------------------------------------===//

// mului_extended(x, y) -> [muli(x, y), x], when the `high` result is unused.
// Since the `high` result it not used, any replacement value will do.
def MulUIExtendedToMulI :
    Pattern<(Arith_MulUIExtendedOp:$res $x, $y),
        [(Arith_MulIOp $x, $y, DefOverflow), (replaceWithValue $x)],
        [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;

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
          (ConstantLikeMatcher ConstantAttr<I1Attr, "1">)),
        (Arith_CmpIOp (InvertPredicate $pred), $a, $b)>;

// xor extui(x), extui(y) -> extui(xor(x,y))
def XOrIOfExtUI :
    Pat<(Arith_XOrIOp (Arith_ExtUIOp $x), (Arith_ExtUIOp $y)), (Arith_ExtUIOp (Arith_XOrIOp $x, $y)),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

// xor extsi(x), extsi(y) -> extsi(xor(x,y))
def XOrIOfExtSI :
    Pat<(Arith_XOrIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)), (Arith_ExtSIOp (Arith_XOrIOp $x, $y)),
      [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

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
         (Constraint<
            CPred<"$0.getValue() == arith::CmpIPredicate::eq || "
                  "$0.getValue() == arith::CmpIPredicate::ne">> $pred)]>;

// cmpi(== or !=, a ext iNN, b ext iNN) == cmpi(== or !=, a, b)
def CmpIExtUI :
    Pat<(Arith_CmpIOp $pred,
          (Arith_ExtUIOp $a),
          (Arith_ExtUIOp $b)),
        (Arith_CmpIOp $pred, $a, $b),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $a, $b),
         (Constraint<
            CPred<"$0.getValue() == arith::CmpIPredicate::eq || "
                  "$0.getValue() == arith::CmpIPredicate::ne">> $pred)]>;

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

// select(not(pred), a, b) => select(pred, b, a)
def SelectNotCond :
    Pat<(SelectOp (Arith_XOrIOp $pred, (ConstantLikeMatcher APIntAttr:$ones)), $a, $b),
        (SelectOp $pred, $b, $a),
        [(IsScalarOrSplatNegativeOne $ones)]>;

// select(pred, select(pred, a, b), c) => select(pred, a, c)
def RedundantSelectTrue :
    Pat<(SelectOp $pred, (SelectOp $pred, $a, $b), $c),
        (SelectOp $pred, $a, $c)>;

// select(pred, a, select(pred, b, c)) => select(pred, a, c)
def RedundantSelectFalse :
    Pat<(SelectOp $pred, $a, (SelectOp $pred, $b, $c)),
        (SelectOp $pred, $a, $c)>;

// select(pred, false, true) => not(pred) 
def SelectI1ToNot :
    Pat<(SelectOp $pred,
                  (ConstantLikeMatcher ConstantAttr<I1Attr, "0">),
                  (ConstantLikeMatcher ConstantAttr<I1Attr, "1">)),
        (Arith_XOrIOp $pred, (Arith_ConstantOp ConstantAttr<I1Attr, "1">))>;

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
// IndexCastUIOp
//===----------------------------------------------------------------------===//

// index_castui(index_castui(x)) -> x, if dstType == srcType.
def IndexCastUIOfIndexCastUI :
    Pat<(Arith_IndexCastUIOp:$res (Arith_IndexCastUIOp $x)),
        (replaceWithValue $x),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $res, $x)]>;

// index_castui(extui(x)) -> index_castui(x)
def IndexCastUIOfExtUI :
    Pat<(Arith_IndexCastUIOp (Arith_ExtUIOp $x)), (Arith_IndexCastUIOp $x)>;


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
    Pat<(Arith_AndIOp (Arith_ExtUIOp $x), (Arith_ExtUIOp $y)),
        (Arith_ExtUIOp (Arith_AndIOp $x, $y)),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

// and extsi(x), extsi(y) -> extsi(and(x,y))
def AndOfExtSI :
    Pat<(Arith_AndIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)),
        (Arith_ExtSIOp (Arith_AndIOp $x, $y)),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

// or extui(x), extui(y) -> extui(or(x,y))
def OrOfExtUI :
    Pat<(Arith_OrIOp (Arith_ExtUIOp $x), (Arith_ExtUIOp $y)),
        (Arith_ExtUIOp (Arith_OrIOp $x, $y)),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

// or extsi(x), extsi(y) -> extsi(or(x,y))
def OrOfExtSI :
    Pat<(Arith_OrIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)),
        (Arith_ExtSIOp (Arith_OrIOp $x, $y)),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

def ValuesWithSameType :
    Constraint<
      CPred<"llvm::all_equal({$0.getType(), $1.getType(), $2.getType()})">>;

def ValueWiderThan :
    Constraint<And<[
      CPred<"getScalarOrElementWidth($0) > getScalarOrElementWidth($1)">,
      CPred<"getScalarOrElementWidth($1) > 0">]>>;

def TruncationMatchesShiftAmount :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($2))">,
      CPred<"(getScalarOrElementWidth($0) - getScalarOrElementWidth($1)) == "
              "*getIntOrSplatIntValue($2)">]>>;

// trunci(extsi(x)) -> extsi(x), when only the sign-extension bits are truncated
def TruncIExtSIToExtSI :
    Pat<(Arith_TruncIOp:$tr (Arith_ExtSIOp:$ext $x)),
        (Arith_ExtSIOp $x),
        [(ValueWiderThan $ext, $tr),
         (ValueWiderThan $tr, $x)]>;

// trunci(extui(x)) -> extui(x), when only the zero-extension bits are truncated
def TruncIExtUIToExtUI :
    Pat<(Arith_TruncIOp:$tr (Arith_ExtUIOp:$ext $x)),
        (Arith_ExtUIOp $x),
        [(ValueWiderThan $ext, $tr),
         (ValueWiderThan $tr, $x)]>;

// trunci(shrsi(x, c)) -> trunci(shrui(x, c))
def TruncIShrSIToTrunciShrUI :
    Pat<(Arith_TruncIOp:$tr
          (Arith_ShRSIOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0))),
        (Arith_TruncIOp (Arith_ShRUIOp $x, (Arith_ConstantOp (cast<"TypedAttr"> $c0)))),
        [(TruncationMatchesShiftAmount $x, $tr, $c0)]>;

// trunci(shrui(mul(sext(x), sext(y)), c)) -> mulsi_extended(x, y)
def TruncIShrUIMulIToMulSIExtended :
    Pat<(Arith_TruncIOp:$tr (Arith_ShRUIOp
                              (Arith_MulIOp:$mul
                                (Arith_ExtSIOp $x), (Arith_ExtSIOp $y), $ovf1),
                              (ConstantLikeMatcher AnyAttr:$c0))),
        (Arith_MulSIExtendedOp:$res__1 $x, $y),
      [(ValuesWithSameType $tr, $x, $y),
       (ValueWiderThan $mul, $x),
       (TruncationMatchesShiftAmount $mul, $x, $c0)]>;

// trunci(shrui(mul(zext(x), zext(y)), c)) -> mului_extended(x, y)
def TruncIShrUIMulIToMulUIExtended :
    Pat<(Arith_TruncIOp:$tr (Arith_ShRUIOp
                              (Arith_MulIOp:$mul
                                (Arith_ExtUIOp $x), (Arith_ExtUIOp $y), $ovf1),
                              (ConstantLikeMatcher AnyAttr:$c0))),
        (Arith_MulUIExtendedOp:$res__1 $x, $y),
      [(ValuesWithSameType $tr, $x, $y),
       (ValueWiderThan $mul, $x),
       (TruncationMatchesShiftAmount $mul, $x, $c0)]>;

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

// mulf(negf(x), negf(y)) -> mulf(x,y)
// (retain fastmath flags of original mulf)
def MulFOfNegF :
    Pat<(Arith_MulFOp (Arith_NegFOp $x, $_), (Arith_NegFOp $y, $_), $fmf),
        (Arith_MulFOp $x, $y, $fmf),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

// divf(negf(x), negf(y)) -> divf(x,y)
// (retain fastmath flags of original divf)
def DivFOfNegF :
    Pat<(Arith_DivFOp (Arith_NegFOp $x, $_), (Arith_NegFOp $y, $_), $fmf),
        (Arith_DivFOp $x, $y, $fmf),
        [(Constraint<CPred<"$0.getType() == $1.getType()">> $x, $y)]>;

#endif // ARITH_PATTERNS
