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

// Bitwise-and two integer attributes and create a new one with the result.
def AndIntAttrs : NativeCodeCall<"andIntegerAttrs($_builder, $0, $1, $2)">;

// Bitwise-or two integer attributes and create a new one with the result.
def OrIntAttrs : NativeCodeCall<"orIntegerAttrs($_builder, $0, $1, $2)">;

// Bitwise-xor two integer attributes and create a new one with the result.
def XorIntAttrs : NativeCodeCall<"xorIntegerAttrs($_builder, $0, $1, $2)">;

// Merge overflow flags from 2 ops, selecting the most conservative combination.
def MergeOverflow : NativeCodeCall<"mergeOverflowFlags($0, $1)">;

// Default overflow flag (all wraparounds allowed).
defvar DefOverflow = ConstantEnumCase<Arith_IntegerOverflowAttr, "none">;

class cast<string type> : NativeCodeCall<"::mlir::cast<" # type # ">($0)">;

// Check that two values have the same type.
def TypesMatch : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Check that an IntegerAttr's type matches the result value's type.
def IntAttrTypeMatchesResult :
    Constraint<CPred<"$0.getType() == cast<IntegerAttr>($1).getType()">>;

// Check that a CmpIPredicateAttr is eq or ne.
def IsEqualityCmpIPredicate :
    Constraint<CPred<"$0.getValue() == arith::CmpIPredicate::eq || "
                     "$0.getValue() == arith::CmpIPredicate::ne">>;

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
            (MergeOverflow $ovf1, $ovf2))>;

// addi(subi(x, c0), c1) -> addi(x, c1 - c0)
def AddISubConstantRHS :
    Pat<(Arith_AddIOp:$res
          (Arith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)),
            (MergeOverflow $ovf1, $ovf2))>;

// addi(subi(c0, x), c1) -> subi(c0 + c1, x)
def AddISubConstantLHS :
    Pat<(Arith_AddIOp:$res
          (Arith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

def IsScalarOrSplatNegativeOne :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($0))">,
      CPred<"getIntOrSplatIntValue($0)->isAllOnes()">]>>;

// addi(x, muli(y, -1)) -> subi(x, y)
def AddIMulNegativeOneRhs :
    Pat<(Arith_AddIOp
           $x,
           (Arith_MulIOp $y, (ConstantLikeMatcher AnyAttr:$c0), $ovf1), $ovf2),
        (Arith_SubIOp $x, $y, DefOverflow), // TODO: overflow flags
        [(IsScalarOrSplatNegativeOne $c0)]>;

// addi(muli(x, -1), y) -> subi(y, x)
def AddIMulNegativeOneLhs :
    Pat<(Arith_AddIOp
           (Arith_MulIOp $x, (ConstantLikeMatcher AnyAttr:$c0), $ovf1),
           $y, $ovf2),
        (Arith_SubIOp $y, $x, DefOverflow), // TODO: overflow flags
        [(IsScalarOrSplatNegativeOne $c0)]>;

// muli(muli(x, c0), c1) -> muli(x, c0 * c1)
def MulIMulIConstant :
    Pat<(Arith_MulIOp:$res
          (Arith_MulIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_MulIOp $x, (Arith_ConstantOp (MulIntAttrs $res, $c0, $c1)),
            (MergeOverflow $ovf1, $ovf2)),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

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
// SubUIExtendedOp
//===----------------------------------------------------------------------===//

// subui_extended(x, y) -> [subi(x, y), x], when the `borrow` result has no
// uses. Since the 'borrow' result is unused, any replacement value will do.
def SubUIExtendedToSubI:
    Pattern<(Arith_SubUIExtendedOp:$res $x, $y),
             [(Arith_SubIOp $x, $y, DefOverflow), (replaceWithValue $x)],
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
            DefOverflow)>; // TODO: overflow flags

// subi(c1, addi(x, c0)) -> subi(c1 - c0, x)
def SubILHSAddConstant :
    Pat<(Arith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (Arith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

// subi(subi(x, c0), c1) -> subi(x, c0 + c1)
def SubIRHSSubConstantRHS :
    Pat<(Arith_SubIOp:$res
          (Arith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_SubIOp $x, (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)),
            (MergeOverflow $ovf1, $ovf2))>;

// subi(subi(c0, x), c1) -> subi(c0 - c1, x)
def SubIRHSSubConstantLHS :
    Pat<(Arith_SubIOp:$res
          (Arith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (SubIntAttrs $res, $c0, $c1)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

// subi(c1, subi(x, c0)) -> subi(c0 + c1, x)
def SubILHSSubConstantRHS :
    Pat<(Arith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (Arith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1), $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

// subi(c1, subi(c0, x)) -> addi(x, c1 - c0)
def SubILHSSubConstantLHS :
    Pat<(Arith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (Arith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1), $ovf2),
        (Arith_AddIOp $x, (Arith_ConstantOp (SubIntAttrs $res, $c1, $c0)),
            (MergeOverflow $ovf1, $ovf2))>;

// subi(subi(a, b), a) -> subi(0, b)
def SubISubILHSRHSLHS :
    Pat<(Arith_SubIOp:$res (Arith_SubIOp $x, $y, $ovf1), $x, $ovf2),
        (Arith_SubIOp (Arith_ConstantOp (GetZeroAttr $y)), $y,
            (MergeOverflow $ovf1, $ovf2))>;

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
      CPred<"getIntOrSplatIntValue($0)->isStrictlyPositive()">,
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

// xori(xori(x, c0), c1) -> xori(x, c0 ^ c1)
def XOrIXOrIConstant :
    Pat<(Arith_XOrIOp:$res
          (Arith_XOrIOp $x, (ConstantLikeMatcher APIntAttr:$c0)),
          (ConstantLikeMatcher APIntAttr:$c1)),
        (Arith_XOrIOp $x, (Arith_ConstantOp (XorIntAttrs $res, $c0, $c1))),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

// not(cmpi(pred, a, b)) -> cmpi(~pred, a, b), where not(x) is xori(x, 1)
def InvertPredicate : NativeCodeCall<"invertPredicate($0)">;
def XOrINotCmpI :
    Pat<(Arith_XOrIOp
          (Arith_CmpIOp $pred, $a, $b),
          (ConstantLikeMatcher ConstantAttr<I1Attr, "1">)),
        (Arith_CmpIOp (InvertPredicate $pred), $a, $b)>;

// xor extui(x), extui(y) -> extui(xor(x,y))
// XOR preserves nneg only if both operands are non-negative: xor of two values
// with sign bit 0 has sign bit 0.
def XOrIOfExtUI :
    Pat<(Arith_XOrIOp (Arith_ExtUIOp $x, $nneg1), (Arith_ExtUIOp $y, $nneg2)),
        (Arith_ExtUIOp (Arith_XOrIOp $x, $y),
                       (NativeCodeCall<"($0 && $1) ? $0 : UnitAttr()"> $nneg1, $nneg2)),
      [(TypesMatch $x, $y)]>;

// xor extsi(x), extsi(y) -> extsi(xor(x,y))
def XOrIOfExtSI :
    Pat<(Arith_XOrIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)), (Arith_ExtSIOp (Arith_XOrIOp $x, $y)),
      [(TypesMatch $x, $y)]>;

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

// cmpi(== or !=, a ext iNN, b ext iNN) == cmpi(== or !=, a, b)
def CmpIExtSI :
    Pat<(Arith_CmpIOp $pred,
          (Arith_ExtSIOp $a),
          (Arith_ExtSIOp $b)),
        (Arith_CmpIOp $pred, $a, $b),
        [(TypesMatch $a, $b), (IsEqualityCmpIPredicate $pred)]>;

// cmpi(== or !=, a ext iNN, b ext iNN) == cmpi(== or !=, a, b)
def CmpIExtUI :
    Pat<(Arith_CmpIOp $pred,
          (Arith_ExtUIOp $a, $nneg1),
          (Arith_ExtUIOp $b, $nneg2)),
        (Arith_CmpIOp $pred, $a, $b),
        [(TypesMatch $a, $b), (IsEqualityCmpIPredicate $pred)]>;

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

class SelectCmpIToMinMaxBase<string pred, Op minMaxOp> :
    Pat<(SelectOp
          (Arith_CmpIOp
            ConstantEnumCase<Arith_CmpIPredicateAttr, pred>,
            $a, $b),
          $a, $b),
        (minMaxOp $a, $b)>;

class SelectCmpIInverseToMinMaxBase<string pred, Op minMaxOp> :
    Pat<(SelectOp
          (Arith_CmpIOp
            ConstantEnumCase<Arith_CmpIPredicateAttr, pred>,
            $a, $b),
          $b, $a),
        (minMaxOp $a, $b)>;

// select(cmpi pred, a, b), a, b => min/max(a, b)
def SelectCmpISltToMinSI : SelectCmpIToMinMaxBase<"slt", Arith_MinSIOp>;
def SelectCmpISleToMinSI : SelectCmpIToMinMaxBase<"sle", Arith_MinSIOp>;
def SelectCmpISgtToMaxSI : SelectCmpIToMinMaxBase<"sgt", Arith_MaxSIOp>;
def SelectCmpISgeToMaxSI : SelectCmpIToMinMaxBase<"sge", Arith_MaxSIOp>;
def SelectCmpIUltToMinUI : SelectCmpIToMinMaxBase<"ult", Arith_MinUIOp>;
def SelectCmpIUleToMinUI : SelectCmpIToMinMaxBase<"ule", Arith_MinUIOp>;
def SelectCmpIUgtToMaxUI : SelectCmpIToMinMaxBase<"ugt", Arith_MaxUIOp>;
def SelectCmpIUgeToMaxUI : SelectCmpIToMinMaxBase<"uge", Arith_MaxUIOp>;

// select(cmpi pred, a, b), b, a => inverse min/max(a, b)
def SelectCmpISltToMaxSI : SelectCmpIInverseToMinMaxBase<"slt", Arith_MaxSIOp>;
def SelectCmpISleToMaxSI : SelectCmpIInverseToMinMaxBase<"sle", Arith_MaxSIOp>;
def SelectCmpISgtToMinSI : SelectCmpIInverseToMinMaxBase<"sgt", Arith_MinSIOp>;
def SelectCmpISgeToMinSI : SelectCmpIInverseToMinMaxBase<"sge", Arith_MinSIOp>;
def SelectCmpIUltToMaxUI : SelectCmpIInverseToMinMaxBase<"ult", Arith_MaxUIOp>;
def SelectCmpIUleToMaxUI : SelectCmpIInverseToMinMaxBase<"ule", Arith_MaxUIOp>;
def SelectCmpIUgtToMinUI : SelectCmpIInverseToMinMaxBase<"ugt", Arith_MinUIOp>;
def SelectCmpIUgeToMinUI : SelectCmpIInverseToMinMaxBase<"uge", Arith_MinUIOp>;

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// index_cast(extsi(x)) -> index_cast(x)
def IndexCastOfExtSI :
    Pat<(Arith_IndexCastOp (Arith_ExtSIOp $x)), (Arith_IndexCastOp $x)>;

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

// index_castui(extui(x)) -> index_castui(x)
def IndexCastUIOfExtUI :
    Pat<(Arith_IndexCastUIOp (Arith_ExtUIOp $x, $nneg1), $nneg2),
        (Arith_IndexCastUIOp $x, $nneg1)>;


//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

// bitcast(type1, bitcast(type2, x)) -> bitcast(type1, x)
def BitcastOfBitcast :
    Pat<(Arith_BitcastOp (Arith_BitcastOp $x)), (Arith_BitcastOp $x)>;

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

// extsi(extui(x iN : iM) : iL) -> extui(x : iL)
def ExtSIOfExtUI :
    Pat<(Arith_ExtSIOp (Arith_ExtUIOp $x, $nneg)),
        (Arith_ExtUIOp $x, $nneg)>;

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

// andi(andi(x, c0), c1) -> andi(x, c0 & c1)
def AndIAndIConstant :
    Pat<(Arith_AndIOp:$res
          (Arith_AndIOp $x, (ConstantLikeMatcher APIntAttr:$c0)),
          (ConstantLikeMatcher APIntAttr:$c1)),
        (Arith_AndIOp $x, (Arith_ConstantOp (AndIntAttrs $res, $c0, $c1))),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

// and extui(x), extui(y) -> extui(and(x,y))
// AND can only clear bits, so if either operand is non-negative (sign bit 0),
// the result also has sign bit 0. Preserves nneg if either operand has it.
def AndOfExtUI :
    Pat<(Arith_AndIOp (Arith_ExtUIOp $x, $nneg1), (Arith_ExtUIOp $y, $nneg2)),
        (Arith_ExtUIOp (Arith_AndIOp $x, $y),
                       (NativeCodeCall<"$0 ? $0 : $1"> $nneg1, $nneg2)),
        [(TypesMatch $x, $y)]>;

// and extsi(x), extsi(y) -> extsi(and(x,y))
def AndOfExtSI :
    Pat<(Arith_AndIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)),
        (Arith_ExtSIOp (Arith_AndIOp $x, $y)),
        [(TypesMatch $x, $y)]>;

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

// ori(ori(x, c0), c1) -> ori(x, c0 | c1)
def OrIOrIConstant :
    Pat<(Arith_OrIOp:$res
          (Arith_OrIOp $x, (ConstantLikeMatcher APIntAttr:$c0)),
          (ConstantLikeMatcher APIntAttr:$c1)),
        (Arith_OrIOp $x, (Arith_ConstantOp (OrIntAttrs $res, $c0, $c1))),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

// or extui(x), extui(y) -> extui(or(x,y))
// OR preserves nneg only if both operands are non-negative: or of two values
// with sign bit 0 has sign bit 0.
def OrOfExtUI :
    Pat<(Arith_OrIOp (Arith_ExtUIOp $x, $nneg1), (Arith_ExtUIOp $y, $nneg2)),
        (Arith_ExtUIOp (Arith_OrIOp $x, $y),
                       (NativeCodeCall<"($0 && $1) ? $0 : UnitAttr()"> $nneg1, $nneg2)),
        [(TypesMatch $x, $y)]>;

// or extsi(x), extsi(y) -> extsi(or(x,y))
def OrOfExtSI :
    Pat<(Arith_OrIOp (Arith_ExtSIOp $x), (Arith_ExtSIOp $y)),
        (Arith_ExtSIOp (Arith_OrIOp $x, $y)),
        [(TypesMatch $x, $y)]>;

//===----------------------------------------------------------------------===//
// TruncIOp
//===----------------------------------------------------------------------===//

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
    Pat<(Arith_TruncIOp:$tr (Arith_ExtSIOp:$ext $x), $overflow),
        (Arith_ExtSIOp $x),
        [(ValueWiderThan $ext, $tr),
         (ValueWiderThan $tr, $x)]>;

// trunci(extui(x)) -> extui(x), when only the zero-extension bits are truncated
def TruncIExtUIToExtUI :
    Pat<(Arith_TruncIOp:$tr (Arith_ExtUIOp:$ext $x, $nneg), $overflow),
        (Arith_ExtUIOp $x, $nneg),
        [(ValueWiderThan $ext, $tr),
         (ValueWiderThan $tr, $x)]>;

// trunci(shrsi(x, c)) -> trunci(shrui(x, c))
def TruncIShrSIToTrunciShrUI :
    Pat<(Arith_TruncIOp:$tr
          (Arith_ShRSIOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0), $exact), $overflow),
        (Arith_TruncIOp (Arith_ShRUIOp $x, (Arith_ConstantOp (cast<"TypedAttr"> $c0)), $exact), $overflow),
        [(TruncationMatchesShiftAmount $x, $tr, $c0)]>;

//===----------------------------------------------------------------------===//
// TruncFOp
//===----------------------------------------------------------------------===//

// truncf(sitofp(x)) -> sitofp(x) if default rounding mode.
def TruncFSIToFPToSIToFP :
    Pat<(Arith_TruncFOp:$tr (Arith_SIToFPOp:$fp $x), $rmf, $fmf),
        (Arith_SIToFPOp $x),
        [(Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rmf)]>;

// truncf(uitofp(x)) -> uitofp(x) if default rounding mode.
def TruncFUIToFPToUIToFP :
    Pat<(Arith_TruncFOp:$tr (Arith_UIToFPOp:$fp $x, $nneg), $rmf, $fmf),
        (Arith_UIToFPOp $x, $nneg),
        [(Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rmf)]>;

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

// sitofp(extsi(x)) -> sitofp(x)
def SIToFPOfExtSI :
    Pat<(Arith_SIToFPOp (Arith_ExtSIOp $x)),
        (Arith_SIToFPOp $x)>;

// sitofp(extui(x)) -> uitofp(x)
def SIToFPOfExtUI :
    Pat<(Arith_SIToFPOp (Arith_ExtUIOp $x, $nneg)),
        (Arith_UIToFPOp $x, $nneg)>;

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

// uitofp(extui(x)) -> uitofp(x)
def UIToFPOfExtUI :
    Pat<(Arith_UIToFPOp (Arith_ExtUIOp $x, $nneg1), $nneg2),
        (Arith_UIToFPOp $x, $nneg1)>;

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

// addf(negf(x), y) -> subf(y, x)
// addf(x, negf(y)) -> subf(x, y)
//
// Valid for any rounding mode, which is propagated to the subf: negf is an
// exact sign flip, IEEE addition is commutative, and y - x is the correctly
// rounded y + (-x), so subf(y, x) and addf(negf(x), y) are bit-identical.
def AddFOfNegFLhs :
    Pat<(Arith_AddFOp (Arith_NegFOp $x, $_), $y, $fmf, $rm),
        (Arith_SubFOp $y, $x, $fmf, $rm)>;

def AddFOfNegFRhs :
    Pat<(Arith_AddFOp $x, (Arith_NegFOp $y, $_), $fmf, $rm),
        (Arith_SubFOp $x, $y, $fmf, $rm)>;

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

// Match a constant (scalar or splat) -0.0 float.
def IsNegZeroFloat :
    Constraint<CPred<"m_NegZeroFloat().match($0)">,
               "is negative-zero float constant">;

// subf(-0, x) -> negf(x)
// TODO: Verify if this canonicalization is safe when a rounding mode is
// specified. For the moment, bail on custom rounding modes.
def SubFOfNegZero :
    Pat<(Arith_SubFOp (ConstantLikeMatcher AnyAttr:$c), $x, $fmf, $rm),
        (Arith_NegFOp $x, $fmf),
        [(IsNegZeroFloat $c),
         (Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rm)]>;

//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

// mulf(negf(x), negf(y)) -> mulf(x,y)
// TODO: Verify if this canonicalization is safe when a rounding mode is
// specified. For the moment, bail on custom rounding modes.
def MulFOfNegF :
    Pat<(Arith_MulFOp (Arith_NegFOp $x, $_), (Arith_NegFOp $y, $_), $fmf, $rm),
        (Arith_MulFOp $x, $y, $fmf, $rm),
        [(TypesMatch $x, $y),
         (Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rm)]>;

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

// divf(negf(x), negf(y)) -> divf(x,y)
// TODO: Verify if this canonicalization is safe when a rounding mode is
// specified. For the moment, bail on custom rounding modes.
def DivFOfNegF :
    Pat<(Arith_DivFOp (Arith_NegFOp $x, $_), (Arith_NegFOp $y, $_), $fmf, $rm),
        (Arith_DivFOp $x, $y, $fmf, $rm),
        [(TypesMatch $x, $y),
         (Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rm)]>;

#endif // ARITH_PATTERNS
