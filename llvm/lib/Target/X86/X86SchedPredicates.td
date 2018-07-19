//===-- X86SchedPredicates.td - X86 Scheduling Predicates --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines scheduling predicate definitions that are common to
// all X86 subtargets.
//
//===----------------------------------------------------------------------===//

// A predicate used to identify dependency-breaking instructions that clear the
// content of the destination register. Note that this predicate only checks if
// input registers are the same. This predicate doesn't make any assumptions on
// the expected instruction opcodes, because different processors may implement
// different zero-idioms.
def ZeroIdiomPredicate : CheckSameRegOperand<1, 2>;

// A predicate used to check if an instruction is a LEA, and if it uses all
// three source operands: base, index, and offset.
def IsThreeOperandsLEAPredicate: CheckAll<[
  CheckOpcode<[LEA32r, LEA64r, LEA64_32r, LEA16r]>,

  // isRegOperand(Base)
  CheckIsRegOperand<1>,
  CheckNot<CheckInvalidRegOperand<1>>,

  // isRegOperand(Index)
  CheckIsRegOperand<3>,
  CheckNot<CheckInvalidRegOperand<3>>,

  // hasLEAOffset(Offset)
  CheckAny<[
    CheckAll<[
      CheckIsImmOperand<4>,
      CheckNot<CheckZeroOperand<4>>
    ]>,
    CheckNonPortable<"MI.getOperand(4).isGlobal()">
  ]>
]>;

// This predicate evaluates to true only if the input machine instruction is a
// 3-operands LEA.  Tablegen automatically generates a new method for it in
// X86GenInstrInfo.
def IsThreeOperandsLEAFn :
    TIIPredicate<"X86", "isThreeOperandsLEA", IsThreeOperandsLEAPredicate>;
