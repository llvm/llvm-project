//===- ControlFlowOps.td - ControlFlow operations ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the operations within the ControlFlow
// dialect.
//
//===----------------------------------------------------------------------===//

#ifndef STANDARD_OPS
#define STANDARD_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ControlFlow_Dialect : Dialect {
  let name = "cf";
  let cppNamespace = "::mlir::cf";
  let dependentDialects = ["arith::ArithmeticDialect"];
  let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let description = [{
    This dialect contains low-level, i.e. non-region based, control flow
    constructs. These constructs generally represent control flow directly
    on SSA blocks of a control flow graph.
  }];
}

class CF_Op<string mnemonic, list<Trait> traits = []> :
    Op<ControlFlow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

def AssertOp : CF_Op<"assert"> {
  let summary = "Assert operation with message attribute";
  let description = [{
    Assert operation with single boolean operand and an error message attribute.
    If the argument is `true` this operation has no effect. Otherwise, the
    program execution will abort. The provided error message may be used by a
    runtime to propagate the error to the user.

    Example:

    ```mlir
    assert %b, "Expected ... to be true"
    ```
  }];

  let arguments = (ins I1:$arg, StrAttr:$msg);

  let assemblyFormat = "$arg `,` $msg attr-dict";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

def BranchOp : CF_Op<"br", [
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
    NoSideEffect, Terminator
  ]> {
  let summary = "branch operation";
  let description = [{
    The `cf.br` operation represents a direct branch operation to a given
    block. The operands of this operation are forwarded to the successor block,
    and the number and type of the operands must match the arguments of the
    target block.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      cf.br ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "Block *":$dest,
                   CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

def CondBranchOp : CF_Op<"cond_br",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "conditional branch operation";
  let description = [{
    The `cond_br` terminator operation represents a conditional branch on a
    boolean (1-bit integer) value. If the bit is set, then the first destination
    is jumped to; if it is false, the second destination is chosen. The count
    and types of operands must align with the arguments in the corresponding
    target blocks.

    The MLIR conditional branch operation is not allowed to target the entry
    block for a region. The two destinations of the conditional branch operation
    are allowed to be the same.

    The following example illustrates a function with a conditional branch
    operation that targets the same block.

    Example:

    ```mlir
    func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
  }];

  let arguments = (ins I1:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "ValueRange":$trueOperands, "Block *":$falseDest,
      "ValueRange":$falseOperands), [{
      build($_builder, $_state, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    }]>,
    OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
      "Block *":$falseDest, CArg<"ValueRange", "{}">:$falseOperands), [{
      build($_builder, $_state, condition, trueDest, ValueRange(), falseDest,
            falseOperands);
    }]>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // Accessors for operands to the 'true' destination.
    Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getTrueDestOperandsMutable().erase(index);
    }

    // Accessors for operands to the 'false' destination.
    Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return getTrueDestOperands(); }
    operand_range getFalseOperands() { return getFalseDestOperands(); }

    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getFalseDestOperandsMutable().erase(index);
    }

  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let hasCanonicalizer = 1;
  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//

def SwitchOp : CF_Op<"switch",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     NoSideEffect, Terminator]> {
  let summary = "switch operation";
  let description = [{
    The `switch` terminator operation represents a switch on a signless integer
    value. If the flag matches one of the specified cases, then the
    corresponding destination is jumped to. If the flag does not match any of
    the cases, the default destination is jumped to. The count and types of
    operands must align with the arguments in the corresponding target blocks.

    Example:

    ```mlir
    switch %flag : i32, [
      default: ^bb1(%a : i32),
      42: ^bb1(%b : i32),
      43: ^bb3(%c : i32)
    ]
    ```
  }];

  let arguments = (ins
    AnyInteger:$flag,
    Variadic<AnyType>:$defaultOperands,
    VariadicOfVariadic<AnyType, "case_operand_segments">:$caseOperands,
    OptionalAttr<AnyIntElementsAttr>:$case_values,
    I32ElementsAttr:$case_operand_segments
  );
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );
  let builders = [
    OpBuilder<(ins "Value":$flag,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<APInt>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>,
    OpBuilder<(ins "Value":$flag,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"ArrayRef<int32_t>", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>,
    OpBuilder<(ins "Value":$flag,
      "Block *":$defaultDestination,
      "ValueRange":$defaultOperands,
      CArg<"DenseIntElementsAttr", "{}">:$caseValues,
      CArg<"BlockRange", "{}">:$caseDestinations,
      CArg<"ArrayRef<ValueRange>", "{}">:$caseOperands)>
  ];

  let assemblyFormat = [{
    $flag `:` type($flag) `,` `[` `\n`
      custom<SwitchOpCases>(ref(type($flag)),$defaultDestination,
                            $defaultOperands,
                            type($defaultOperands),
                            $case_values,
                            $caseDestinations,
                            $caseOperands,
                            type($caseOperands))
   `]`
    attr-dict
  }];

  let extraClassDeclaration = [{
    /// Return the operands for the case destination block at the given index.
    OperandRange getCaseOperands(unsigned index) {
      return getCaseOperands()[index];
    }

    /// Return a mutable range of operands for the case destination block at the
    /// given index.
    MutableOperandRange getCaseOperandsMutable(unsigned index) {
      return getCaseOperandsMutable()[index];
    }
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

#endif // STANDARD_OPS
