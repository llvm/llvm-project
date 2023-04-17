//===- OpenACC.td - OpenACC operation definitions ----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// =============================================================================
//
// Defines MLIR OpenACC operations.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACC_OPS
#define OPENACC_OPS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def OpenACC_Dialect : Dialect {
  let name = "acc";

  let summary = "An OpenACC dialect for MLIR.";

  let description = [{
    This dialect models the construct from the OpenACC 3.1 directive language.
  }];

  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::acc";
}

// AccCommon requires definition of OpenACC_Dialect.
include "mlir/Dialect/OpenACC/AccCommon.td"

// Base class for OpenACC dialect ops.
class OpenACC_Op<string mnemonic, list<Trait> traits = []> :
  Op<OpenACC_Dialect, mnemonic, traits>;

// Reduction operation enumeration.
def OpenACC_ReductionOpAdd     : I32EnumAttrCase<"redop_add", 0>;
def OpenACC_ReductionOpMul     : I32EnumAttrCase<"redop_mul", 1>;
def OpenACC_ReductionOpMax     : I32EnumAttrCase<"redop_max", 2>;
def OpenACC_ReductionOpMin     : I32EnumAttrCase<"redop_min", 3>;
def OpenACC_ReductionOpAnd     : I32EnumAttrCase<"redop_and", 4>;
def OpenACC_ReductionOpOr      : I32EnumAttrCase<"redop_or", 5>;
def OpenACC_ReductionOpXor     : I32EnumAttrCase<"redop_xor", 6>;
def OpenACC_ReductionOpLogEqv  : I32EnumAttrCase<"redop_leqv", 7>;
def OpenACC_ReductionOpLogNeqv : I32EnumAttrCase<"redop_lneqv", 8>;
def OpenACC_ReductionOpLogAnd  : I32EnumAttrCase<"redop_land", 9>;
def OpenACC_ReductionOpLogOr   : I32EnumAttrCase<"redop_lor", 10>;

def OpenACC_ReductionOp : I32EnumAttr<"ReductionOp",
    "built-in reduction operations supported by OpenACC",
    [OpenACC_ReductionOpAdd, OpenACC_ReductionOpMul, OpenACC_ReductionOpMax,
     OpenACC_ReductionOpMin, OpenACC_ReductionOpAnd, OpenACC_ReductionOpOr,
     OpenACC_ReductionOpXor, OpenACC_ReductionOpLogEqv,
     OpenACC_ReductionOpLogNeqv, OpenACC_ReductionOpLogAnd,
     OpenACC_ReductionOpLogOr
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::acc";
}
def OpenACC_ReductionOpAttr : EnumAttr<OpenACC_Dialect, OpenACC_ReductionOp,
                                       "reduction_op">;

// Type used in operation below.
def IntOrIndex : AnyTypeOf<[AnyInteger, Index]>;

//===----------------------------------------------------------------------===//
// 2.5.1 parallel Construct
//===----------------------------------------------------------------------===//

def OpenACC_ParallelOp : OpenACC_Op<"parallel",
    [AttrSizedOperandSegments]> {
  let summary = "parallel construct";
  let description = [{
    The "acc.parallel" operation represents a parallel construct block. It has
    one region to be executed in parallel on the current device.

    Example:

    ```mlir
    acc.parallel num_gangs(%c10) num_workers(%c10)
        private(%c : memref<10xf32>) {
      // parallel region
    }
    ```
  }];

  let arguments = (ins Optional<IntOrIndex>:$async,
                       UnitAttr:$asyncAttr,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$waitAttr,
                       Optional<IntOrIndex>:$numGangs,
                       Optional<IntOrIndex>:$numWorkers,
                       Optional<IntOrIndex>:$vectorLength,
                       Optional<I1>:$ifCond,
                       Optional<I1>:$selfCond,
                       UnitAttr:$selfAttr,
                       OptionalAttr<OpenACC_ReductionOpAttr>:$reductionOp,
                       Variadic<AnyType>:$reductionOperands,
                       Variadic<AnyType>:$copyOperands,
                       Variadic<AnyType>:$copyinOperands,
                       Variadic<AnyType>:$copyinReadonlyOperands,
                       Variadic<AnyType>:$copyoutOperands,
                       Variadic<AnyType>:$copyoutZeroOperands,
                       Variadic<AnyType>:$createOperands,
                       Variadic<AnyType>:$createZeroOperands,
                       Variadic<AnyType>:$noCreateOperands,
                       Variadic<AnyType>:$presentOperands,
                       Variadic<AnyType>:$devicePtrOperands,
                       Variadic<AnyType>:$attachOperands,
                       Variadic<AnyType>:$gangPrivateOperands,
                       Variadic<AnyType>:$gangFirstPrivateOperands,
                       OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `attach` `(` $attachOperands `:` type($attachOperands) `)`
      | `async` `(` $async `:` type($async) `)`
      | `copy` `(` $copyOperands `:` type($copyOperands) `)`
      | `copyin` `(` $copyinOperands `:` type($copyinOperands) `)`
      | `copyin_readonly` `(` $copyinReadonlyOperands `:`
          type($copyinReadonlyOperands) `)`
      | `copyout` `(` $copyoutOperands `:` type($copyoutOperands) `)`
      | `copyout_zero` `(` $copyoutZeroOperands `:`
          type($copyoutZeroOperands) `)`
      | `create` `(` $createOperands `:` type($createOperands) `)`
      | `create_zero` `(` $createZeroOperands `:`
          type($createZeroOperands) `)`
      | `deviceptr` `(` $devicePtrOperands `:` type($devicePtrOperands) `)`
      | `firstprivate` `(` $gangFirstPrivateOperands `:`
            type($gangFirstPrivateOperands) `)`
      | `no_create` `(` $noCreateOperands `:` type($noCreateOperands) `)`
      | `num_gangs` `(` $numGangs `:` type($numGangs) `)`
      | `num_workers` `(` $numWorkers `:` type($numWorkers) `)`
      | `private` `(` $gangPrivateOperands `:` type($gangPrivateOperands) `)`
      | `present` `(` $presentOperands `:` type($presentOperands) `)`
      | `vector_length` `(` $vectorLength `:` type($vectorLength) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `self` `(` $selfCond `)`
      | `if` `(` $ifCond `)`
      | `reduction` `(` $reductionOperands `:` type($reductionOperands) `)`
    )
    $region attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// 2.6.5 data Construct
//===----------------------------------------------------------------------===//

def OpenACC_DataOp : OpenACC_Op<"data",
    [AttrSizedOperandSegments]> {
  let summary = "data construct";

  let description = [{
    The "acc.data" operation represents a data construct. It defines vars to
    be allocated in the current device memory for the duration of the region,
    whether data should be copied from local memory to the current device
    memory upon region entry , and copied from device memory to local memory
    upon region exit.

    Example:

    ```mlir
    acc.data present(%a: memref<10x10xf32>, %b: memref<10x10xf32>,
        %c: memref<10xf32>, %d: memref<10xf32>) {
      // data region
    }
    ```
  }];


  let arguments = (ins Optional<I1>:$ifCond,
                       Variadic<AnyType>:$copyOperands,
                       Variadic<AnyType>:$copyinOperands,
                       Variadic<AnyType>:$copyinReadonlyOperands,
                       Variadic<AnyType>:$copyoutOperands,
                       Variadic<AnyType>:$copyoutZeroOperands,
                       Variadic<AnyType>:$createOperands,
                       Variadic<AnyType>:$createZeroOperands,
                       Variadic<AnyType>:$noCreateOperands,
                       Variadic<AnyType>:$presentOperands,
                       Variadic<AnyType>:$deviceptrOperands,
                       Variadic<AnyType>:$attachOperands,
                       OptionalAttr<DefaultValueAttr>:$defaultAttr);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `copy` `(` $copyOperands `:` type($copyOperands) `)`
      | `copyin` `(` $copyinOperands `:` type($copyinOperands) `)`
      | `copyin_readonly` `(` $copyinReadonlyOperands `:`
            type($copyinReadonlyOperands) `)`
      | `copyout` `(` $copyoutOperands `:` type($copyoutOperands) `)`
      | `copyout_zero` `(` $copyoutZeroOperands `:`
            type($copyoutZeroOperands) `)`
      | `create` `(` $createOperands `:` type($createOperands) `)`
      | `create_zero` `(` $createZeroOperands `:`
            type($createZeroOperands) `)`
      | `no_create` `(` $noCreateOperands `:` type($noCreateOperands) `)`
      | `present` `(` $presentOperands `:` type($presentOperands) `)`
      | `deviceptr` `(` $deviceptrOperands `:` type($deviceptrOperands) `)`
      | `attach` `(` $attachOperands `:` type($attachOperands) `)`
    )
    $region attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

def OpenACC_TerminatorOp : OpenACC_Op<"terminator", [Terminator]> {
  let summary = "Generic terminator for OpenACC regions";

  let description = [{
    A terminator operation for regions that appear in the body of OpenACC
    operation. Generic OpenACC construct regions are not expected to return any
    value so the terminator takes no operands. The terminator op returns control
    to the enclosing op.
  }];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// 2.6.6 Enter Data Directive
//===----------------------------------------------------------------------===//

def OpenACC_EnterDataOp : OpenACC_Op<"enter_data", [AttrSizedOperandSegments]> {
  let summary = "enter data operation";

  let description = [{
    The "acc.enter_data" operation represents the OpenACC enter data directive.

    Example:

    ```mlir
    acc.enter_data create(%d1 : memref<10xf32>) attributes {async}
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$asyncOperand,
                       UnitAttr:$async,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$wait,
                       Variadic<AnyType>:$copyinOperands,
                       Variadic<AnyType>:$createOperands,
                       Variadic<AnyType>:$createZeroOperands,
                       Variadic<AnyType>:$attachOperands);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $asyncOperand `:` type($asyncOperand) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `copyin` `(` $copyinOperands `:` type($copyinOperands) `)`
      | `create` `(` $createOperands `:` type($createOperands) `)`
      | `create_zero` `(` $createZeroOperands `:`
          type($createZeroOperands) `)`
      | `attach` `(` $attachOperands `:` type($attachOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.6.6 Exit Data Directive
//===----------------------------------------------------------------------===//

def OpenACC_ExitDataOp : OpenACC_Op<"exit_data", [AttrSizedOperandSegments]> {
  let summary = "exit data operation";

  let description = [{
    The "acc.exit_data" operation represents the OpenACC exit data directive.

    Example:

    ```mlir
    acc.exit_data delete(%d1 : memref<10xf32>) attributes {async}
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$asyncOperand,
                       UnitAttr:$async,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$wait,
                       Variadic<AnyType>:$copyoutOperands,
                       Variadic<AnyType>:$deleteOperands,
                       Variadic<AnyType>:$detachOperands,
                       UnitAttr:$finalize);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $asyncOperand `:` type($asyncOperand) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `copyout` `(` $copyoutOperands `:` type($copyoutOperands) `)`
      | `delete` `(` $deleteOperands `:` type($deleteOperands) `)`
      | `detach` `(` $detachOperands `:` type($detachOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.9 loop Construct
//===----------------------------------------------------------------------===//

def OpenACC_LoopOp : OpenACC_Op<"loop", [AttrSizedOperandSegments]> {
  let summary = "loop construct";

  let description = [{
    The "acc.loop" operation represents the OpenACC loop construct.

    Example:

    ```mlir
    acc.loop gang vector {
      scf.for %arg3 = %c0 to %c10 step %c1 {
        scf.for %arg4 = %c0 to %c10 step %c1 {
          scf.for %arg5 = %c0 to %c10 step %c1 {
            // ... body
          }
        }
      }
      acc.yield
    } attributes { collapse = 3 }
    ```
  }];

  let arguments = (ins OptionalAttr<I64Attr>:$collapse,
                       Optional<IntOrIndex>:$gangNum,
                       Optional<IntOrIndex>:$gangStatic,
                       Optional<IntOrIndex>:$workerNum,
                       Optional<IntOrIndex>:$vectorLength,
                       UnitAttr:$seq,
                       UnitAttr:$independent,
                       UnitAttr:$auto_,
                       Variadic<IntOrIndex>:$tileOperands,
                       Variadic<AnyType>:$privateOperands,
                       OptionalAttr<OpenACC_ReductionOpAttr>:$reductionOp,
                       Variadic<AnyType>:$reductionOperands,
                       DefaultValuedAttr<I64Attr, "0">:$exec_mapping);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    static StringRef getCollapseAttrStrName() { return "collapse"; }
    static StringRef getSeqAttrStrName() { return "seq"; }
    static StringRef getIndependentAttrStrName() { return "independent"; }
    static StringRef getAutoAttrStrName() { return "auto"; }
    static StringRef getExecutionMappingAttrStrName() { return "exec_mapping"; }
    static StringRef getGangKeyword() { return "gang"; }
    static StringRef getGangNumKeyword() { return "num"; }
    static StringRef getGangStaticKeyword() { return "static"; }
    static StringRef getVectorKeyword() { return "vector"; }
    static StringRef getWorkerKeyword() { return "worker"; }
    static StringRef getTileKeyword() { return "tile"; }
    static StringRef getPrivateKeyword() { return "private"; }
    static StringRef getReductionKeyword() { return "reduction"; }
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

// Yield operation for the acc.loop and acc.parallel operations.
def OpenACC_YieldOp : OpenACC_Op<"yield", [Terminator,
    ParentOneOf<["ParallelOp, LoopOp"]>]> {
  let summary = "Acc yield and termination operation";

  let description = [{
    `acc.yield` is a special terminator operation for block inside regions in
    acc ops (parallel and loop). It returns values to the immediately enclosing
    acc op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

//===----------------------------------------------------------------------===//
// 2.14.1. Init Directive
//===----------------------------------------------------------------------===//

def OpenACC_InitOp : OpenACC_Op<"init", [AttrSizedOperandSegments]> {
  let summary = "init operation";

  let description = [{
    The "acc.init" operation represents the OpenACC init executable
    directive.

    Example:

    ```mlir
    acc.init
    acc.init device_num(%dev1 : i32)
    ```
  }];

  let arguments = (ins Variadic<AnyInteger>:$deviceTypeOperands,
                       Optional<IntOrIndex>:$deviceNumOperand,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    oilist(
        `device_type` `(` $deviceTypeOperands `:` type($deviceTypeOperands) `)`
      | `device_num` `(` $deviceNumOperand `:` type($deviceNumOperand) `)`
      | `if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.2. Shutdown
//===----------------------------------------------------------------------===//

def OpenACC_ShutdownOp : OpenACC_Op<"shutdown", [AttrSizedOperandSegments]> {
  let summary = "shutdown operation";

  let description = [{
    The "acc.shutdown" operation represents the OpenACC shutdown executable
    directive.

    Example:

    ```mlir
    acc.shutdown
    acc.shutdown device_num(%dev1 : i32)
    ```
  }];

  let arguments = (ins Variadic<AnyInteger>:$deviceTypeOperands,
                       Optional<IntOrIndex>:$deviceNumOperand,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    oilist(`device_type` `(` $deviceTypeOperands `:` type($deviceTypeOperands) `)`
    |`device_num` `(` $deviceNumOperand `:` type($deviceNumOperand) `)`
    |`if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.14.4. Update Directive
//===----------------------------------------------------------------------===//

def OpenACC_UpdateOp : OpenACC_Op<"update", [AttrSizedOperandSegments]> {
  let summary = "update operation";

  let description = [{
    The "acc.udpate" operation represents the OpenACC update executable
    directive.
    As host and self clauses are synonyms, any operands for host and self are
    add to $hostOperands.

    Example:

    ```mlir
    acc.update device(%d1 : memref<10xf32>) attributes {async}
    ```
  }];

  let arguments = (ins Optional<I1>:$ifCond,
                       Optional<IntOrIndex>:$asyncOperand,
                       Optional<IntOrIndex>:$waitDevnum,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$async,
                       UnitAttr:$wait,
                       Variadic<IntOrIndex>:$deviceTypeOperands,
                       Variadic<AnyType>:$hostOperands,
                       Variadic<AnyType>:$deviceOperands,
                       UnitAttr:$ifPresent);

  let extraClassDeclaration = [{
    /// The number of data operands.
    unsigned getNumDataOperands();

    /// The i-th data operand passed.
    Value getDataOperand(unsigned i);
  }];

  let assemblyFormat = [{
    oilist(
        `if` `(` $ifCond `)`
      | `async` `(` $asyncOperand `:` type($asyncOperand) `)`
      | `wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      | `device_type` `(` $deviceTypeOperands `:`
          type($deviceTypeOperands) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `host` `(` $hostOperands `:` type($hostOperands) `)`
      | `device` `(` $deviceOperands `:` type($deviceOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.16.3. Wait Directive
//===----------------------------------------------------------------------===//

def OpenACC_WaitOp : OpenACC_Op<"wait", [AttrSizedOperandSegments]> {
  let summary = "wait operation";

  let description = [{
    The "acc.wait" operation represents the OpenACC wait executable
    directive.

    Example:

    ```mlir
    acc.wait(%value1: index)
    acc.wait() async(%async1: i32)
    ```
  }];

  let arguments = (ins Variadic<IntOrIndex>:$waitOperands,
                       Optional<IntOrIndex>:$asyncOperand,
                       Optional<IntOrIndex>:$waitDevnum,
                       UnitAttr:$async,
                       Optional<I1>:$ifCond);

  let assemblyFormat = [{
    ( `(` $waitOperands^ `:` type($waitOperands) `)` )?
    oilist(`async` `(` $asyncOperand `:` type($asyncOperand) `)`
      |`wait_devnum` `(` $waitDevnum `:` type($waitDevnum) `)`
      |`if` `(` $ifCond `)`
    ) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
}

#endif // OPENACC_OPS
