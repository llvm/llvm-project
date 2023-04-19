//===- OpenACCOps.td - OpenACC operation definitions -------*- tablegen -*-===//
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

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/OpenACC/OpenACCBase.td"
include "mlir/Dialect/OpenACC/OpenACCOpsTypes.td"
include "mlir/Dialect/OpenACC/OpenACCTypeInterfaces.td"

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

// Simple alias to pointer-like interface to reduce verbosity.
def OpenACC_PointerLikeType : TypeAlias<OpenACC_PointerLikeTypeInterface,
	"pointer-like type">;

// Define the OpenACC data clauses. There are a few cases where a modifier
// is used, like create(zero), copyin(readonly), and copyout(zero). Since in
// some cases we decompose the original acc data clauses into multiple acc
// dialect operations, we need to keep track of original clause. Thus even
// for the clause with modifier, we create separate operation to make this
// possible.
def OpenACC_CopyinClause          : I64EnumAttrCase<"acc_copyin", 1>;
def OpenACC_CopyinReadonlyClause  : I64EnumAttrCase<"acc_copyin_readonly", 2>;
def OpenACC_CopyClause            : I64EnumAttrCase<"acc_copy", 3>;
def OpenACC_CopyoutClause         : I64EnumAttrCase<"acc_copyout", 4>;
def OpenACC_CopyoutZeroClause     : I64EnumAttrCase<"acc_copyout_zero", 5>;
def OpenACC_PresentClause         : I64EnumAttrCase<"acc_present", 6>;
def OpenACC_CreateClause          : I64EnumAttrCase<"acc_create", 7>;
def OpenACC_CreateZeroClause      : I64EnumAttrCase<"acc_create_zero", 8>;
def OpenACC_DeleteClause          : I64EnumAttrCase<"acc_delete", 9>;
def OpenACC_AttachClause          : I64EnumAttrCase<"acc_attach", 10>;
def OpenACC_DetachClause          : I64EnumAttrCase<"acc_detach", 11>;
def OpenACC_NoCreateClause        : I64EnumAttrCase<"acc_no_create", 12>;
def OpenACC_PrivateClause         : I64EnumAttrCase<"acc_private", 13>;
def OpenACC_FirstPrivateClause    : I64EnumAttrCase<"acc_firstprivate", 14>;
def OpenACC_IsDevicePtrClause     : I64EnumAttrCase<"acc_deviceptr", 15>;
def OpenACC_GetDevicePtrClause    : I64EnumAttrCase<"acc_getdeviceptr", 16>;

def OpenACC_DataClauseEnum : I64EnumAttr<"DataClause",
    "data clauses supported by OpenACC",
    [OpenACC_CopyinClause, OpenACC_CopyinReadonlyClause, OpenACC_CopyClause,
     OpenACC_CopyoutClause, OpenACC_CopyoutZeroClause, OpenACC_PresentClause,
     OpenACC_CreateClause, OpenACC_CreateZeroClause, OpenACC_DeleteClause,
     OpenACC_AttachClause, OpenACC_DetachClause, OpenACC_NoCreateClause,
     OpenACC_PrivateClause, OpenACC_FirstPrivateClause,
     OpenACC_IsDevicePtrClause, OpenACC_GetDevicePtrClause
    ]> {
  let cppNamespace = "::mlir::acc";
}

// Used for data specification in data clauses (2.7.1).
// Either (or both) extent and upperbound must be specified.
def OpenACC_DataBoundsOp : OpenACC_Op<"bounds",
    [AttrSizedOperandSegments, NoMemoryEffect]> {
  let summary = "Represents bounds information for acc data clause.";

  let arguments = (ins Optional<AnyType>:$lowerbound,
                       Optional<AnyType>:$upperbound,
                       Optional<AnyType>:$extent,
                       Optional<AnyType>:$stride,
                       DefaultValuedAttr<BoolAttr, "false">:$strideInBytes,
                       Optional<AnyType>:$startIdx);
  let results = (outs OpenACC_DataBoundsType:$result);

  let assemblyFormat = [{
    oilist(
        `lowerbound` `(` $lowerbound `:` type($lowerbound) `)`
      | `upperbound` `(` $upperbound `:` type($upperbound) `)`
      | `extent` `(` $extent `:` type($extent) `)`
      | `stride` `(` $stride `:` type($stride) `)`
      | `startIdx` `(` $startIdx `:` type($startIdx) `)`
    ) attr-dict
  }];

  let hasVerifier = 1;
}

// Data entry operation does not refer to OpenACC spec terminology, but to
// terminology used in this dialect. It refers to data operations that will
// appear before data or compute region. It will be used as the base of acc
// dialect operations for the following OpenACC data clauses: copyin, create,
// present, attach, deviceptr.
//
// The bounds are represented in rank order. Rank 0 (inner-most dimension) is
// the first.
class OpenACC_DataEntryOp<string mnemonic, string clause, list<Trait> traits = []> :
    OpenACC_Op<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments])> {
  let arguments = (ins OpenACC_PointerLikeTypeInterface:$varPtr,
                       Optional<OpenACC_PointerLikeTypeInterface>:$varPtrPtr,
                       Variadic<OpenACC_DataBoundsType>:$bounds, /* rank-0 to rank-{n-1} */
                       DefaultValuedAttr<OpenACC_DataClauseEnum,clause>:$dataClause,
                       OptionalAttr<OpenACC_DataClauseEnum>:$decomposedFrom,
                       DefaultValuedAttr<BoolAttr, "true">:$structured,
                       DefaultValuedAttr<BoolAttr, "false">:$implicit,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs OpenACC_PointerLikeTypeInterface:$accPtr);

  let assemblyFormat = [{
    `varPtr` `(` $varPtr `:` type($varPtr) `)`
    oilist(
        `varPtrPtr` `(` $varPtrPtr `:` type($varPtrPtr) `)`
      | `bounds` `(` $bounds `)`
    ) `->` type($accPtr) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.7.4 deviceptr clause
//===----------------------------------------------------------------------===//
def OpenACC_DevicePtrOp : OpenACC_DataEntryOp<"deviceptr",
    "mlir::acc::DataClause::acc_deviceptr"> {
  let summary = "Specifies that the variable pointer is a device pointer.";
}

//===----------------------------------------------------------------------===//
// 2.7.5 present clause
//===----------------------------------------------------------------------===//
def OpenACC_PresentOp : OpenACC_DataEntryOp<"present",
    "mlir::acc::DataClause::acc_present"> {
  let summary = "Specifies that the variable is already present on device.";
}

//===----------------------------------------------------------------------===//
// 2.7.7 copyin clause
//===----------------------------------------------------------------------===//
def OpenACC_CopyinOp : OpenACC_DataEntryOp<"copyin",
    "mlir::acc::DataClause::acc_copyin"> {
  let summary = "Represents copyin semantics for acc data clauses like acc "
                "copyin and acc copy.";

  let extraClassDeclaration = [{
    /// Check if this is a copyin with readonly modifier.
    bool isCopyinReadonly();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.9 create clause
//===----------------------------------------------------------------------===//
def OpenACC_CreateOp : OpenACC_DataEntryOp<"create",
    "mlir::acc::DataClause::acc_create"> {
  let summary = "Represents create semantics for acc data clauses like acc "
                "create and acc copyout.";

  let extraClassDeclaration = [{
    /// Check if this is a create with zero modifier.
    bool isCreateZero();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.10 no_create clause
//===----------------------------------------------------------------------===//
def OpenACC_NoCreateOp : OpenACC_DataEntryOp<"nocreate",
    "mlir::acc::DataClause::acc_no_create"> {
  let summary = "Represents acc no_create semantics.";
}

//===----------------------------------------------------------------------===//
// 2.7.12 attach clause
//===----------------------------------------------------------------------===//
def OpenACC_AttachOp : OpenACC_DataEntryOp<"attach",
    "mlir::acc::DataClause::acc_attach"> {
  let summary = "Represents acc attach semantics which updates a pointer in "
                "device memory with the corresponding device address of the "
                "pointee.";
}

//===----------------------------------------------------------------------===//
// 3.2.23 acc_deviceptr
//===----------------------------------------------------------------------===//
// This is needed to get device address without the additional semantics in
// acc present.
// It is also useful for providing the device address for unstructured construct
// exit_data since unlike structured constructs, there is no matching data entry
// operation.
def OpenACC_GetDevicePtrOp : OpenACC_DataEntryOp<"getdeviceptr",
    "mlir::acc::DataClause::acc_getdeviceptr"> {
  let summary = "Gets device address from host address if it exists on device.";
}

// Data exit operation does not refer to OpenACC spec terminology, but to
// terminology used in this dialect. It refers to data operations that will appear
// after data or compute region. It will be used as the base of acc dialect
// operations for the following OpenACC data clauses: copyout, detach, delete.
class OpenACC_DataExitOp<string mnemonic, string clause, list<Trait> traits = []> :
    OpenACC_Op<mnemonic, !listconcat(traits,
        [AttrSizedOperandSegments])> {
  let arguments = (ins Optional<OpenACC_PointerLikeTypeInterface>:$varPtr,
                       OpenACC_PointerLikeTypeInterface:$accPtr,
                       Variadic<OpenACC_DataBoundsType>:$bounds,
                       DefaultValuedAttr<OpenACC_DataClauseEnum,clause>:$dataClause,
                       OptionalAttr<OpenACC_DataClauseEnum>:$decomposedFrom,
                       DefaultValuedAttr<BoolAttr, "true">:$structured,
                       DefaultValuedAttr<BoolAttr, "false">:$implicit,
                       OptionalAttr<StrAttr>:$name);

  let assemblyFormat = [{
    `accPtr` `(` $accPtr `:` type($accPtr) `)`
    oilist(
        `bounds` `(` $bounds `)`
      | `to` `varPtr` `(` $varPtr `:` type($varPtr) `)`
    ) attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.7.8 copyout clause
//===----------------------------------------------------------------------===//
def OpenACC_CopyoutOp : OpenACC_DataExitOp<"copyout",
    "mlir::acc::DataClause::acc_copyout"> {
  let summary = "Represents acc copyout semantics - reverse of copyin.";

  let extraClassDeclaration = [{
    /// Check if this is a copyout with zero modifier.
    bool isCopyoutZero();
  }];
}

//===----------------------------------------------------------------------===//
// 2.7.11 delete clause
//===----------------------------------------------------------------------===//
def OpenACC_DeleteOp : OpenACC_DataExitOp<"delete",
    "mlir::acc::DataClause::acc_delete"> {
  let summary = "Represents acc delete semantics - reverse of create.";
}

//===----------------------------------------------------------------------===//
// 2.7.13 detach clause
//===----------------------------------------------------------------------===//
def OpenACC_DetachOp : OpenACC_DataExitOp<"detach",
    "mlir::acc::DataClause::acc_detach"> {
  let summary = "Represents acc detach semantics - reverse of attach.";
}

//===----------------------------------------------------------------------===//
// 2.5.1 parallel Construct
//===----------------------------------------------------------------------===//

def OpenACC_ParallelOp : OpenACC_Op<"parallel",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
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
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
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
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `attach` `(` $attachOperands `:` type($attachOperands) `)`
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
// 2.5.2 serial Construct
//===----------------------------------------------------------------------===//

def OpenACC_SerialOp : OpenACC_Op<"serial",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "serial construct";
  let description = [{
    The "acc.serial" operation represents a serial construct block. It has
    one region to be executed in serial on the current device.

    Example:

    ```mlir
    acc.serial private(%c : memref<10xf32>) {
      // serial region
    }
    ```
  }];

  let arguments = (ins Optional<IntOrIndex>:$async,
                       UnitAttr:$asyncAttr,
                       Variadic<IntOrIndex>:$waitOperands,
                       UnitAttr:$waitAttr,
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
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
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
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `attach` `(` $attachOperands `:` type($attachOperands) `)`
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
      | `private` `(` $gangPrivateOperands `:` type($gangPrivateOperands) `)`
      | `present` `(` $presentOperands `:` type($presentOperands) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `self` `(` $selfCond `)`
      | `if` `(` $ifCond `)`
      | `reduction` `(` $reductionOperands `:` type($reductionOperands) `)`
    )
    $region attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// 2.5.1 kernels Construct
//===----------------------------------------------------------------------===//

def OpenACC_KernelsOp : OpenACC_Op<"kernels",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
  let summary = "kernels construct";
  let description = [{
    The "acc.kernels" operation represents a kernels construct block. It has
    one region to be compiled into a sequence of kernels for execution on the
    current device.

    Example:

    ```mlir
    acc.kernels num_gangs(%c10) num_workers(%c10)
        private(%c : memref<10xf32>) {
      // kernels region
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
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
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
        `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
      | `attach` `(` $attachOperands `:` type($attachOperands) `)`
      | `async` `(` $async `:` type($async) `)`
      | `copy` `(` $copyOperands `:` type($copyOperands) `)`
      | `copyin` `(` $copyinOperands `:` type($copyinOperands) `)`
      | `copyin_readonly` `(` $copyinReadonlyOperands `:`
          type($copyinReadonlyOperands) `)`
      | `copyout` `(` $copyoutOperands `:` type($copyoutOperands) `)`
      | `copyout_zero` `(` $copyoutZeroOperands `:`
          type($copyoutZeroOperands) `)`
      | `create` `(` $createOperands `:` type($createOperands) `)`
      | `create_zero` `(` $createZeroOperands `:` type($createZeroOperands) `)`
      | `deviceptr` `(` $devicePtrOperands `:` type($devicePtrOperands) `)`
      | `no_create` `(` $noCreateOperands `:` type($noCreateOperands) `)`
      | `num_gangs` `(` $numGangs `:` type($numGangs) `)`
      | `num_workers` `(` $numWorkers `:` type($numWorkers) `)`
      | `present` `(` $presentOperands `:` type($presentOperands) `)`
      | `vector_length` `(` $vectorLength `:` type($vectorLength) `)`
      | `wait` `(` $waitOperands `:` type($waitOperands) `)`
      | `self` `(` $selfCond `)`
      | `if` `(` $ifCond `)`
    )
    $region attr-dict-with-keyword
  }];
}

//===----------------------------------------------------------------------===//
// 2.6.5 data Construct
//===----------------------------------------------------------------------===//

def OpenACC_DataOp : OpenACC_Op<"data",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
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
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
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
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
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
                       Variadic<AnyType>:$attachOperands,
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands);

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
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
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
                       Variadic<OpenACC_PointerLikeTypeInterface>:$dataClauseOperands,
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
      | `dataOperands` `(` $dataClauseOperands `:` type($dataClauseOperands) `)`
    )
    attr-dict-with-keyword
  }];

  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// 2.9 loop Construct
//===----------------------------------------------------------------------===//

def OpenACC_LoopOp : OpenACC_Op<"loop",
    [AttrSizedOperandSegments, RecursiveMemoryEffects]> {
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
                       UnitAttr:$hasGang,
                       UnitAttr:$hasWorker,
                       UnitAttr:$hasVector,
                       Variadic<IntOrIndex>:$tileOperands,
                       Variadic<AnyType>:$privateOperands,
                       OptionalAttr<OpenACC_ReductionOpAttr>:$reductionOp,
                       Variadic<AnyType>:$reductionOperands);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$region);

  let extraClassDeclaration = [{
    static StringRef getAutoAttrStrName() { return "auto"; }
    static StringRef getGangNumKeyword() { return "num"; }
    static StringRef getGangStaticKeyword() { return "static"; }
  }];

  let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    oilist(
        `gang` `` custom<GangClause>($gangNum, type($gangNum), $gangStatic, type($gangStatic), $hasGang)
      | `worker` `` custom<WorkerClause>($workerNum, type($workerNum), $hasWorker)
      | `vector` `` custom<VectorClause>($vectorLength, type($vectorLength), $hasVector)
      | `private` `(` $privateOperands `:` type($privateOperands) `)`
      | `tile` `(` $tileOperands `:` type($tileOperands) `)`
      | `reduction` `(` $reductionOperands `:` type($reductionOperands) `)`
    )
    $region
    ( `(` type($results)^ `)` )?
    attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

// Yield operation for the acc.loop and acc.parallel operations.
def OpenACC_YieldOp : OpenACC_Op<"yield", [Terminator,
    ParentOneOf<["ParallelOp, LoopOp, SerialOp"]>]> {
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
