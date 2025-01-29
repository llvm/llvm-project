//===- MPI.td - Base defs for mpi dialect ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MPI_IR_MPI_TD
#define MLIR_DIALECT_MPI_IR_MPI_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def MPI_Dialect : Dialect {
  let name = "mpi";
  let cppNamespace = "::mlir::mpi";
  let description = [{
    This dialect models the Message Passing Interface (MPI), version 
    4.0. It is meant to serve as an interfacing dialect that is targeted
    by higher-level dialects. The MPI dialect itself can be lowered to 
    multiple MPI implementations and hide differences in ABI. The dialect
    models the functions of the MPI specification as close to 1:1 as possible
    while preserving SSA value semantics where it makes sense, and uses 
    `memref` types instead of bare pointers.

    This dialect is under active development, and while stability is an
    eventual goal, it is not guaranteed at this juncture. Given the early 
    state, it is recommended to inquire further prior to using this dialect.

    For an in-depth documentation of the MPI library interface, please refer 
    to official documentation such as the 
    [OpenMPI online documentation](https://www.open-mpi.org/doc/current/).
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Error classes enum:
//===----------------------------------------------------------------------===//

def MPI_CodeSuccess : I32EnumAttrCase<"MPI_SUCCESS", 0, "MPI_SUCCESS">;
def MPI_CodeErrAccess : I32EnumAttrCase<"MPI_ERR_ACCESS", 1, "MPI_ERR_ACCESS">;
def MPI_CodeErrAmode : I32EnumAttrCase<"MPI_ERR_AMODE", 2, "MPI_ERR_AMODE">;
def MPI_CodeErrArg : I32EnumAttrCase<"MPI_ERR_ARG", 3, "MPI_ERR_ARG">;
def MPI_CodeErrAssert : I32EnumAttrCase<"MPI_ERR_ASSERT", 4, "MPI_ERR_ASSERT">;
def MPI_CodeErrBadFile
    : I32EnumAttrCase<"MPI_ERR_BAD_FILE", 5, "MPI_ERR_BAD_FILE">;
def MPI_CodeErrBase : I32EnumAttrCase<"MPI_ERR_BASE", 6, "MPI_ERR_BASE">;
def MPI_CodeErrBuffer : I32EnumAttrCase<"MPI_ERR_BUFFER", 7, "MPI_ERR_BUFFER">;
def MPI_CodeErrComm : I32EnumAttrCase<"MPI_ERR_COMM", 8, "MPI_ERR_COMM">;
def MPI_CodeErrConversion
    : I32EnumAttrCase<"MPI_ERR_CONVERSION", 9, "MPI_ERR_CONVERSION">;
def MPI_CodeErrCount : I32EnumAttrCase<"MPI_ERR_COUNT", 10, "MPI_ERR_COUNT">;
def MPI_CodeErrDims : I32EnumAttrCase<"MPI_ERR_DIMS", 11, "MPI_ERR_DIMS">;
def MPI_CodeErrDisp : I32EnumAttrCase<"MPI_ERR_DISP", 12, "MPI_ERR_DISP">;
def MPI_CodeErrDupDatarep
    : I32EnumAttrCase<"MPI_ERR_DUP_DATAREP", 13, "MPI_ERR_DUP_DATAREP">;
def MPI_CodeErrErrhandler
    : I32EnumAttrCase<"MPI_ERR_ERRHANDLER", 14, "MPI_ERR_ERRHANDLER">;
def MPI_CodeErrFile : I32EnumAttrCase<"MPI_ERR_FILE", 15, "MPI_ERR_FILE">;
def MPI_CodeErrFileExists
    : I32EnumAttrCase<"MPI_ERR_FILE_EXISTS", 16, "MPI_ERR_FILE_EXISTS">;
def MPI_CodeErrFileInUse
    : I32EnumAttrCase<"MPI_ERR_FILE_IN_USE", 17, "MPI_ERR_FILE_IN_USE">;
def MPI_CodeErrGroup : I32EnumAttrCase<"MPI_ERR_GROUP", 18, "MPI_ERR_GROUP">;
def MPI_CodeErrInfo : I32EnumAttrCase<"MPI_ERR_INFO", 19, "MPI_ERR_INFO">;
def MPI_CodeErrInfoKey
    : I32EnumAttrCase<"MPI_ERR_INFO_KEY", 20, "MPI_ERR_INFO_KEY">;
def MPI_CodeErrInfoNokey
    : I32EnumAttrCase<"MPI_ERR_INFO_NOKEY", 21, "MPI_ERR_INFO_NOKEY">;
def MPI_CodeErrInfoValue
    : I32EnumAttrCase<"MPI_ERR_INFO_VALUE", 22, "MPI_ERR_INFO_VALUE">;
def MPI_CodeErrInStatus
    : I32EnumAttrCase<"MPI_ERR_IN_STATUS", 23, "MPI_ERR_IN_STATUS">;
def MPI_CodeErrIntern : I32EnumAttrCase<"MPI_ERR_INTERN", 24, "MPI_ERR_INTERN">;
def MPI_CodeErrIo : I32EnumAttrCase<"MPI_ERR_IO", 25, "MPI_ERR_IO">;
def MPI_CodeErrKeyval : I32EnumAttrCase<"MPI_ERR_KEYVAL", 26, "MPI_ERR_KEYVAL">;
def MPI_CodeErrLocktype
    : I32EnumAttrCase<"MPI_ERR_LOCKTYPE", 27, "MPI_ERR_LOCKTYPE">;
def MPI_CodeErrName : I32EnumAttrCase<"MPI_ERR_NAME", 28, "MPI_ERR_NAME">;
def MPI_CodeErrNoMem : I32EnumAttrCase<"MPI_ERR_NO_MEM", 29, "MPI_ERR_NO_MEM">;
def MPI_CodeErrNoSpace
    : I32EnumAttrCase<"MPI_ERR_NO_SPACE", 30, "MPI_ERR_NO_SPACE">;
def MPI_CodeErrNoSuchFile
    : I32EnumAttrCase<"MPI_ERR_NO_SUCH_FILE", 31, "MPI_ERR_NO_SUCH_FILE">;
def MPI_CodeErrNotSame
    : I32EnumAttrCase<"MPI_ERR_NOT_SAME", 32, "MPI_ERR_NOT_SAME">;
def MPI_CodeErrOp : I32EnumAttrCase<"MPI_ERR_OP", 33, "MPI_ERR_OP">;
def MPI_CodeErrOther : I32EnumAttrCase<"MPI_ERR_OTHER", 34, "MPI_ERR_OTHER">;
def MPI_CodeErrPending
    : I32EnumAttrCase<"MPI_ERR_PENDING", 35, "MPI_ERR_PENDING">;
def MPI_CodeErrPort : I32EnumAttrCase<"MPI_ERR_PORT", 36, "MPI_ERR_PORT">;
def MPI_CodeErrProcAborted
    : I32EnumAttrCase<"MPI_ERR_PROC_ABORTED", 37, "MPI_ERR_PROC_ABORTED">;
def MPI_CodeErrQuota : I32EnumAttrCase<"MPI_ERR_QUOTA", 38, "MPI_ERR_QUOTA">;
def MPI_CodeErrRank : I32EnumAttrCase<"MPI_ERR_RANK", 39, "MPI_ERR_RANK">;
def MPI_CodeErrReadOnly
    : I32EnumAttrCase<"MPI_ERR_READ_ONLY", 40, "MPI_ERR_READ_ONLY">;
def MPI_CodeErrRequest
    : I32EnumAttrCase<"MPI_ERR_REQUEST", 41, "MPI_ERR_REQUEST">;
def MPI_CodeErrRmaAttach
    : I32EnumAttrCase<"MPI_ERR_RMA_ATTACH", 42, "MPI_ERR_RMA_ATTACH">;
def MPI_CodeErrRmaConflict
    : I32EnumAttrCase<"MPI_ERR_RMA_CONFLICT", 43, "MPI_ERR_RMA_CONFLICT">;
def MPI_CodeErrRmaFlavor
    : I32EnumAttrCase<"MPI_ERR_RMA_FLAVOR", 44, "MPI_ERR_RMA_FLAVOR">;
def MPI_CodeErrRmaRange
    : I32EnumAttrCase<"MPI_ERR_RMA_RANGE", 45, "MPI_ERR_RMA_RANGE">;
def MPI_CodeErrRmaShared
    : I32EnumAttrCase<"MPI_ERR_RMA_SHARED", 46, "MPI_ERR_RMA_SHARED">;
def MPI_CodeErrRmaSync
    : I32EnumAttrCase<"MPI_ERR_RMA_SYNC", 47, "MPI_ERR_RMA_SYNC">;
def MPI_CodeErrRoot : I32EnumAttrCase<"MPI_ERR_ROOT", 48, "MPI_ERR_ROOT">;
def MPI_CodeErrService
    : I32EnumAttrCase<"MPI_ERR_SERVICE", 49, "MPI_ERR_SERVICE">;
def MPI_CodeErrSession
    : I32EnumAttrCase<"MPI_ERR_SESSION", 50, "MPI_ERR_SESSION">;
def MPI_CodeErrSize : I32EnumAttrCase<"MPI_ERR_SIZE", 51, "MPI_ERR_SIZE">;
def MPI_CodeErrSpawn : I32EnumAttrCase<"MPI_ERR_SPAWN", 52, "MPI_ERR_SPAWN">;
def MPI_CodeErrTag : I32EnumAttrCase<"MPI_ERR_TAG", 53, "MPI_ERR_TAG">;
def MPI_CodeErrTopology
    : I32EnumAttrCase<"MPI_ERR_TOPOLOGY", 54, "MPI_ERR_TOPOLOGY">;
def MPI_CodeErrTruncate
    : I32EnumAttrCase<"MPI_ERR_TRUNCATE", 55, "MPI_ERR_TRUNCATE">;
def MPI_CodeErrType : I32EnumAttrCase<"MPI_ERR_TYPE", 56, "MPI_ERR_TYPE">;
def MPI_CodeErrUnknown
    : I32EnumAttrCase<"MPI_ERR_UNKNOWN", 57, "MPI_ERR_UNKNOWN">;
def MPI_CodeErrUnsupportedDatarep
    : I32EnumAttrCase<"MPI_ERR_UNSUPPORTED_DATAREP", 58,
                      "MPI_ERR_UNSUPPORTED_DATAREP">;
def MPI_CodeErrUnsupportedOperation
    : I32EnumAttrCase<"MPI_ERR_UNSUPPORTED_OPERATION", 59,
                      "MPI_ERR_UNSUPPORTED_OPERATION">;
def MPI_CodeErrValueTooLarge
    : I32EnumAttrCase<"MPI_ERR_VALUE_TOO_LARGE", 60, "MPI_ERR_VALUE_TOO_LARGE">;
def MPI_CodeErrWin : I32EnumAttrCase<"MPI_ERR_WIN", 61, "MPI_ERR_WIN">;
def MPI_CodeErrLastcode
    : I32EnumAttrCase<"MPI_ERR_LASTCODE", 62, "MPI_ERR_LASTCODE">;

def MPI_ErrorClassEnum
    : I32EnumAttr<"MPI_ErrorClassEnum", "MPI error class name", [
      MPI_CodeSuccess,
      MPI_CodeErrAccess,
      MPI_CodeErrAmode,
      MPI_CodeErrArg,
      MPI_CodeErrAssert,
      MPI_CodeErrBadFile,
      MPI_CodeErrBase,
      MPI_CodeErrBuffer,
      MPI_CodeErrComm,
      MPI_CodeErrConversion,
      MPI_CodeErrCount,
      MPI_CodeErrDims,
      MPI_CodeErrDisp,
      MPI_CodeErrDupDatarep,
      MPI_CodeErrErrhandler,
      MPI_CodeErrFile,
      MPI_CodeErrFileExists,
      MPI_CodeErrFileInUse,
      MPI_CodeErrGroup,
      MPI_CodeErrInfo,
      MPI_CodeErrInfoKey,
      MPI_CodeErrInfoNokey,
      MPI_CodeErrInfoValue,
      MPI_CodeErrInStatus,
      MPI_CodeErrIntern,
      MPI_CodeErrIo,
      MPI_CodeErrKeyval,
      MPI_CodeErrLocktype,
      MPI_CodeErrName,
      MPI_CodeErrNoMem,
      MPI_CodeErrNoSpace,
      MPI_CodeErrNoSuchFile,
      MPI_CodeErrNotSame,
      MPI_CodeErrOp,
      MPI_CodeErrOther,
      MPI_CodeErrPending,
      MPI_CodeErrPort,
      MPI_CodeErrProcAborted,
      MPI_CodeErrQuota,
      MPI_CodeErrRank,
      MPI_CodeErrReadOnly,
      MPI_CodeErrRequest,
      MPI_CodeErrRmaAttach,
      MPI_CodeErrRmaConflict,
      MPI_CodeErrRmaFlavor,
      MPI_CodeErrRmaRange,
      MPI_CodeErrRmaShared,
      MPI_CodeErrRmaSync,
      MPI_CodeErrRoot,
      MPI_CodeErrService,
      MPI_CodeErrSession,
      MPI_CodeErrSize,
      MPI_CodeErrSpawn,
      MPI_CodeErrTag,
      MPI_CodeErrTopology,
      MPI_CodeErrTruncate,
      MPI_CodeErrType,
      MPI_CodeErrUnknown,
      MPI_CodeErrUnsupportedDatarep,
      MPI_CodeErrUnsupportedOperation,
      MPI_CodeErrValueTooLarge,
      MPI_CodeErrWin,
      MPI_CodeErrLastcode
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mpi";
}

def MPI_ErrorClassAttr : EnumAttr<MPI_Dialect, MPI_ErrorClassEnum, "errclass"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // MLIR_DIALECT_MPI_IR_MPI_TD
