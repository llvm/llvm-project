//===- MPIops.td - Message Passing Interface Ops -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MPI_MLIR_IR_MPIOPS_TD
#define MPI_MLIR_IR_MPIOPS_TD

include "mlir/Dialect/MPI/IR/MPI.td"
include "mlir/Dialect/MPI/IR/MPITypes.td"

class MPI_Op<string mnemonic, list<Trait> traits = []>
    : Op<MPI_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// InitOp
//===----------------------------------------------------------------------===//

def MPI_InitOp : MPI_Op<"init", []> {
  let summary =
      "Initialize the MPI library, equivalent to `MPI_Init(NULL, NULL)`";
  let description = [{
    This operation must preceed most MPI calls (except for very few exceptions,
    please consult with the MPI specification on these).

    Passing &argc, &argv is not supported currently.

    This operation can optionally return an `!mpi.retval` value that can be used
    to check for errors.
  }];

  let results = (outs Optional<MPI_Retval>:$retval);

  let assemblyFormat = "attr-dict (`:` type($retval)^)?";
}

//===----------------------------------------------------------------------===//
// CommRankOp
//===----------------------------------------------------------------------===//

def MPI_CommRankOp : MPI_Op<"comm_rank", []> {
  let summary = "Get the current rank, equivalent to "
                "`MPI_Comm_rank(MPI_COMM_WORLD, &rank)`";
  let description = [{
    Communicators other than `MPI_COMM_WORLD` are not supported for now.

    This operation can optionally return an `!mpi.retval` value that can be used
    to check for errors.
  }];

  let results = (
    outs Optional<MPI_Retval> : $retval,
    I32 : $rank
  );

  let assemblyFormat = "attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// SendOp
//===----------------------------------------------------------------------===//

def MPI_SendOp : MPI_Op<"send", []> {
  let summary =
      "Equivalent to `MPI_Send(ptr, size, dtype, dest, tag, MPI_COMM_WORLD)`";
  let description = [{
    MPI_Send performs a blocking send of `size` elements of type `dtype` to rank
    `dest`. The `tag` value and communicator enables the library to determine 
    the matching of multiple sends and receives between the same ranks.

    Communicators other than `MPI_COMM_WORLD` are not supprted for now.

    This operation can optionally return an `!mpi.retval` value that can be used
    to check for errors.
  }];

  let arguments = (ins AnyMemRef : $ref, I32 : $tag, I32 : $rank);

  let results = (outs Optional<MPI_Retval>:$retval);

  let assemblyFormat = "`(` $ref `,` $tag `,` $rank `)` attr-dict `:` "
                       "type($ref) `,` type($tag) `,` type($rank)"
                       "(`->` type($retval)^)?";
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// RecvOp
//===----------------------------------------------------------------------===//

def MPI_RecvOp : MPI_Op<"recv", []> {
  let summary = "Equivalent to `MPI_Recv(ptr, size, dtype, dest, tag, "
                "MPI_COMM_WORLD, MPI_STATUS_IGNORE)`";
  let description = [{
    MPI_Recv performs a blocking receive of `size` elements of type `dtype` 
    from rank `dest`. The `tag` value and communicator enables the library to 
    determine the matching of multiple sends and receives between the same 
    ranks.

    Communicators other than `MPI_COMM_WORLD` are not supprted for now.
    The MPI_Status is set to `MPI_STATUS_IGNORE`, as the status object 
    is not yet ported to MLIR.

    This operation can optionally return an `!mpi.retval` value that can be used
    to check for errors.
  }];

  let arguments = (ins AnyMemRef : $ref, I32 : $tag, I32 : $rank);

  let results = (outs Optional<MPI_Retval>:$retval);

  let assemblyFormat = "`(` $ref `,` $tag `,` $rank `)` attr-dict `:` "
                       "type($ref) `,` type($tag) `,` type($rank)"
                       "(`->` type($retval)^)?";
  let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// FinalizeOp
//===----------------------------------------------------------------------===//

def MPI_FinalizeOp : MPI_Op<"finalize", []> {
  let summary = "Finalize the MPI library, equivalent to `MPI_Finalize()`";
  let description = [{
    This function cleans up the MPI state. Afterwards, no MPI methods may 
    be invoked (excpet for MPI_Get_version, MPI_Initialized, and MPI_Finalized).
    Notably, MPI_Init cannot be called again in the same program.

    This operation can optionally return an `!mpi.retval` value that can be used
    to check for errors.
  }];

  let results = (outs Optional<MPI_Retval>:$retval);

  let assemblyFormat = "attr-dict (`:` type($retval)^)?";
}


//===----------------------------------------------------------------------===//
// RetvalCheckOp
//===----------------------------------------------------------------------===//

def MPI_RetvalCheckOp : MPI_Op<"retval_check", []> {
  let summary = "Check an MPI return value against an error class";
  let description = [{
    This operation compares MPI status codes to known error class
    constants such as `MPI_SUCCESS`, or `MPI_ERR_COMM`.
  }];

  let arguments = (
    ins MPI_Retval:$val,
    MPI_ErrorClassAttr:$errclass
  );

  let results = (
    outs I1:$res
  );

  let assemblyFormat = "$val `=` $errclass attr-dict `:` type($res)";
}



//===----------------------------------------------------------------------===//
// RetvalCheckOp
//===----------------------------------------------------------------------===//

def MPI_ErrorClassOp : MPI_Op<"error_class", []> {
  let summary = "Get the error class from an error code, equivalent to "
                "the `MPI_Error_class` function";
  let description = [{
    `MPI_Error_class` maps return values from MPI calls to a set of well-known
    MPI error classes.
  }];

  let arguments = (
    ins MPI_Retval:$val
  );

  let results = (
    outs MPI_Retval:$errclass
  );

  let assemblyFormat = "$val attr-dict `:` type($val)";
}

#endif // MPI_MLIR_IR_MPIOPS_TD
