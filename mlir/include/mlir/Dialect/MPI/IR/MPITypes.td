//===- MPITypes.td - Message Passing Interface types -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Message Passing Interface dialect types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MPI_IR_MPITYPES_TD
#define MLIR_DIALECT_MPI_IR_MPITYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/MPI/IR/MPI.td"

//===----------------------------------------------------------------------===//
// MPI Types
//===----------------------------------------------------------------------===//

class MPI_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MPI_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// mpi::RetvalType
//===----------------------------------------------------------------------===//

def MPI_Retval : MPI_Type<"Retval", "retval"> {
  let summary = "MPI function call return value";
  let description = [{
    This type represents a return value from an MPI function call.
    This value can be MPI_SUCCESS, MPI_ERR_IN_STATUS, or any error code.

    This return value can be compared agains the known MPI error classes
    represented by `#mpi.errclass` using the `mpi.retval_check` operation.
  }];
}

#endif // MLIR_DIALECT_MPI_IR_MPITYPES_TD
