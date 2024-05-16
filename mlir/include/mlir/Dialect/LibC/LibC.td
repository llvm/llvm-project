//===-- AMX.td - AMX dialect operation definitions *- tablegen -*----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the basic operations for the libc dialect.
//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LIBC_LIBC
#define MLIR_DIALECT_LIBC_LIBC

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// LibC dialect definition.
//===----------------------------------------------------------------------===//

def LibC_Dialect : Dialect {
  let name = "libc";
  let cppNamespace = "::mlir::libc";
  let description = [{
 The dialect provides interfaces for the standard C library functions, to
 enable users to call them in CPU environments if a valid standard C library
 (like glibc) is linked. The dialect's ops are designed to be lowered to
 function calls to the corresponding LibC functions.
  }];
}

class LibC_Op<string mnemonic, list<Trait> traits = []> :
  Op<LibC_Dialect, mnemonic, traits> {}

def LibC_PrintfOp : LibC_Op<"printf", [MemoryEffects<[MemWrite]>]>,
  Arguments<(ins StrAttr:$format,
                Variadic<AnyTypeOf<[AnyInteger, Index, AnyFloat]>>:$args)> {
  let summary = "C-style printf";
  let description = [{
    `cpuruntime.printf` takes a literal format string `format` and an arbitrary
    number of scalar arguments that should be printed.

    The format string is a C-style printf string, subject to any restrictions
    imposed by the target platform.
  }];
  let assemblyFormat = [{
    $format attr-dict ($args^ `:` type($args))?
  }];
}


#endif // MLIR_DIALECT_LIBC_LIBC
