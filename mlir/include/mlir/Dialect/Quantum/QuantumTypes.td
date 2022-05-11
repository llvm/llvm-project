//===- QuantumTypes.td - Quantum dialect Types -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_TYPES
#define QUANTUM_TYPES

include "mlir/IR/OpBase.td"
include "Dialect/Quantum/QuantumDialect.td"

//===----------------------------------------------------------------------===//
// Qubit Type
//===----------------------------------------------------------------------===//

def Quantum_QubitType : Quantum_Type<"Qubit"> {
  let mnemonic = "qubit";
  let summary = "Quantum qubit array type";
  let parameters = (ins "int64_t":$size);
  let printer = [{ ::print(*this, $_printer); }];
  let parser = [{ return ::parseQubit($_parser, $_ctxt); }];
  let extraClassDeclaration = [{
    bool hasStaticSize() const;
    bool isSingleQubit() const;
  }];
}

class QubitSizePred<int N> : CPred<"$_self.cast<QubitType>().getSize() == " # N>;

class QubitArrayOfSize<int N>
  : Type<And<[Quantum_QubitType.predicate, QubitSizePred<N>]>,
         "Quantum " # N # "-Qubit Type">,
    BuildableType<"QubitType::get($_builder.getContext(), " # N # ")">;

def Quantum_SingleQubitType : QubitArrayOfSize<1>;

#endif // QUANTUM_TYPES
