//===- QuantumDialect.td - Quantum dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_DIALECT
#define QUANTUM_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Quantum dialect definition.
//===----------------------------------------------------------------------===//

def Quantum_Dialect : Dialect {
    let name = "qssa";
    let summary = "A quantum MLIR dialect.";
    let description = [{
        A WIP Quantum dialect for MLIR. Attempts to gather ideas from current 
        languages like Q#, Silq, Cirq, Qiskit etc. and build a sufficiently
        general-purpose dialect.
    }];
    let cppNamespace = "::mlir::quantum";
}

//===----------------------------------------------------------------------===//
// Base quantum operation definition.
//===----------------------------------------------------------------------===//

class Quantum_Op<string mnemonic, list<OpTrait> traits = []> :
        Op<Quantum_Dialect, mnemonic, traits> {

    // For each Quantum op, the following static functions need to be defined in
    // QuantumOps.cpp:
    //
    // * static ParseResult parse<op-c++-class-name>(OpAsmParser &parser,
    //                                               OperationState &state);
    // * static void print(OpAsmPrinter &p, <op-c++-class-name> op)
    // * static LogicalResult verify(<op-c++-class-name> op)
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ ::print(*this, p); }];
    let verifier = [{ return ::verify(*this); }];
}

class Quantum_Type<string name> : TypeDef<Quantum_Dialect, name> {}

#endif // QUANTUM_DIALECT
