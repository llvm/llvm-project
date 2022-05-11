//===- QuantumOps.td - Quantum dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUM_OPS
#define QUANTUM_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "Dialect/Quantum/QuantumDialect.td"
include "Dialect/Quantum/QuantumTypes.td"

//===----------------------------------------------------------------------===//
// Allocate Op
//===----------------------------------------------------------------------===//

def Quantum_AllocateOp : Quantum_Op<"alloc", [MemoryEffects<[MemRead]>]> {
    let summary = "Allocate an array of qubits.";
    let description = [{
        This operation takes no arguments,
        and returns a qubit array `qubit<N>`.

        Example:

        ```mlir
        // alloc a fixed number of qubits
        %q0 = qssa.alloc : !qssa.qubit<10>

        // alloc a variable number of qubits
        %n = constant 10 : index
        %q1 = qssa.alloc(%n) : !qssa.qubit<?>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicsize);
    let results = (outs Quantum_QubitType:$qout);

    let builders = [
      OpBuilder<(ins "int":$size), [{
        return build($_builder, $_state, ::mlir::quantum::QubitType::get($_builder.getContext(), size), {});
      }]>
    ];

    let verifier = ?;
    let assemblyFormat = [{ ( `(` $dynamicsize^ `)` )? attr-dict `:` type($qout) }];
}

//===----------------------------------------------------------------------===//
// Dimension Op
//===----------------------------------------------------------------------===//

def Quantum_DimensionOp : Quantum_Op<"dim", [NoSideEffect, AllTypesMatch<["qinp", "qout"]>]> {
    let summary = "Get the dimension of a qubit array";
    let description = [{
        This operation takes a qubit array
        and returns the size as an IndexType

        Example:

        ```mlir
        func @use_dim(%qs: !qssa.qubit<?>) : !qssa.qubit<?> {
            %qs1, %n = qssa.dim %qs : !qssa.qubit<?>
            // use %n
            return %qs1 : !qssa.qubit<?>
        }
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_QubitType:$qout, Index:$size);
    let assemblyFormat = [{ $qinp attr-dict `:` type($qinp) }];
    let verifier = ?;
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Cast Op
//===----------------------------------------------------------------------===//

def Quantum_CastOp : Quantum_Op<"cast", [NoSideEffect]> {
    let summary = " an array of qubits.";
    let description = [{
        Convert a qubit array to an equivalent type.
        Takes one argument, and returns the converted qubit array.
        Useful for converting to variable length arrays,
        to pass to library functions etc.

        Example:

        ```mlir
        %0 = quantum.allocate() : !quantum.qubit<10>
        %1 = quantum.cast %0 : !quantum.qubit<10> to !quantum.qubit<?>
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_QubitType:$qout);

    let assemblyFormat = [{ $qinp attr-dict `:` type($qinp) `to` type($qout) }];
    let extraClassDeclaration = [{
        /// Return true if `a` and `b` are valid operand and result pairs for
        /// the operation.
        static bool areCastCompatible(Type a, Type b) {
            QubitType qa = a.cast<QubitType>();
            QubitType qb = b.cast<QubitType>();

            if (qa.hasStaticSize() && qb.hasStaticSize() && qa.getSize() != qb.getSize()) {
                return false;
            }
            return true;
        }
    }];

    let verifier = [{
        if (!areCastCompatible(qinp().getType(), qout().getType()))
            return emitOpError("Incompatible types for cast.");
        return success();
    }];
}

//===----------------------------------------------------------------------===//
// Concat Op
//===----------------------------------------------------------------------===//

def Quantum_ConcatOp : Quantum_Op<"concat", [NoSideEffect]> {
    let summary = "Concatenate two arrays of qubits.";
    let description = [{
        This operation takes two qubit arrays and concatenates them into one.

        Example:

        ```mlir
        %q1 = quantum.allocate() : !quantum.qubit<4>
        %q2 = quantum.allocate() : !quantum.qubit<5>
        %qall = quantum.concat %q1, %q2 : (!quantum.qubit<4>, !quantum.qubit<5>) -> !quantum.qubit<9>
        ```
    }];

    let arguments = (ins Variadic<Quantum_QubitType>:$qinps);
    let results = (outs Quantum_QubitType:$qout);
    let assemblyFormat = [{ $qinps attr-dict `:` functional-type($qinps, $qout) }];
    let verifier = ?;
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Split Op
//===----------------------------------------------------------------------===//

def Quantum_SplitOp : Quantum_Op<"split", [NoSideEffect]> {
    let summary = "Split an array of qubits into two disjoint arrays";
    let description = [{
        This operation takes a qubit array, and splits it into two arrays.
        For every dynamic result type, it takes a mandatory index type size argument.

        Examples:

        ```mlir
        func @try_split(%q : !quantum.qubit<10>) {
          %q1, %q2 = quantum.split %q : !quantum.qubit<10> -> (!quantum.qubit<5>, !quantum.qubit<5>)
          %1 = constant 1 : index
          %2 = constant 2 : index
          %q3:3 = quantum.split %q1[%1, %2, %2] : !quantum.qubit<5> -> (!quantum.qubit<?>, !quantum.qubit<?>, !quantum.qubit<?>)
          return
        }
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp, Variadic<Index>:$sizes);
    let results = (outs Variadic<Quantum_QubitType>:$qouts);
    let assemblyFormat = [{ $qinp (`[` $sizes^ `]`)? attr-dict `:` functional-type($qinp, $qouts) }];
    let verifier = ?;
    let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Measure Op
//===----------------------------------------------------------------------===//


def Quantum_MeasureQubitOp : Quantum_Op<"measure_one", [NoSideEffect]> {
    let summary = "Measure a single qubit.";
    let description = [{
        This operation takes a single qubit of type `qubit<1>`
        and returns the measurement outcome in the standard basis
        as a bit (`i1`)

        Note: The qubit is consumed and can no longer be used.

        Example:

        ```mlir
        %q = qssa.allocate() : !qssa.qubit<1>
        %mem = memref.alloc() : memref<10xi1>
        %res = qssa.measure_one %q
        ```
    }];

    let arguments = (ins Quantum_SingleQubitType:$qinp);
    let results = (outs I1:$res, Quantum_SingleQubitType:$qout);

    let assemblyFormat = [{ $qinp attr-dict }];
    let builders = [
      OpBuilder<(ins "::mlir::Value":$inp), [{
        return build($_builder, $_state,
                     ::mlir::IntegerType::get($_builder.getContext(), 1),
                     inp.getType(),
                     inp);
      }]>
    ];
    let verifier = ?;
}

def Quantum_MeasurementType : 1DTensorOf<[I1]>;

def Quantum_MeasureOp : Quantum_Op<"measure", [NoSideEffect, AllTypesMatch<["qinp", "qout"]>]> {
    let summary = "Measure an array of qubits.";
    let description = [{
        This operation takes an array of qubits of type `qubit<N>`
        and returns the measurement outcome in the standard basis
        as a 1-D tensor of bits (`i1`) and the final qubits.

        Note: The qubits are consumed and can no longer be used.

        Example:

        ```mlir
        %qs = qssa.allocate : !qssa.qubit<10>
        %res, %qs' = qssa.measure %qs : !qssa.qubit<10> -> tensor<10xi1>
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let results = (outs Quantum_MeasurementType:$res, Quantum_QubitType:$qout);

    let assemblyFormat = [{ $qinp attr-dict `:` type($qinp) `->` type($res) }];
}

def Quantum_SinkOp : Quantum_Op<"sink"> {
    let summary = "Sink an array of qubits.";
    let description = [{
      Use this operation to sink qubits without storing measurement results.
      This helps prevent over-optimization which removes all the transforms

        Note: The qubits are consumed and can no longer be used.

        Example:

        ```mlir
        %qs = qssa.allocate : !qssa.qubit<10>
        qssa.sink %qs : !qssa.qubit<10>
        ```
    }];

    let arguments = (ins Quantum_QubitType:$qinp);
    let assemblyFormat = [{ $qinp attr-dict `:` type($qinp) }];
    let verifier = ?;
}

def Quantum_BarrierOp : Quantum_Op<"barrier", [NoSideEffect, AllTypesMatch<["qinp", "qout"]>]> {
  let summary = "Place a barrier for optimization";
  let description = [{
    Places a barrier, preventing optimizations to move components across it.

      Example:

      ```mlir
      // Does NOT fold the two Xs away
      %0 = qssa.allocate : !qssa.qubit<10>
      %1 = qssa.X %0 : !qssa.qubit<10>
      %2 = qssa.barrier %1 : !qssa.qubit<10>
      %3 = qssa.X %2 : !qssa.qubit<10>
      qssa.sink %3
      ```
  }];

  let arguments = (ins Quantum_QubitType:$qinp);
  let results = (outs Quantum_QubitType:$qout);
  let assemblyFormat = [{ $qinp attr-dict `:` type($qout) }];
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// Primitive Gate Ops
//===----------------------------------------------------------------------===//

// Base Class for parameterless gate ops
//
//   %1 = qssa.<gate> %0 : !qssa.qubit<?>
//
class SimplePrimitiveGateOp<string mnemonic, list<OpTrait> traits = []>
  : Quantum_Op<mnemonic, !listconcat([NoSideEffect, AllTypesMatch<["qinp", "qout"]>], traits)> {
  let arguments = (ins Quantum_QubitType:$qinp);
  let results = (outs Quantum_QubitType:$qout);
  let assemblyFormat = [{ $qinp attr-dict `:` type($qinp) }];
  let extraClassDeclaration = [{
    int64_t numQubits() {
      auto qubitType = qinp().getType().cast<QubitType>();
      if (qubitType.hasStaticSize())
        return qubitType.getSize();
      return -1;
    }
  }];
  let builders = [
    OpBuilder<(ins "Value":$qinp_), [{
      return build($_builder, $_state, qinp_.getType(), qinp_);
    }]>
  ];
  let verifier = ?;
}

// Base Class for single parameter gate ops
//
//   %1 = qssa.<gate> (%a : f32) %0 : !qssa.qubit<?>
//
class PrimitiveSingleParamGateOp<string mnemonic, list<OpTrait> traits = []>
  : Quantum_Op<mnemonic, !listconcat([NoSideEffect, AllTypesMatch<["qinp", "qout"]>], traits)> {
  let arguments = (ins AnyFloat:$param, Quantum_QubitType:$qinp);
  let results = (outs Quantum_QubitType:$qout);
  let assemblyFormat = [{ `(` $param `:` type($param) `)`$qinp attr-dict `:` type($qinp) }];
  let extraClassDeclaration = [{
    int64_t numQubits() {
      auto qubitType = qinp().getType().cast<QubitType>();
      if (qubitType.hasStaticSize())
        return qubitType.getSize();
      return -1;
    }
  }];
  let builders = [
    OpBuilder<(ins "Value":$param_, "Value":$qinp_), [{
      return build($_builder, $_state, qinp_.getType(), param_, qinp_);
    }]>
  ];
  let verifier = ?;
}

//===----------------------------------------------------------------------===//
// Standard Quantum Gates
//===----------------------------------------------------------------------===//

def PauliXGateOp : SimplePrimitiveGateOp<"X"> {
  let summary = "Quantum Pauli X gate";
  let description = [{
      Takes a qubit array, and applies the Pauli X gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.pauliX %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.pauliX %q0 : !quantum.qubit<?>
      ```
  }];
  let hasFolder = 1;
}

def PauliYGateOp : SimplePrimitiveGateOp<"Y"> {
  let summary = "Quantum Pauli X gate";
  let description = [{
      Takes a qubit array, and applies the Pauli Y gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.pauliY %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.pauliY %q0 : !quantum.qubit<?>
      ```
  }];
  let hasFolder = 1;
}

def PauliZGateOp : SimplePrimitiveGateOp<"Z"> {
  let summary = "Quantum Pauli X gate";
  let description = [{
      Takes a qubit array, and applies the Pauli Z gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.pauliZ %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.pauliZ %q0 : !quantum.qubit<?>
      ```
  }];
  let hasFolder = 1;
}

def HadamardGateOp : SimplePrimitiveGateOp<"H"> {
  let summary = "Quantum Hadamard gate";
  let description = [{
      Takes a qubit array, and applies the Hadamard gate to each qubit.

      Example:
      ```mlir
      // fixed length
      %q1 = quantum.H %q0 : !quantum.qubit<10>

      // variable length
      %q1 = quantum.H %q0 : !quantum.qubit<?>
      ```
  }];
  let hasFolder = 1;
}

def PhaseGateOp : SimplePrimitiveGateOp<"S"> {
  let summary = "Phase (S) gate";
}
def PhaseDaggerGateOp : SimplePrimitiveGateOp<"Sdg"> {
  let summary = "Phase inverse (S-dagger) gate";
}
def TGateOp : SimplePrimitiveGateOp<"T"> {
  let summary = "T gate";
}
def TDaggerGateOp : SimplePrimitiveGateOp<"Tdg"> {
  let summary = "T-dagger gate";
}

def RotateXOp : PrimitiveSingleParamGateOp<"Rx"> {
  let summary = "Rotate about X axis by the given angle.";
}
def RotateYOp : PrimitiveSingleParamGateOp<"Ry"> {
  let summary = "Rotate about Y axis by the given angle.";
}
def RotateZOp : PrimitiveSingleParamGateOp<"Rz"> {
  let summary = "Rotate about Z axis by the given angle.";
}

def CNOTGateOp : Quantum_Op<"CNOT", [NoSideEffect]> {
  let summary = "Quantum CNOT gate";
  let description = [{
      Takes two 1-qubit array, and applies the CNOT gate,
      with the first qubit as control, and second as target.

      Example:
      ```mlir
      %qa1, %qb1 = quantum.CNOT %qa0, %qb0
      ```
  }];

  let arguments = (ins
                    Quantum_SingleQubitType:$qinp_cont,
                    Quantum_SingleQubitType:$qinp_targ);
  let results = (outs
                  Quantum_SingleQubitType:$qout_cont,
                  Quantum_SingleQubitType:$qout_targ);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$cont, "::mlir::Value":$targ), [{
      build($_builder, $_state, cont.getType(), targ.getType(), cont, targ);
    }]>
  ];
  let assemblyFormat = [{ $qinp_cont `,` $qinp_targ attr-dict }];
  let verifier = ?;
  let hasFolder = 1;
}

def UniversalRotationGateOp : Quantum_Op<"U", [NoSideEffect, AllTypesMatch<["qinp", "qout"]>]> {
  let summary = "Quantum Universal Single Qubit Rotation Op";
  let description = [{
    Takes a single qubit and rotates it based on the parameters:
    Example:
    ```mlir
    %q1 = qssa.U (%theta: f32, %phi: f32, %lambda: f32) %q0 : !qssa.qubit<1>
    ```
  }];

  let arguments = (ins AnyFloat:$theta,
                       AnyFloat:$phi,
                       AnyFloat:$lambda,
                       Quantum_QubitType:$qinp);
  let results = (outs Quantum_QubitType:$qout);
  let assemblyFormat = [{
    `(` $theta `:` type($theta) `,` $phi `:` type($phi) `,` $lambda `:` type($lambda) `)`
    $qinp attr-dict `:` type($qinp)
  }];
  let verifier = [{ return success(); }];
}

//===----------------------------------------------------------------------===//
// Unregistered Quantum Gates
//===----------------------------------------------------------------------===//

// TODO Support for generic gate ops, which can be modified and replaced
//      by IR tranformations.

//===----------------------------------------------------------------------===//
// Support Ops
//===----------------------------------------------------------------------===//
def CanonicalizeSupportOp : Quantum_Op<"canon"> {
  let summary = "Op to register extra canonicalization patterns. DO NOT USE";
  let assemblyFormat = "attr-dict";
  let verifier = ?;
  let hasCanonicalizer = 1;
}

#endif // QUANTUM_OPS
