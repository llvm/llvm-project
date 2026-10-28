//===-- AIR.td - AIR dialect definition --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIR_OPS
#define AIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def AIR_Dialect : Dialect {
  let name = "air";
  let cppNamespace = "::mlir::air";
  let summary = "The AIR dialect that targets the Apple GPU (Metal/AIR) "
                "LLVM backend.";
  let description = [{
    The AIR dialect owns the `#air.target` GPU compilation target attribute,
    making the Apple GPU (Apple Intermediate Representation / Metal) LLVM
    backend a first-class `gpu` dialect target, and the operations of AIR
    that the `gpu` dialect cannot express: SIMD-group matrices.
    `convert-gpu-to-air` lowers them, with the `gpu` dialect, to calls of
    AIR's functions.
  }];
  let useDefaultAttributePrinterParser = 1;

}

class AIR_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AIR_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// AIR target attribute.
//===----------------------------------------------------------------------===//

def AIR_TargetAttr : AIR_Attr<"AIRTarget", "target"> {
  let description = [{
    GPU target attribute for controlling compilation of Apple GPU targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values: a `.metallib` for Apple's Metal driver.
    ```
      gpu.module @mymodule [#air.target] attributes {...} {
        ...
      }
    ```

    2. An AGX chip (an LLVM AGX processor, such as `agx-g14g`) as `chip`: an
       AGX object, compiled from the module's AIR by the AGX backend
       (agx-lower-air), with the AGX target features `features`.
    ```
      gpu.module @mymodule [#air.target<chip = "agx-g14g">,
                            #air.target<chip = "agx-g13g", features = "+agx-strict">]
    ```
  }];
  let parameters =
      (ins DefaultValuedParameter<"int", "3",
                                  "Optimization level to apply.">:$O,
          StringRefParameter<"Target triple.",
                             "\"air64_v28-apple-macosx26.0.0\"">:$triple,
          StringRefParameter<"Target chip.", "\"\"">:$chip,
          StringRefParameter<"Target features (AGX chips).", "\"\"">:$features);
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features)^ `>`)?
  }];
  let builders = [AttrBuilder<
      (ins CArg<"int", "3">:$optLevel,
          CArg<"::llvm::StringRef", "\"air64_v28-apple-macosx26.0.0\"">:$triple,
          CArg<"::llvm::StringRef", "\"\"">:$chip,
          CArg<"::llvm::StringRef", "\"\"">:$features),
      [{
      return Base::get($_ctxt, optLevel, triple, chip, features);
    }]>];
  let extraClassDeclaration = [{
    /// Whether the chip is an AGX processor: the target is then an AGX object.
    bool isAGX() const { return getChip().starts_with("agx-"); }
  }];
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// SIMD-group matrices.
//===----------------------------------------------------------------------===//

class AIR_Op<string mnemonic, list<Trait> traits = []>
    : Op<AIR_Dialect, mnemonic, traits>;

// An 8x8 SIMD-group matrix: a value shared by the 32 threads of a SIMD-group,
// whose elements are not individually addressable.
def AIR_Matrix : FixedVectorOfLengthAndType<[64], [F16, BF16, F32]>;

def AIR_MatrixLoadOp : AIR_Op<"simdgroup_matrix_load"> {
  let summary = "Load an 8x8 SIMD-group matrix";
  let description = [{
    Loads the 8x8 matrix whose element (r, c) is `base[i + r, j + c]`, or
    `base[i + c, j + r]` with `transpose`, from a 2-D memref (any strides).
    Every thread of the SIMD-group must execute it with the same operands.
    ```mlir
    %m = air.simdgroup_matrix_load %A[%i, %j] : memref<?x?xf32, strided<[?, 1]>, 1> -> vector<64xf32>
    ```
  }];
  let arguments = (ins Arg<MemRefRankOf<[F16, BF16, F32], [2]>, "",
                           [MemRead]>:$base,
                   Variadic<Index>:$indices, UnitAttr:$transpose);
  let results = (outs AIR_Matrix:$result);
  let assemblyFormat = [{
    $base `[` $indices `]` (`transpose` $transpose^)? attr-dict `:`
    type($base) `->` type($result)
  }];
  let hasVerifier = 1;
}

def AIR_MatrixStoreOp : AIR_Op<"simdgroup_matrix_store"> {
  let summary = "Store an 8x8 SIMD-group matrix";
  let description = [{
    Stores the matrix to `base[i + r, j + c]` (`base[i + c, j + r]` with
    `transpose`), as `air.simdgroup_matrix_load` reads it.
  }];
  let arguments = (ins AIR_Matrix:$value,
                   Arg<MemRefRankOf<[F16, BF16, F32], [2]>, "",
                       [MemWrite]>:$base,
                   Variadic<Index>:$indices, UnitAttr:$transpose);
  let assemblyFormat = [{
    $value `,` $base `[` $indices `]` (`transpose` $transpose^)? attr-dict
    `:` type($value) `->` type($base)
  }];
  let hasVerifier = 1;
}

def AIR_MatrixMultiplyAccumulateOp
    : AIR_Op<"simdgroup_matrix_multiply_accumulate",
             [Pure, AllTypesMatch<["a", "b"]>,
              AllTypesMatch<["c", "result"]>]> {
  let summary = "D = A * B + C on 8x8 SIMD-group matrices";
  let description = [{
    A and B have one element type and C and D another: the same, or f16 or
    bf16 A and B with f32 C and D (products and sums in f32).
    ```mlir
    %d = air.simdgroup_matrix_multiply_accumulate %a, %b, %c : vector<64xf32>
    %e = air.simdgroup_matrix_multiply_accumulate %h, %h, %d : vector<64xf16>, vector<64xf32>
    ```
  }];
  let arguments = (ins AIR_Matrix:$a, AIR_Matrix:$b, AIR_Matrix:$c);
  let results = (outs AIR_Matrix:$result);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` custom<MatrixMMATypes>(type($a), type($c))
  }];
  let builders = [OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c), [{
    build($_builder, $_state, c.getType(), a, b, c);
  }]>];
  let hasVerifier = 1;
}

def AIR_MatrixFillOp
    : AIR_Op<"simdgroup_matrix_fill", [Pure,
             TypesMatchWith<"the value is an element", "result", "value",
                            "::llvm::cast<VectorType>($_self).getElementType()">]> {
  let summary = "An 8x8 SIMD-group matrix of one value";
  let arguments = (ins AnyTypeOf<[F16, BF16, F32]>:$value);
  let results = (outs AIR_Matrix:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

#endif // AIR_OPS
