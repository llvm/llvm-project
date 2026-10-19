//===-- AGXIROps.td - AGXIR dialect operation definitions --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The AGXIR dialect wraps the llvm.agx.* intrinsics of the experimental AGX
// (Apple GPU) LLVM backend, like ROCDL wraps the AMDGPU intrinsics.
//
//===----------------------------------------------------------------------===//

#ifndef AGXIR_OPS
#define AGXIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AGXIR dialect definition
//===----------------------------------------------------------------------===//

def AGXIR_Dialect : Dialect {
  let name = "agxir";
  let cppNamespace = "::mlir::AGXIR";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let summary = "Dialect for wrapping LLVM AGX backend intrinsics";
  let hasOperationAttrVerify = 1;

  let description = [{
    The AGXIR dialect contains 1:1 wrappers around the `llvm.agx.*`
    intrinsics of the experimental AGX LLVM backend, which generates code for
    the GPUs of Apple silicon, and the infrastructure to compile GPU modules
    for them (`#agxir.target`). It plays the role ROCDL plays for AMDGPU; the
    higher level `agx` dialect, modelled on `amdgpu`, lowers to it.

    The semantics of the operations are those of the intrinsics, see
    `llvm/include/llvm/IR/IntrinsicsAGX.td` and `llvm/docs/AGXUsage.rst`.

    Functions with the `agxir.kernel` attribute are translated to
    `agx_kernel` functions: kernel entry points whose arguments are passed in
    uniform registers.
  }];

  let extraClassDeclaration = [{
    /// The address spaces of the AGX backend.
    static constexpr unsigned kGenericMemoryAddressSpace = 0;
    static constexpr unsigned kDeviceMemoryAddressSpace = 1;
    static constexpr unsigned kThreadgroupMemoryAddressSpace = 3;
    static constexpr unsigned kConstantMemoryAddressSpace = 4;
    static constexpr unsigned kPrivateMemoryAddressSpace = 5;
    /// The number of threads in a SIMD-group.
    static constexpr unsigned kSIMDGroupSize = 32;
  }];

  let discardableAttrs = (ins
    "::mlir::UnitAttr":$kernel
  );

  let useDefaultAttributePrinterParser = 1;
}

class AGXIR_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<AGXIR_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class AGXIR_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<AGXIR_Dialect, mnemonic, traits>;

class AGXIR_IntrOp<string mnemonic, list<int> overloadedResults,
                   list<int> overloadedOperands, list<Trait> traits,
                   int numResults, list<int> immArgPositions = [],
                   list<string> immArgAttrNames = []> :
  LLVM_IntrOpBase<AGXIR_Dialect, mnemonic,
    "agx_" # !subst(".", "_", mnemonic), overloadedResults,
    overloadedOperands, traits, numResults, 0, 0, 0, 0, 0,
    immArgPositions, immArgAttrNames>;

//===----------------------------------------------------------------------===//
// Special registers
//===----------------------------------------------------------------------===//

class AGXIR_SpecialRegisterOp<string mnemonic, list<Trait> traits = [Pure]> :
    AGXIR_IntrOp<mnemonic, [], [], traits, 1>,
    Arguments<(ins OptionalAttr<LLVM_ConstantRangeAttr>:$range)> {
  let results = (outs I32:$res);
  string llvmBuilder = baseLlvmBuilder # setRangeRetAttrCode #
                       baseLlvmBuilderCoda;
  string mlirBuilder = baseMlirBuilder # importRangeRetAttrCode #
                       baseMlirBuilderCoda;
  let assemblyFormat = "(`range` $range^)? attr-dict `:` type($res)";
  let builders = [
    OpBuilder<(ins "Type":$resultType), [{
      build($_builder, $_state, resultType, ::mlir::LLVM::ConstantRangeAttr{});
    }]>
  ];
}

multiclass AGXIR_SpecialRegisterXYZ<string mnemonic> {
  def XOp : AGXIR_SpecialRegisterOp<mnemonic # ".x">;
  def YOp : AGXIR_SpecialRegisterOp<mnemonic # ".y">;
  def ZOp : AGXIR_SpecialRegisterOp<mnemonic # ".z">;
}

defm AGXIR_ThreadPositionInGrid
    : AGXIR_SpecialRegisterXYZ<"thread.position.in.grid">;
defm AGXIR_ThreadPositionInThreadgroup
    : AGXIR_SpecialRegisterXYZ<"thread.position.in.threadgroup">;
defm AGXIR_ThreadgroupPositionInGrid
    : AGXIR_SpecialRegisterXYZ<"threadgroup.position.in.grid">;
defm AGXIR_ThreadsPerThreadgroup
    : AGXIR_SpecialRegisterXYZ<"threads.per.threadgroup">;
defm AGXIR_ThreadgroupsPerGrid
    : AGXIR_SpecialRegisterXYZ<"threadgroups.per.grid">;

def AGXIR_ThreadIndexInThreadgroupOp
    : AGXIR_SpecialRegisterOp<"thread.index.in.threadgroup">;
def AGXIR_ThreadIndexInSIMDGroupOp
    : AGXIR_SpecialRegisterOp<"thread.index.in.simdgroup">;
def AGXIR_SIMDGroupIndexInThreadgroupOp
    : AGXIR_SpecialRegisterOp<"simdgroup.index.in.threadgroup">;

// These depend on the active threads, so they cannot be moved across control
// flow.
def AGXIR_ActiveThreadIndexInSIMDGroupOp
    : AGXIR_SpecialRegisterOp<"active.thread.index.in.simdgroup", []>;
def AGXIR_TotalActiveThreadsInSIMDGroupOp
    : AGXIR_SpecialRegisterOp<"total.active.threads.in.simdgroup", []>;
def AGXIR_ActiveThreadIndexInQuadgroupOp
    : AGXIR_SpecialRegisterOp<"active.thread.index.in.quadgroup", []>;
def AGXIR_TotalActiveThreadsInQuadgroupOp
    : AGXIR_SpecialRegisterOp<"total.active.threads.in.quadgroup", []>;

def AGXIR_CoreIdOp : AGXIR_SpecialRegisterOp<"core.id", []> {
  let summary = "Index of the GPU core that runs the thread";
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def AGXIR_ThreadgroupBarrierOp : AGXIR_IntrOp<"threadgroup.barrier", [], [],
                                              [], 0> {
  let summary = "Wait for all threads of the threadgroup";
  let description = [{
    Execution barrier for the threadgroup. It does not order memory: combine
    it with a fence, or use `agxir.memory.barrier`.

    ```mlir
    agxir.threadgroup.barrier
    ```
  }];
  let assemblyFormat = "attr-dict";
}

def AGXIR_MemoryBarrierOp : AGXIR_IntrOp<"memory.barrier", [], [], [], 0,
                                         [0, 1, 2], ["a", "b", "c"]>,
    Arguments<(ins I32Attr:$a, I32Attr:$b, I32Attr:$c)> {
  let summary = "Memory barrier";
  let description = [{
    The `memory_barrier` instruction with its three fields; `1, 2, 9` orders
    device memory, as emitted for fences.

    ```mlir
    agxir.memory.barrier 1, 2, 9
    ```
  }];
  let assemblyFormat = "$a `,` $b `,` $c attr-dict";
}

def AGXIR_WaitOp : AGXIR_IntrOp<"wait", [], [], [], 0, [0], ["slot"]>,
    Arguments<(ins I32Attr:$slot)> {
  let summary = "Wait for the asynchronous operations of a scoreboard slot";
  let assemblyFormat = "$slot attr-dict";
}

//===----------------------------------------------------------------------===//
// SIMD-group operations
//===----------------------------------------------------------------------===//

class AGXIR_BallotOp<string mnemonic> :
    AGXIR_IntrOp<mnemonic, [], [], [], 1>,
    Arguments<(ins I1:$pred)> {
  let results = (outs I32:$res);
  let assemblyFormat = "$pred attr-dict `:` type($res)";
}

def AGXIR_SIMDBallotOp : AGXIR_BallotOp<"simd.ballot"> {
  let summary = "Mask of the active threads of the SIMD-group where the "
                "predicate is true";
}
def AGXIR_QuadBallotOp : AGXIR_BallotOp<"quad.ballot"> {
  let summary = "Mask of the threads of the quad where the predicate is "
                "true";
}

class AGXIR_ShuffleOp<string mnemonic> :
    AGXIR_IntrOp<mnemonic, [0], [], [AllTypesMatch<["res", "value"]>], 1>,
    Arguments<(ins AnyTypeOf<[I16, I32, F16, F32]>:$value, I32:$index)> {
  let results = (outs AnyTypeOf<[I16, I32, F16, F32]>:$res);
  let assemblyFormat = "$value `,` $index attr-dict `:` type($res)";
}

def AGXIR_SIMDShuffleOp : AGXIR_ShuffleOp<"simd.shuffle"> {
  let summary = "Read a value from another thread of the SIMD-group";
  let description = [{
    ```mlir
    %r = agxir.simd.shuffle %v, %lane : f32
    ```
  }];
}
def AGXIR_SIMDShuffleXorOp : AGXIR_ShuffleOp<"simd.shuffle.xor">;
def AGXIR_SIMDShuffleUpOp : AGXIR_ShuffleOp<"simd.shuffle.up">;
def AGXIR_SIMDShuffleDownOp : AGXIR_ShuffleOp<"simd.shuffle.down">;
def AGXIR_QuadShuffleOp : AGXIR_ShuffleOp<"quad.shuffle">;
def AGXIR_QuadShuffleXorOp : AGXIR_ShuffleOp<"quad.shuffle.xor">;
def AGXIR_QuadShuffleUpOp : AGXIR_ShuffleOp<"quad.shuffle.up">;
def AGXIR_QuadShuffleDownOp : AGXIR_ShuffleOp<"quad.shuffle.down">;

// The operation numbers of reductions and prefixes.
def AGXIR_ReduceAnd : I32EnumAttrCase<"and_", 0, "and">;
def AGXIR_ReduceFAdd : I32EnumAttrCase<"fadd", 1>;
def AGXIR_ReduceOr : I32EnumAttrCase<"or_", 2, "or">;
def AGXIR_ReduceFMul : I32EnumAttrCase<"fmul", 3>;
def AGXIR_ReduceXor : I32EnumAttrCase<"xor_", 4, "xor">;
def AGXIR_ReduceFMin : I32EnumAttrCase<"fmin", 5>;
def AGXIR_ReduceFMax : I32EnumAttrCase<"fmax", 7>;
def AGXIR_ReduceIAdd : I32EnumAttrCase<"iadd", 16>;
def AGXIR_ReduceSMin : I32EnumAttrCase<"smin", 20>;
def AGXIR_ReduceSMax : I32EnumAttrCase<"smax", 22>;
def AGXIR_ReduceUMin : I32EnumAttrCase<"umin", 28>;
def AGXIR_ReduceUMax : I32EnumAttrCase<"umax", 30>;

def AGXIR_ReduceKind : I32EnumAttr<"ReduceKind",
    "Operation of a SIMD-group or quad reduction",
    [AGXIR_ReduceAnd, AGXIR_ReduceFAdd, AGXIR_ReduceOr, AGXIR_ReduceFMul,
     AGXIR_ReduceXor, AGXIR_ReduceFMin, AGXIR_ReduceFMax, AGXIR_ReduceIAdd,
     AGXIR_ReduceSMin, AGXIR_ReduceSMax, AGXIR_ReduceUMin,
     AGXIR_ReduceUMax]> {
  let cppNamespace = "::mlir::AGXIR";
}

class AGXIR_ReduceOp<string mnemonic> :
    AGXIR_IntrOp<mnemonic, [0], [], [AllTypesMatch<["res", "value"]>], 1,
                 [1], ["kind"]>,
    Arguments<(ins AnyTypeOf<[I32, F32]>:$value, AGXIR_ReduceKind:$kind)> {
  let results = (outs AnyTypeOf<[I32, F32]>:$res);
  let assemblyFormat = "$kind $value attr-dict `:` type($res)";
  let hasVerifier = 1;
}

def AGXIR_SIMDReduceOp : AGXIR_ReduceOp<"simd.reduce"> {
  let summary = "Reduce a value over the active threads of the SIMD-group";
  let description = [{
    ```mlir
    %sum = agxir.simd.reduce fadd %v : f32
    ```
  }];
}
def AGXIR_SIMDPrefixOp : AGXIR_ReduceOp<"simd.prefix"> {
  let summary = "Prefix of a value over the active threads of the "
                "SIMD-group";
}
def AGXIR_QuadReduceOp : AGXIR_ReduceOp<"quad.reduce">;
def AGXIR_QuadPrefixOp : AGXIR_ReduceOp<"quad.prefix">;

//===----------------------------------------------------------------------===//
// Floating point mode
//===----------------------------------------------------------------------===//

def AGXIR_SetFPModeOp : AGXIR_IntrOp<"set.fpmode", [], [], [], 0>,
    Arguments<(ins I32:$mode)> {
  let summary = "Set the emulated floating point mode register";
  let description = [{
    Bits 0-1: float rounding (0 toward zero, 1 to nearest even, 2 up, 3
    down), bits 2-3: half and double rounding, bit 4: float denormals
    flushed, bit 5: half and double denormals flushed. It applies to the
    constrained floating point operations with dynamic rounding.

    ```mlir
    agxir.set.fpmode %mode
    ```
  }];
  let assemblyFormat = "$mode attr-dict";
}

def AGXIR_GetFPModeOp : AGXIR_IntrOp<"get.fpmode", [], [], [], 1> {
  let summary = "Read the emulated floating point mode register";
  let results = (outs I32:$res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// SIMD-group matrices
//===----------------------------------------------------------------------===//

def AGXIR_MatrixFragment : VectorOfLengthAndType<[2], [F16, BF16, F32]>;

def AGXIR_SIMDGroupMatrixFMAOp
    : AGXIR_IntrOp<"simdgroup.matrix.fma", [0], [],
                   [AllTypesMatch<["res", "a", "b", "c"]>], 1>,
      Arguments<(ins AGXIR_MatrixFragment:$a, AGXIR_MatrixFragment:$b,
                     AGXIR_MatrixFragment:$c)> {
  let summary = "8x8 matrix multiply-accumulate over the SIMD-group";
  let description = [{
    D = A * B + C on 8x8 matrices; thread t holds the elements (r, c) and
    (r, c + 1) with r = 4 * (t / 16) + (t / 2) % 4 and
    c = 4 * ((t / 8) % 2) + 2 * (t % 2).

    ```mlir
    %d = agxir.simdgroup.matrix.fma %a, %b, %c : vector<2xf32>
    ```
  }];
  let results = (outs AGXIR_MatrixFragment:$res);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Special functions
//===----------------------------------------------------------------------===//

class AGXIR_FloatUnaryOp<string mnemonic, list<Trait> traits = [Pure]> :
    AGXIR_IntrOp<mnemonic, [0], [],
                 !listconcat(traits, [AllTypesMatch<["res", "value"]>]), 1>,
    Arguments<(ins AnyTypeOf<[F16, F32]>:$value)> {
  let results = (outs AnyTypeOf<[F16, F32]>:$res);
  let assemblyFormat = "$value attr-dict `:` type($res)";
}

def AGXIR_RcpOp : AGXIR_FloatUnaryOp<"rcp"> {
  let summary = "Hardware reciprocal";
}
def AGXIR_RsqrtOp : AGXIR_FloatUnaryOp<"rsqrt"> {
  let summary = "Hardware reciprocal square root";
}
def AGXIR_RsqrtSpecialOp : AGXIR_FloatUnaryOp<"rsqrt.special"> {
  let summary = "Hardware reciprocal square root, with special cases so "
                "that rsqrt.special(x) * x == sqrt(x)";
}
def AGXIR_DfdxOp : AGXIR_FloatUnaryOp<"dfdx", []> {
  let summary = "Derivative across the quad in x";
}
def AGXIR_DfdyOp : AGXIR_FloatUnaryOp<"dfdy", []> {
  let summary = "Derivative across the quad in y";
}

//===----------------------------------------------------------------------===//
// Target attribute
//===----------------------------------------------------------------------===//

def AGXIR_TargetAttr : AGXIR_Attr<"AGXIRTarget", "target"> {
  let description = [{
    Target attribute for compiling GPU modules for Apple GPUs with the AGX
    LLVM backend. All parameters decay into default values if not present.

    ```mlir
    gpu.module @kernels [#agxir.target<chip = "agx-g13c">] {
      ...
    }
    ```

    Serialization produces the backend's assembly (`format=isa`), an ELF
    object with the machine code of each kernel in `.text` (`format=bin`) or
    LLVM bitcode (`format=llvm`).
  }];
  let parameters = (ins
    DefaultValuedParameter<"int", "2", "Optimization level to apply.">:$O,
    StringRefParameter<"Target triple.", "\"agx-apple-macosx\"">:$triple,
    StringRefParameter<"Target chip.", "\"agx-g14g\"">:$chip,
    StringRefParameter<"Target chip features.", "\"\"">:$features,
    OptionalParameter<"ArrayAttr", "Files to link to the LLVM module.">:$link
  );
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $features, $link)^ `>`)?
  }];
  let builders = [
    AttrBuilder<(ins CArg<"int", "2">:$optLevel,
                     CArg<"StringRef", "\"agx-apple-macosx\"">:$triple,
                     CArg<"StringRef", "\"agx-g14g\"">:$chip,
                     CArg<"StringRef", "\"\"">:$features,
                     CArg<"ArrayAttr", "nullptr">:$linkFiles), [{
      return Base::get($_ctxt, optLevel, triple, chip, features, linkFiles);
    }]>
  ];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

#endif // AGXIR_OPS
