//===-- XeVMOps.td - XeVM dialect definition ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef XEVMIR_OPS
#define XEVMIR_OPS

include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def XeVM_Dialect : Dialect {
  let name = "xevm";
  let cppNamespace = "::mlir::xevm";
  let summary = "The XeVM dialect that extends LLVM dialect and models Intel "
                "GPU's hardware features.";
  let description = [{
    The XeVM dialect is extension to the LLVM dialect that models hardware
    features of Intel GPUs. The dialect is designed to work with the Xe
    architecture for Intel GPUs, supporting advanced operations like 2D block
    loads, stores, prefetch and matrix multiply-add (MMA) operations.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];

  let extraClassDeclaration = [{
    /// Get the name for the attribute used to specify cache control
    /// decorations.
    static constexpr ::llvm::StringRef getCacheControlsAttrName() {
      return ::llvm::StringLiteral("xevm.DecorationCacheControl");
    }
  }];

  let useDefaultAttributePrinterParser = 1;
}

class XeVM_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<XeVM_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

class XeVM_Op<string mnemonic, list<Trait> traits = []>
    : LLVM_OpBase<XeVM_Dialect, mnemonic, traits> {

  code extraBaseClassDeclaration = [{
    void printProperties(::mlir::MLIRContext *ctx,
            ::mlir::OpAsmPrinter &p, const Properties &prop,
            ::mlir::ArrayRef<::llvm::StringRef> elidedProps) {
      Attribute propAttr = getPropertiesAsAttr(ctx, prop);
      if (propAttr)
        p << "<" << propAttr << ">";
    }

    static ::mlir::ParseResult parseProperties(::mlir::OpAsmParser &parser,
                                     ::mlir::OperationState &result) {
      if (mlir::succeeded(parser.parseOptionalLess())) {
        if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
          return failure();
      }
      return success();
    }

  }];
}

def XeVM_ElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, TF32, F16, BF16]>;
def XeVM_1DBlockElemType : AnyTypeOf<[I8, I16, I32, I64]>;

//===----------------------------------------------------------------------===//
// XeVM Load Cache Control
//   L1, L2, L3 - cache levels
//   uc - uncached
//   c - cached
//   s - streaming
//   ir - invalidated after read
//===----------------------------------------------------------------------===//

def LoadCacheControl_L1uc_L2uc_L3uc
    : I32EnumAttrCase<"L1UC_L2UC_L3UC", 1, "L1uc_L2uc_L3uc">;
def LoadCacheControl_L1uc_L2uc_L3c
    : I32EnumAttrCase<"L1UC_L2UC_L3C", 2, "L1uc_L2uc_L3c">;
def LoadCacheControl_L1uc_L2c_L3uc
    : I32EnumAttrCase<"L1UC_L2C_L3UC", 3, "L1uc_L2c_L3uc">;
def LoadCacheControl_L1uc_L2c_L3c
    : I32EnumAttrCase<"L1UC_L2C_L3C", 4, "L1uc_L2c_L3c">;
def LoadCacheControl_L1c_L2uc_L3uc
    : I32EnumAttrCase<"L1C_L2UC_L3UC", 5, "L1c_L2uc_L3uc">;
def LoadCacheControl_L1c_L2uc_L3c
    : I32EnumAttrCase<"L1C_L2UC_L3C", 6, "L1c_L2uc_L3c">;
def LoadCacheControl_L1c_L2c_L3uc
    : I32EnumAttrCase<"L1C_L2C_L3UC", 7, "L1c_L2c_L3uc">;
def LoadCacheControl_L1c_L2c_L3c
    : I32EnumAttrCase<"L1C_L2C_L3C", 8, "L1c_L2c_L3c">;
def LoadCacheControl_L1s_L2uc_L3uc
    : I32EnumAttrCase<"L1S_L2UC_L3UC", 9, "L1s_L2uc_L3uc">;
def LoadCacheControl_L1s_L2uc_L3c
    : I32EnumAttrCase<"L1S_L2UC_L3C", 10, "L1s_L2uc_L3c">;
def LoadCacheControl_L1s_L2c_L3uc
    : I32EnumAttrCase<"L1S_L2C_L3UC", 11, "L1s_L2c_L3uc">;
def LoadCacheControl_L1s_L2c_L3c
    : I32EnumAttrCase<"L1S_L2C_L3C", 12, "L1s_L2c_L3c">;
def LoadCacheControlInvalidateRead
    : I32EnumAttrCase<"INVALIDATE_READ", 13, "ir">;

def XeVM_LoadCacheControl
    : I32EnumAttr<
          "LoadCacheControl", "XeVM load ops cache control",
          [LoadCacheControl_L1uc_L2uc_L3uc, LoadCacheControl_L1uc_L2uc_L3c,
           LoadCacheControl_L1uc_L2c_L3uc, LoadCacheControl_L1uc_L2c_L3c,
           LoadCacheControl_L1c_L2uc_L3uc, LoadCacheControl_L1c_L2uc_L3c,
           LoadCacheControl_L1c_L2c_L3uc, LoadCacheControl_L1c_L2c_L3c,
           LoadCacheControl_L1s_L2uc_L3uc, LoadCacheControl_L1s_L2uc_L3c,
           LoadCacheControl_L1s_L2c_L3uc, LoadCacheControl_L1s_L2c_L3c,
           LoadCacheControlInvalidateRead]> {
  let cppNamespace = "::mlir::xevm";
  let genSpecializedAttr = 0;
}

def XeVM_LoadCacheControlAttr
    : EnumAttr<XeVM_Dialect, XeVM_LoadCacheControl, "load_cache_control"> {
  let summary = [{Describe the cache settings for load operators}];
  let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// XeVM Store Cache Control
//   L1, L2, L3 - cache levels
//   uc - uncached
//   wb - write-back
//   wt - write-through
//   s - streaming
//===----------------------------------------------------------------------===//

def StoreCacheControl_L1uc_L2uc_L3uc
    : I32EnumAttrCase<"L1UC_L2UC_L3UC", 1, "L1uc_L2uc_L3uc">;
def StoreCacheControl_L1uc_L2uc_L3wb
    : I32EnumAttrCase<"L1UC_L2UC_L3WB", 2, "L1uc_L2uc_L3wb">;
def StoreCacheControl_L1uc_L2wb_L3uc
    : I32EnumAttrCase<"L1UC_L2WB_L3UC", 3, "L1uc_L2wb_L3uc">;
def StoreCacheControl_L1uc_L2wb_L3wb
    : I32EnumAttrCase<"L1UC_L2WB_L3WB", 4, "L1uc_L2wb_L3wb">;
def StoreCacheControl_L1wt_L2uc_L3uc
    : I32EnumAttrCase<"L1WT_L2UC_L3UC", 5, "L1wt_L2uc_L3uc">;
def StoreCacheControl_L1wt_L2uc_L3wb
    : I32EnumAttrCase<"L1WT_L2UC_L3WB", 6, "L1wt_L2uc_L3wb">;
def StoreCacheControl_L1wt_L2wb_L3uc
    : I32EnumAttrCase<"L1WT_L2WB_L3UC", 7, "L1wt_L2wb_L3uc">;
def StoreCacheControl_L1wt_L2wb_L3wb
    : I32EnumAttrCase<"L1WT_L2WB_L3WB", 8, "L1wt_L2wb_L3wb">;
def StoreCacheControl_L1s_L2uc_L3uc
    : I32EnumAttrCase<"L1S_L2UC_L3UC", 9, "L1s_L2uc_L3uc">;
def StoreCacheControl_L1s_L2uc_L3wb
    : I32EnumAttrCase<"L1S_L2UC_L3WB", 10, "L1s_L2uc_L3wb">;
def StoreCacheControl_L1s_L2wb_L3uc
    : I32EnumAttrCase<"L1S_L2WB_L3UC", 11, "L1s_L2wb_L3uc">;
def StoreCacheControl_L1s_L2wb_L3wb
    : I32EnumAttrCase<"L1S_L2WB_L3WB", 12, "L1s_L2wb_L3wb">;
def StoreCacheControl_L1wb_L2uc_L3uc
    : I32EnumAttrCase<"L1WB_L2UC_L3UC", 13, "L1wb_L2uc_L3uc">;
def StoreCacheControl_L1wb_L2wb_L3uc
    : I32EnumAttrCase<"L1WB_L2WB_L3UC", 14, "L1wb_L2wb_L3uc">;
def StoreCacheControl_L1wb_L2uc_L3wb
    : I32EnumAttrCase<"L1WB_L2UC_L3WB", 15, "L1wb_L2uc_L3wb">;

def XeVM_StoreCacheControl
    : I32EnumAttr<
          "StoreCacheControl", "XeVM store ops cache control",
          [StoreCacheControl_L1uc_L2uc_L3uc, StoreCacheControl_L1uc_L2uc_L3wb,
           StoreCacheControl_L1uc_L2wb_L3uc, StoreCacheControl_L1uc_L2wb_L3wb,
           StoreCacheControl_L1wt_L2uc_L3uc, StoreCacheControl_L1wt_L2uc_L3wb,
           StoreCacheControl_L1wt_L2wb_L3uc, StoreCacheControl_L1wt_L2wb_L3wb,
           StoreCacheControl_L1s_L2uc_L3uc, StoreCacheControl_L1s_L2uc_L3wb,
           StoreCacheControl_L1s_L2wb_L3uc, StoreCacheControl_L1s_L2wb_L3wb,
           StoreCacheControl_L1wb_L2uc_L3uc, StoreCacheControl_L1wb_L2wb_L3uc,
           StoreCacheControl_L1wb_L2uc_L3wb]> {
  let cppNamespace = "::mlir::xevm";
  let genSpecializedAttr = 0;
}

def XeVM_StoreCacheControlAttr
    : EnumAttr<XeVM_Dialect, XeVM_StoreCacheControl, "store_cache_control"> {
  let summary = [{Describe the cache settings for store operators}];
  let assemblyFormat = "`<` $value `>`";
}

def XeVM_BlockLoadOp
    : XeVM_Op<"blockload">,
      Results<(outs AnyTypeOf<
          [XeVM_1DBlockElemType,
           FixedVectorOfRankAndType<[1], [XeVM_1DBlockElemType]>]>:$res)>,
      Arguments<(ins Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
          OptionalAttr<XeVM_LoadCacheControlAttr>:$cache_control)> {
  let summary = "subgroup block load";
  let description = [{
    Reads one or more components of Result data for each invocation
    in the subgroup from the specified `ptr` as a block operation.
    The data is read strided, so the first value read is:
    ```
      ptr[ SubgroupLocalInvocationId ]
    ```
    and the second value read is:
    ```
      ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ]
    ```
    Result type may be a scalar or vector type of scalar element type.

    The parameters are:
      * `ptr` - the base address to load from. Must be uniform across subgroup.
      * `cache_control` - an enumerator that sets the cache behaviour

    Example:
    ```mlir
      %loaded_a = xevm.blockload %src,
                      <{cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}>
                    : (!llvm.ptr<1>) -> vector<4xi16>
    ```
  }];
  let assemblyFormat = [{
    operands prop-dict attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def XeVM_BlockStoreOp
    : XeVM_Op<"blockstore">,
      Arguments<(ins Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
          AnyTypeOf<[XeVM_1DBlockElemType,
                     FixedVectorOfRankAndType<[1],
                                              [XeVM_1DBlockElemType]>]>:$val,
          OptionalAttr<XeVM_StoreCacheControlAttr>:$cache_control)> {
  let summary = "subgroup block store";
  let description = [{
    Writes one or more components of `val` for each invocation
    in the subgroup to the specified `ptr` as a block operation.
    The data is written strided, so the first value is written to:
    ```
      ptr[ SubgroupLocalInvocationId ]
    ```
    and the second value is written to:
    ```
      ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ]
    ```
    `val` type may be a scalar or vector type of scalar element type.

    The parameters are:
      * `ptr` - the base address to store to. Must be uniform across subgroup.
      * `val` - the value to store
      * `cache_control` - an enumerator that sets the cache behaviour

    Example:
    ```mlir
      xevm.blockstore %ptr, %val
        <{cache_control=#xevm.store_cache_control<L1uc_L2uc_L3uc>}>
        : (!llvm.ptr<1>, vector<4xi16>)
    ```
  }];

  let assemblyFormat = [{
    operands prop-dict attr-dict `:` `(` type(operands) `)`
  }];
  let hasVerifier = 1;
}

def XeVM_BlockLoad2dOp
    : XeVM_Op<"blockload2d">,
      Results<(outs FixedVectorOfRankAndType<[1], [XeVM_ElemType]>:$res)>,
      Arguments<(ins Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr, I32:$base_width,
          I32:$base_height, I32:$base_pitch, I32:$x, I32:$y,
          I32Attr:$elem_size_in_bits, I32Attr:$tile_width, I32Attr:$tile_height,
          I32Attr:$v_blocks, I1Attr:$transpose, I1Attr:$pack_register,
          OptionalAttr<XeVM_LoadCacheControlAttr>:$cache_control)> {

  let summary = "2D block load";

  let description = [{
    The `xevm.blockload2d` operation loads a two dimensional matrix tile
    from a base matrix residing in global memory. The parameters are:
      * `ptr` - the base address of the base matrix containing the tile to load
      * `base_width` - the width of the base matrix in number of bytes.
      * `base_height` - the number of rows in the base matrix
      * `base_pitch` - the physical stride between the first columns of the current
        row and the subsequent row in number of bytes.
      * `x`, `y`, `tile_width`, `tile_height` - the starting offsets and shape of
        the tile to load in number of elements.
      * `elem_size_in_bits` - the size in bits of the matrix element type
        - 32 for f32, tf32
        - 16 for f16, int16, bf16
        - 8 for int8
      * `v_blocks` - number of consecutive tiles in innermost dimension direction to load
      * `transpose` - transpose the tile in registers (useful for 32 bit element type)
      * `pack_register` - pack element types narrower than register bit width.
        [M, N] => [M/factor, N, factor] where factor is register_size_in_bits / elem_size_in_bits
      * `cache_control` - an enumerator that sets the cache behaviour

    Notes:
      - the `transpose` and `pack_register` parameters are mutual exclusive
      - transposing the tile loaded is used for A matrix in backward path or used for the B matrix operand
        (D = C + A * B), where A has row-major layout and B should have column-major layout in memory.
      - if the tile loaded contains out of bound elements of the matrix, they are filled with 0.

    Example:
    ```mlir
      %base_width_a = arith.constant 32 : i32
      %base_height_a = arith.constant 8 : i32
      %base_pitch_a = arith.constant 32 : i32
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      %loaded_a = xevm.blockload2d %src, %base_width_a, %base_height_a, %base_pitch_a, %x, %y
                    <{elem_size_in_bits=16 : i32, tile_width=16 : i32, tile_height=8 : i32,
                      v_blocks=1 : i32, transpose=false : i32, pack_register=false,
                      cache_control=#xevm.load_cache_control<Default>}>
                    : (!llvm.ptr<1>, i32, i32, i32, i32, i32) -> vector<8xi16>
    ```
  }];

  let assemblyFormat = [{
    operands prop-dict attr-dict `:` functional-type(operands, results)
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
  }];

  let hasVerifier = 1;
}

def XeVM_BlockStore2dOp
    : XeVM_Op<"blockstore2d">,
      Arguments<(ins Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr, I32:$base_width,
          I32:$base_height, I32:$base_pitch, I32:$x, I32:$y,
          I32Attr:$elem_size_in_bits, I32Attr:$tile_width, I32Attr:$tile_height,
          FixedVectorOfRankAndType<[1], [XeVM_ElemType]>:$stored_val,
          OptionalAttr<XeVM_StoreCacheControlAttr>:$cache_control)> {

  let summary = "2D block store";

  let description = [{
    The `xevm.blockstore2d` operation stores a two dimensional tile into a
    larger matrix residing in global memory. The parameters are:
      * `ptr` - the base address of the target matrix where to store the tile
      * `base_width` - the width of the base matrix in number of bytes.
      * `base_height` - the number of rows in the base matrix
      * `base_pitch` - the physical stride between the first columns of the current
        row and the subsequent row in number of bytes.
      * `x`, `y`, `tile_width`, `tile_height` - the starting offsets and shape of the tile to store
      in number of elements.
      * `elem_size_in_bits` - the size in bits of the matrix element
        - 32 for f32, tf32
        - 16 for f16, int16, bf16
        - 8 for int8
      * `cache_control` - an enumerator that sets the cache behaviour
      * `stored_val` - the tile to store

    Example:
    ```mlir
      %base_width_c = arith.constant 64 : i32
      %base_height_c = arith.constant 8 : i32
      %base_pitch_c = arith.constant 64 : i32
      %x = arith.constant 0 : i32
      %y = arith.constant 0 : i32
      xevm.blockstore2d %dst, %base_width_c, %base_height_c, %base_pitch_c, %x, %y, %src
        <{elem_size_in_bits=32 : i32, tile_width=16 : i32, tile_height=8 : i32,
          cache_control=#xevm.load_cache_control<Default>}>
        : (!llvm.ptr<1>, i32, i32, i32, i32, i32, vector<8xi32>)
    ```
  }];

  let assemblyFormat = [{
    operands prop-dict attr-dict `:` `(` type(operands) `)`
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
    /// Default value for v_blocks is 1.
    constexpr uint32_t getVBlocks() {
      return 1;
    }
  }];

  let hasVerifier = 1;
}

def MemScopeLane : I32EnumAttrCase<"LANE", 0, "lane">;
def MemScopeSubgroup : I32EnumAttrCase<"SUBGROUP", 1, "subgroup">;
def MemScopeWorkgroup : I32EnumAttrCase<"WORKGROUP", 2, "workgroup">;
def MemScopeCluster : I32EnumAttrCase<"CLUSTER", 3, "cluster">;
def MemScopeDevice : I32EnumAttrCase<"DEVICE", 4, "device">;
def MemScopeSystem : I32EnumAttrCase<"SYSTEM", 5, "system">;

def XeVM_MemScope
    : I32EnumAttr<"MemScope", "XeVM memory scope",
                  [MemScopeLane, MemScopeSubgroup, MemScopeWorkgroup,
                   MemScopeCluster, MemScopeDevice, MemScopeSystem]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xevm";
}
def XeVM_MemScopeAttr : EnumAttr<XeVM_Dialect, XeVM_MemScope, "mem_scope"> {
  let summary = [{Describe memory scopes}];
  let assemblyFormat = "`<` $value `>`";
}

def AddrSpacePrivate : I32EnumAttrCase<"PRIVATE", 0, "private">;
def AddrSpaceGlobal : I32EnumAttrCase<"GLOBAL", 1, "global">;
def AddrSpaceConstant : I32EnumAttrCase<"CONSTANT", 2, "constant">;
def AddrSpaceShared : I32EnumAttrCase<"SHARED", 3, "shared">;
def AddrSpaceGeneric : I32EnumAttrCase<"GENERIC", 4, "generic">;

def XeVM_AddrSpace
    : I32EnumAttr<"AddrSpace", "Address spaces",
                  [AddrSpacePrivate, AddrSpaceGlobal, AddrSpaceConstant,
                   AddrSpaceShared, AddrSpaceGeneric]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "mlir::xevm";
}
def XeVM_AddrSpaceAttr : EnumAttr<XeVM_Dialect, XeVM_AddrSpace, "addr_space"> {
  let summary = [{Describe address spaces}];
  let assemblyFormat = "`<` $value `>`";
}

def XeVM_MemfenceOp
    : XeVM_Op<"memfence">,
      Arguments<(ins XeVM_MemScopeAttr:$scope,
          DefaultValuedAttr<XeVM_AddrSpaceAttr,
                            "mlir::xevm::AddrSpace::GENERIC">:$addrspace)> {
  let summary = "Work-item's memory fence.";
  let description = [{
    This operation ensures that all prior memory accesses of this
    work-item to `addrspace` are visible to all other work-items in `scope`.
    Parameters description:
      * `scope` - specify the memory scope at which all other work-items should observe
        memory operations prior to the fence.
      * `addrspace` - specify the address space of work-item's memory accesses
        to be affected by the fence.
  }];
  let assemblyFormat = [{prop-dict  attr-dict}];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
  }];
}

def XeVM_PrefetchOp
    : XeVM_Op<"prefetch">,
      Arguments<(ins Arg<AnyTypeOf<[LLVM_PointerInAddressSpace<1>,
                                    LLVM_PointerInAddressSpace<4>]>>:$ptr,
          OptionalAttr<XeVM_LoadCacheControlAttr>:$cache_control)> {
  let summary = "Prefetch data into a cache subsystem.";
  let description = [{
    Work-item issues a prefetch from global memory to cache:
      * `ptr` - LLVM pointer with address space. Address space must be 1 (global)
        or 4 (generic)
      * `cache_control` - specify caching options
  }];
  let assemblyFormat = [{
    operands prop-dict attr-dict `:` `(` type(operands) `)`
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
  }];
}

def XeVM_BlockPrefetch2dOp
    : XeVM_Op<"blockprefetch2d">,
      Arguments<(ins Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr, I32:$base_width,
          I32:$base_height, I32:$base_pitch, I32:$x, I32:$y,
          I32Attr:$elem_size_in_bits, I32Attr:$tile_width, I32Attr:$tile_height,
          I32Attr:$v_blocks,
          OptionalAttr<XeVM_LoadCacheControlAttr>:$cache_control)> {

  let summary = "2D block prefetch";

  let description = [{
    The `xevm.blockprefetch2d` operation prefetches a two dimensional tile
    from a larger base matrix residing in global memory. The parameters are:
      * `ptr` - the base address of the base matrix containing the tile to prefetch
      * `base_width` - the width of the base matrix in number of bytes.
      * `base_height` - the number of rows in the base matrix
      * `base_pitch` - the physical stride between the first columns of the current
        row and the subsequent row in number of bytes.
      * `x`, `y`, `tile_width`, `tile_height` - the starting offsets and shape of tile
        to prefetch in number of elements.
      * `elem_size_in_bits` - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      * `v_blocks` - number of tiles in innermost dimension direction to prefetch
      * `cache_control` - an enumerator that sets the cache behaviour

    Example:
    ```mlir
      xevm.blockprefetch2d %ptr, %base_width, %base_height, %base_pitch, %x, %y
        <{elem_size_in_bits=8 : i32, tile_width=32 : i32, tile_height=8 : i32,
          v_blocks=1 : i32, cache_control=#xevm.load_cache_control<L1uc_L2uc_L3uc>}>
        : (!llvm.ptr<1>, i32, i32, i32, i32, i32)
    ```
  }];

  let assemblyFormat = [{
    operands prop-dict attr-dict `:` `(` type(operands) `)`
  }];

  let extraClassDeclaration = extraBaseClassDeclaration#[{
  }];

  let hasVerifier = 1;
}

def XeVM_MatrixElemType
    : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, TF32, F16, BF16]>;

/// Enum attribute of the different element types.
def XeVM_ET_BF16 : I32EnumAttrCase<"BF16", 8, "bf16">;
def XeVM_ET_F16 : I32EnumAttrCase<"F16", 9, "f16">;
def XeVM_ET_S8 : I32EnumAttrCase<"S8", 10, "s8">;
def XeVM_ET_U8 : I32EnumAttrCase<"U8", 11, "u8">;
def XeVM_ET_S4 : I32EnumAttrCase<"S4", 12, "s4">;
def XeVM_ET_U4 : I32EnumAttrCase<"U4", 13, "u4">;
def XeVM_ET_TF32 : I32EnumAttrCase<"TF32", 14, "tf32">;
def XeVM_ET_F32 : I32EnumAttrCase<"F32", 15, "f32">;
def XeVM_ET_S32 : I32EnumAttrCase<"S32", 16, "s32">;

def XeVM_ElemTypeAttr : I32EnumAttr<"ElemType", "XeVM element type",
                                    [XeVM_ET_BF16, XeVM_ET_F16, XeVM_ET_S8,
                                     XeVM_ET_U8, XeVM_ET_S4, XeVM_ET_U4,
                                     XeVM_ET_TF32, XeVM_ET_F32, XeVM_ET_S32]> {
  let cppNamespace = "::mlir::xevm";
}

def XeVM_MMAShapeAttr : XeVM_Attr<"MMAShape", "mma_shape"> {
  let description = [{
    MMA operation is represented as D=AxB+C, where
      - A has the shape MxK.
      - B has the shape KxN.
      - D and C have the shape MxN.
    This attribute encodes the shape of all matrices that participate in MMA.
  }];
  let parameters = (ins "int":$m, "int":$n, "int":$k);
  let assemblyFormat = "`<` struct(params) `>`";
}

def XeVM_MMATypesAttr : XeVM_Attr<"MMATypes", "mma_types"> {
  let parameters = (ins "xevm::ElemType":$d, "xevm::ElemType":$a,
      "xevm::ElemType":$b, OptionalParameter<"xevm::ElemType">:$c);
  let assemblyFormat = "`<` struct(params) `>`";
}

def XeVM_MMAOp
    : XeVM_Op<"mma">,
      Results<(outs FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$d)>,
      Arguments<(ins FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$a,
          FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>:$b,
          Optional<FixedVectorOfRankAndType<[1], [XeVM_MatrixElemType]>>:$c,
          XeVM_MMAShapeAttr:$shape, XeVM_MMATypesAttr:$types)> {

  let summary = "Subgroup matrix multiply-add";

  let description = [{
    The `xevm.mma` is a cooperative operation where all threads/lanes in
    a subgroup participates and carries out matrix multiplication plus accumulation:

      D = C + A x B

      where the A, B, C input matrices and the result D have shapes:
        - D : MxN
        - C : MxN
        - A : MxK
        - B : KxN

    Parameters:
      * `a` - vector of matrix A elements.
      * `b` - vector of matrix B elements.
      * `c` - (optional) vector of matrix C elements.
      * `shape` - the shape of the matrices, specified as `M`, `N`, and `K` values.
      * `types` - the data types of the matrices, specified as `D`, `A`, `B`, and optionally `C`.

    Example:
    ```mlir
      %d = xevm.mma %a, %b, %c { shape=<m=8, n=16, k=16>, types=<d=f32, a=f16, b=f16, c=f32> }
             : (vector<8xi16>, vector<8xi32>, vector<8xf32>) -> vector<8xf32>
    ```
  }];

  let assemblyFormat = [{
    $a `,` $b (`,` $c^)? ` `
    `{`
      `shape` `=` $shape `,`
      `types` `=` $types
    `}` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// XeVM target attribute.
//===----------------------------------------------------------------------===//

def XeVM_TargetAttr : XeVM_Attr<"XeVMTarget", "target"> {
  let description = [{
    GPU target attribute for controlling compilation of Intel GPU targets. All
    parameters decay into default values if not present.

    Examples:

    1. Target with default values.
    ```
      gpu.module @mymodule [#xevm.target] attributes {...} {
        ...
      }
    ```
  }];
  let parameters =
      (ins DefaultValuedParameter<"int", "2",
                                  "Optimization level to apply.">:$O,
          StringRefParameter<"Target triple.",
                             "\"spirv64-unknown-unknown\"">:$triple,
          StringRefParameter<"Target chip.", "\"bmg\"">:$chip,
          OptionalParameter<"::mlir::DictionaryAttr",
                            "Target specific flags.">:$flags,
          OptionalParameter<"::mlir::ArrayAttr",
                            "Files to link to the LLVM module.">:$linkFiles);
  let assemblyFormat = [{
    (`<` struct($O, $triple, $chip, $flags, $linkFiles)^ `>`)?
  }];
  let builders = [AttrBuilder<
      (ins CArg<"int", "2">:$optLevel,
          CArg<"::llvm::StringRef", "\"spirv64-unknown-unknown\"">:$triple,
          CArg<"::llvm::StringRef", "\"bmg\"">:$chip,
          CArg<"::mlir::DictionaryAttr", "nullptr">:$targetFlags,
          CArg<"::mlir::ArrayAttr", "nullptr">:$linkFiles),
      [{
      return Base::get($_ctxt, optLevel, triple, chip, targetFlags, linkFiles);
    }]>];
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
}

//===----------------------------------------------------------------------===//
// XeVM special register op definitions
//===----------------------------------------------------------------------===//

class XeVM_SpecialIdRegisterOp<string mnemonic, list<Trait> traits = []>
    : XeVM_Op<mnemonic, traits>,
      Results<(outs AnyTypeOf<[I32, I64]>:$res)>,
      Arguments<(ins OptionalAttr<LLVM_ConstantRangeAttr>:$range)> {
  let assemblyFormat = "(`range` $range^)? attr-dict `:` type($res)";
}

multiclass XeVM_SpecialRegisterXYZ<string mnemonic, list<Trait> traits = []> {
  def XOp : XeVM_SpecialIdRegisterOp<!strconcat(mnemonic, ".x"), traits>;
  def YOp : XeVM_SpecialIdRegisterOp<!strconcat(mnemonic, ".y"), traits>;
  def ZOp : XeVM_SpecialIdRegisterOp<!strconcat(mnemonic, ".z"), traits>;
}

//===----------------------------------------------------------------------===//
// Workitem index and range
defm XeVM_WorkitemId : XeVM_SpecialRegisterXYZ<"local_id">;
defm XeVM_WorkgroupDim : XeVM_SpecialRegisterXYZ<"local_size">;

//===----------------------------------------------------------------------===//
// Workgroup index and range
defm XeVM_WorkgroupId : XeVM_SpecialRegisterXYZ<"group_id">;
defm XeVM_GridDim : XeVM_SpecialRegisterXYZ<"group_count">;

//===----------------------------------------------------------------------===//
// Lane, Subgroup index and range
def XeVM_LaneIdOp : XeVM_SpecialIdRegisterOp<"lane_id">;
def XeVM_SubgroupIdOp : XeVM_SpecialIdRegisterOp<"subgroup_id">;
def XeVM_SubgroupSizeOp : XeVM_SpecialIdRegisterOp<"subgroup_size">;

#endif // XEVMIR_OPS
