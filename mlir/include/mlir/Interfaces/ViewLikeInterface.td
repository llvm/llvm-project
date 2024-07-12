//===- ViewLikeInterface.td - ViewLike interface -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface for view-like operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_VIEWLIKEINTERFACE
#define MLIR_INTERFACES_VIEWLIKEINTERFACE

include "mlir/IR/OpBase.td"

def ViewLikeOpInterface : OpInterface<"ViewLikeOpInterface"> {
  let description = [{
    A view-like operation "views" a buffer in a potentially different way. It
    takes in a (view of) buffer (and potentially some other operands) and returns
    another view of buffer.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      "Returns the source buffer from which the view is created.",
      "::mlir::Value", "getViewSource">
  ];
}

def OffsetSizeAndStrideOpInterface : OpInterface<"OffsetSizeAndStrideOpInterface"> {
  let description = [{
    Common interface for ops that allow specifying mixed dynamic and static
    offsets, sizes and strides variadic operands.
    Ops that implement this interface need to expose the following methods:
      1. `getArrayAttrMaxRanks` to specify the length of static integer
          attributes.
      2. `offsets`, `sizes` and `strides` variadic operands.
      3. `static_offsets`, resp. `static_sizes` and `static_strides` integer
          array attributes.
      4. `getOffsetSizeAndStrideStartOperandIndex` method that specifies the
         starting index of the OffsetSizeAndStrideOpInterface operands

    The invariants of this interface are:
      1. `static_offsets`, `static_sizes` and `static_strides` have length
          exactly `getArrayAttrMaxRanks()`[0] (resp. [1], [2]).
      2. `offsets`, `sizes` and `strides` have each length at most
         `getArrayAttrMaxRanks()`[0] (resp. [1], [2]).
      3. if an entry of `static_offsets` (resp. `static_sizes`,
         `static_strides`) is equal to a special sentinel value, namely
         `ShapedType::kDynamic`, then the corresponding entry is a dynamic
         offset (resp. size, stride).
      4. a variadic `offset` (resp. `sizes`, `strides`) operand  must be present
         for each dynamic offset (resp. size, stride).
      5. `offsets`, `sizes` and `strides` operands are specified in this order
         at operand index starting at `getOffsetSizeAndStrideStartOperandIndex`.
      6. `offsets` and `sizes` operands are non-negative.

    This interface is useful to factor out common behavior and provide support
    for carrying or injecting static behavior through the use of the static
    attributes.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    StaticInterfaceMethod<
      /*desc=*/[{
        Return the number of leading operands before the `offsets`, `sizes` and
        and `strides` operands.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getOffsetSizeAndStrideStartOperandIndex",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the expected rank of each of the`static_offsets`, `static_sizes`
        and `static_strides` attributes.
      }],
      /*retTy=*/"std::array<unsigned, 3>",
      /*methodName=*/"getArrayAttrMaxRanks",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic offset operands.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOffsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic size operands.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic stride operands.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getStrides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static offset attributes.
      }],
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getStaticOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getStaticOffsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static size attributes.
      }],
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getStaticSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getStaticSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic stride attributes.
      }],
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getStaticStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getStaticStrides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic offsets of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult, 4>",
      /*methodName=*/"getMixedOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getStaticOffsets(),
                                      $_op.getOffsets(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic sizes of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult, 4>",
      /*methodName=*/"getMixedSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return
            ::mlir::getMixedValues($_op.getStaticSizes(), $_op.getSizes(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic strides of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult, 4>",
      /*methodName=*/"getMixedStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getStaticStrides(),
                                      $_op.getStrides(), b);
      }]
    >,

    InterfaceMethod<
      /*desc=*/"Return true if the offset `idx` is dynamic.",
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicOffset",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::ShapedType::isDynamic(getStaticOffsets()[idx]);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return true if the size `idx` is dynamic.",
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicSize",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::ShapedType::isDynamic(getStaticSizes()[idx]);
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return true if the stride `idx` is dynamic.",
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicStride",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::ShapedType::isDynamic(getStaticStrides()[idx]);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the offset `idx` is a static constant and return its value.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getStaticOffset",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(!$_op.isDynamicOffset(idx) && "expected static offset");
        return getStaticOffsets()[idx];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the size `idx` is a static constant and return its value.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getStaticSize",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(!$_op.isDynamicSize(idx) && "expected static size");
        return getStaticSizes()[idx];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the stride `idx` is a static constant and return its value.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getStaticStride",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(!$_op.isDynamicStride(idx) && "expected static stride");
        return getStaticStrides()[idx];
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Assert the offset `idx` is dynamic and return the position of the
        corresponding operand.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getIndexOfDynamicOffset",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.isDynamicOffset(idx) && "expected dynamic offset");
        auto numDynamic = ::mlir::detail::getNumDynamicEntriesUpToIdx(
          getStaticOffsets(), idx);
        return $_op.getOffsetSizeAndStrideStartOperandIndex() + numDynamic;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the size `idx` is dynamic and return the position of the
        corresponding operand.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getIndexOfDynamicSize",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.isDynamicSize(idx) && "expected dynamic size");
        auto numDynamic = ::mlir::detail::getNumDynamicEntriesUpToIdx(
          getStaticSizes(), idx);
        return $_op.getOffsetSizeAndStrideStartOperandIndex() +
          getOffsets().size() + numDynamic;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the stride `idx` is dynamic and return the position of the
        corresponding operand.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getIndexOfDynamicStride",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert($_op.isDynamicStride(idx) && "expected dynamic stride");
        auto numDynamic = ::mlir::detail::getNumDynamicEntriesUpToIdx(
          getStaticStrides(), idx);
        return $_op.getOffsetSizeAndStrideStartOperandIndex() +
          getOffsets().size() + getSizes().size() + numDynamic;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the offset `idx` is dynamic and return its value.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getDynamicOffset",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperand(getIndexOfDynamicOffset(idx));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the size `idx` is dynamic and return its value.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getDynamicSize",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperand(getIndexOfDynamicSize(idx));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Assert the stride `idx` is dynamic and return its value.
      }],
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getDynamicStride",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperand(getIndexOfDynamicStride(idx));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if all `other`'s offsets, sizes and strides are the same.
        Takes a custom `cmp` comparison function on OpFoldResult to avoid taking
        a dialect dependence.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isSameAs",
      /*args=*/(ins "::mlir::OffsetSizeAndStrideOpInterface":$other,
                    "::llvm::function_ref<bool(::mlir::OpFoldResult, ::mlir::OpFoldResult)>":$cmp),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::mlir::detail::sameOffsetsSizesAndStrides(
          ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(
            $_op.getOperation()), other, cmp);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return true if all strides are guaranteed to be 1. }],
      /*retTy=*/"bool",
      /*methodName=*/"hasUnitStride",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::all_of(getMixedStrides(), [](::mlir::OpFoldResult ofr) {
          return ::mlir::getConstantIntValue(ofr) == static_cast<int64_t>(1);
        });
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{ Return true if all offsets are guaranteed to be 0. }],
      /*retTy=*/"bool",
      /*methodName=*/"hasZeroOffset",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return ::llvm::all_of(getMixedOffsets(), [](::mlir::OpFoldResult ofr) {
          return ::mlir::getConstantIntValue(ofr) == static_cast<int64_t>(0);
        });
      }]
    >,
  ];

  let verify = [{
    return ::mlir::detail::verifyOffsetSizeAndStrideOp(
        ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>($_op));
  }];
}

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE
