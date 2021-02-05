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
         `ShapedType::kDynamicStrideOrOffset` (resp. `ShapedType::kDynamicSize`,
         `ShapedType::kDynamicStrideOrOffset`), then the corresponding entry is
         a dynamic offset (resp. size, stride).
      4. a variadic `offset` (resp. `sizes`, `strides`) operand  must be present
         for each dynamic offset (resp. size, stride).
      5. `offsets`, `sizes` and `strides` operands are specified in this order
         at operand index starting at `getOffsetSizeAndStrideStartOperandIndex`.

    This interface is useful to factor out common behavior and provide support
    for carrying or injecting static behavior through the use of the static
    attributes.
  }];

  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
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
      /*methodName=*/"offsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.offsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic size operands.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"sizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.sizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic stride operands.
      }],
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"strides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.strides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static offset attributes.
      }],
      /*retTy=*/"::mlir::ArrayAttr",
      /*methodName=*/"static_offsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.static_offsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static size attributes.
      }],
      /*retTy=*/"::mlir::ArrayAttr",
      /*methodName=*/"static_sizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.static_sizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic stride attributes.
      }],
      /*retTy=*/"::mlir::ArrayAttr",
      /*methodName=*/"static_strides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.static_strides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic sizes of the op.
      }],
      /*retTy=*/"::mlir::SmallVector<::mlir::OpFoldResult, 4>",
      /*methodName=*/"getMixedOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::SmallVector<::mlir::OpFoldResult, 4> res;
        unsigned numDynamic = 0;
        unsigned count = $_op.static_offsets().size();
        for (unsigned idx = 0; idx < count; ++idx) {
          if (isDynamicOffset(idx))
            res.push_back($_op.offsets()[numDynamic++]);
          else
            res.push_back($_op.static_offsets()[idx]);
        }
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic sizes of the op.
      }],
      /*retTy=*/"::mlir::SmallVector<::mlir::OpFoldResult, 4>",
      /*methodName=*/"getMixedSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::SmallVector<::mlir::OpFoldResult, 4> res;
        unsigned numDynamic = 0;
        unsigned count = $_op.static_sizes().size();
        for (unsigned idx = 0; idx < count; ++idx) {
          if (isDynamicSize(idx))
            res.push_back($_op.sizes()[numDynamic++]);
          else
            res.push_back($_op.static_sizes()[idx]);
        }
        return res;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic strides of the op.
      }],
      /*retTy=*/"::mlir::SmallVector<::mlir::OpFoldResult, 4>",
      /*methodName=*/"getMixedStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::mlir::SmallVector<::mlir::OpFoldResult, 4> res;
        unsigned numDynamic = 0;
        unsigned count = $_op.static_strides().size();
        for (unsigned idx = 0; idx < count; ++idx) {
          if (isDynamicStride(idx))
            res.push_back($_op.strides()[numDynamic++]);
          else
            res.push_back($_op.static_strides()[idx]);
        }
        return res;
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Return true if the offset `idx` is dynamic.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicOffset",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::llvm::APInt v = *(static_offsets()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return true if the size `idx` is dynamic.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicSize",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::llvm::APInt v = *(static_sizes()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamic(v.getSExtValue());
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
       Return true if the stride `idx` is dynamic.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDynamicStride",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        ::llvm::APInt v = *(static_strides()
          .template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return ::mlir::ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
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
        ::llvm::APInt v = *(static_offsets().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
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
        ::llvm::APInt v = *(static_sizes().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
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
        ::llvm::APInt v = *(static_strides().
          template getAsValueRange<::mlir::IntegerAttr>().begin() + idx);
        return v.getSExtValue();
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
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_offsets().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
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
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_sizes().template cast<::mlir::ArrayAttr>(), ::mlir::ShapedType::isDynamic, idx);
        return $_op.getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + numDynamic;
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
        auto numDynamic = getNumDynamicEntriesUpToIdx(
          static_strides().template cast<::mlir::ArrayAttr>(),
          ::mlir::ShapedType::isDynamicStrideOrOffset,
          idx);
        return $_op.getOffsetSizeAndStrideStartOperandIndex() +
          offsets().size() + sizes().size() + numDynamic;
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Helper method to compute the number of dynamic entries of `attr`, up to
        `idx` using `isDynamic` to determine whether an entry is dynamic.
      }],
      /*retTy=*/"unsigned",
      /*methodName=*/"getNumDynamicEntriesUpToIdx",
      /*args=*/(ins "::mlir::ArrayAttr":$attr,
                    "::llvm::function_ref<bool(int64_t)>":$isDynamic,
                    "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::count_if(
          attr.getValue().begin(), attr.getValue().begin() + idx,
          [&](::mlir::Attribute attr) {
            return isDynamic(attr.cast<::mlir::IntegerAttr>().getInt());
          });
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
  ];

  let extraClassDeclaration = [{
    static unsigned getOffsetOperandGroupPosition() { return 0; }
    static unsigned getSizeOperandGroupPosition() { return 1; }
    static unsigned getStrideOperandGroupPosition() { return 2; }
    static ::llvm::StringRef getStaticOffsetsAttrName() {
      return "static_offsets";
    }
    static ::llvm::StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }
    static ::llvm::StringRef getStaticStridesAttrName() {
      return "static_strides";
    }
    static ::llvm::ArrayRef<::llvm::StringRef> getSpecialAttrNames() {
      static ::llvm::SmallVector<::llvm::StringRef, 4> names{
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName(),
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticSizesAttrName(),
        ::mlir::OffsetSizeAndStrideOpInterface::getStaticStridesAttrName(),
        ::mlir::OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()};
      return names;
   }
  }];

  let verify = [{
    return ::mlir::verify(::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>($_op));
  }];
}

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE
