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
      "Value", "getViewSource">
  ];
}

def OffsetSizeAndStrideOpInterface : OpInterface<"OffsetSizeAndStrideOpInterface"> {
  let description = [{
    Common interface for ops that allow specifying mixed dynamic and static
    offsets, sizes and strides variadic operands.
    Ops that implement this interface need to expose the following methods:
      1. `getArrayAttrRanks` to specify the length of static integer
          attributes.
      2. `offsets`, `sizes` and `strides` variadic operands.
      3. `static_offsets`, resp. `static_sizes` and `static_strides` integer
          array attributes.
      4. `getOffsetSizeAndStrideStartOperandIndex` method that specifies the
         starting index of the OffsetSizeAndStrideOpInterface operands

    The invariants of this interface are:
      1. `static_offsets`, `static_sizes` and `static_strides` have length
          exactly `getArrayAttrRanks()`[0] (resp. [1], [2]).
      2. `offsets`, `sizes` and `strides` have each length at most
         `getArrayAttrRanks()`[0] (resp. [1], [2]).
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
      /*methodName=*/"getArrayAttrRanks",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic offset operands.
      }],
      /*retTy=*/"OperandRange",
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
      /*retTy=*/"OperandRange",
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
      /*retTy=*/"OperandRange",
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
      /*retTy=*/"ArrayAttr",
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
      /*retTy=*/"ArrayAttr",
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
      /*retTy=*/"ArrayAttr",
      /*methodName=*/"static_strides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.static_strides();
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
        APInt v = *(static_offsets()
          .template getAsValueRange<IntegerAttr>().begin() + idx);
        return ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
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
        APInt v = *(static_sizes()
          .template getAsValueRange<IntegerAttr>().begin() + idx);
        return ShapedType::isDynamic(v.getSExtValue());
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
        APInt v = *(static_strides()
          .template getAsValueRange<IntegerAttr>().begin() + idx);
        return ShapedType::isDynamicStrideOrOffset(v.getSExtValue());
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
        APInt v = *(static_offsets().
          template getAsValueRange<IntegerAttr>().begin() + idx);
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
        APInt v = *(static_sizes().
          template getAsValueRange<IntegerAttr>().begin() + idx);
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
        APInt v = *(static_strides().
          template getAsValueRange<IntegerAttr>().begin() + idx);
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
          static_offsets().template cast<ArrayAttr>(),
          ShapedType::isDynamicStrideOrOffset,
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
          static_sizes().template cast<ArrayAttr>(), ShapedType::isDynamic, idx);
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
          static_strides().template cast<ArrayAttr>(),
          ShapedType::isDynamicStrideOrOffset,
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
      /*args=*/(ins "ArrayAttr":$attr,
                    "llvm::function_ref<bool(int64_t)>":$isDynamic,
                    "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return std::count_if(
          attr.getValue().begin(), attr.getValue().begin() + idx,
          [&](Attribute attr) {
            return isDynamic(attr.cast<IntegerAttr>().getInt());
          });
      }]
    >,

    InterfaceMethod<
      /*desc=*/[{
        Assert the offset `idx` is dynamic and return its value.
      }],
      /*retTy=*/"Value",
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
      /*retTy=*/"Value",
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
      /*retTy=*/"Value",
      /*methodName=*/"getDynamicStride",
      /*args=*/(ins "unsigned":$idx),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperand(getIndexOfDynamicStride(idx));
      }]
    >,
  ];

  let extraClassDeclaration = [{
    static StringRef getStaticOffsetsAttrName() {
      return "static_offsets";
    }
    static StringRef getStaticSizesAttrName() {
      return "static_sizes";
    }
    static StringRef getStaticStridesAttrName() {
      return "static_strides";
    }
    static ArrayRef<StringRef> getSpecialAttrNames() {
      static SmallVector<StringRef, 4> names{
        OffsetSizeAndStrideOpInterface::getStaticOffsetsAttrName(),
        OffsetSizeAndStrideOpInterface::getStaticSizesAttrName(),
        OffsetSizeAndStrideOpInterface::getStaticStridesAttrName(),
        OpTrait::AttrSizedOperandSegments<void>::getOperandSegmentSizeAttr()};
      return names;
   }
  }];

  let verify = [{
    return mlir::verify(cast<OffsetSizeAndStrideOpInterface>($_op));
  }];
}

#endif // MLIR_INTERFACES_VIEWLIKEINTERFACE
