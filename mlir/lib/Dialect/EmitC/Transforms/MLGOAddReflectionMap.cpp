//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/EmitC/Transforms/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace emitc;

namespace mlir {
namespace emitc {
#define GEN_PASS_DEF_MLGOADDREFLECTIONMAPPASS
#include "mlir/Dialect/EmitC/Transforms/Passes.h.inc"

namespace {
constexpr const char *mapLibraryHeader = "map";
constexpr const char *stringLibraryHeader = "string";

struct PatternMatchListener : public RewriterBase::Listener {
  bool patternApplied = false;

  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    patternApplied = true;
  }
};

IncludeOp addHeader(OpBuilder &builder, ModuleOp module, StringRef headerName) {
  StringAttr includeAttr = builder.getStringAttr(headerName);
  return IncludeOp::create(builder, module.getLoc(), includeAttr,
                           /*is_standard_include=*/builder.getUnitAttr());
}

class MLGOAddReflectionMapPass
    : public impl::MLGOAddReflectionMapPassBase<MLGOAddReflectionMapPass> {
  using MLGOAddReflectionMapPassBase::MLGOAddReflectionMapPassBase;
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    RewritePatternSet patterns(&getContext());
    populateMLGOAddReflectionMapPatterns(patterns, includedFieldAttrs);

    PatternMatchListener listener;
    walkAndApplyPatterns(moduleOp, std::move(patterns), &listener);

    // If nothing was matched, no reflection maps were added, removing the need
    // to add include headers for map and string
    if (!listener.patternApplied)
      return;

    // Check if the map and/or string headers are already present
    bool hasMapHdr = false;
    bool hasStringHdr = false;
    for (auto &op : *moduleOp.getBody()) {
      IncludeOp includeOp = llvm::dyn_cast<IncludeOp>(op);
      if (!includeOp)
        continue;

      if (includeOp.getIsStandardInclude()) {
        auto include = includeOp.getInclude();

        hasMapHdr |= include == mapLibraryHeader;
        hasStringHdr |= include == stringLibraryHeader;
      }

      if (hasMapHdr && hasStringHdr)
        return;
    }

    mlir::OpBuilder builder(moduleOp.getBody(), moduleOp.getBody()->begin());
    if (!hasMapHdr)
      addHeader(builder, moduleOp, mapLibraryHeader);

    if (!hasStringHdr)
      addHeader(builder, moduleOp, stringLibraryHeader);
  }
};

} // namespace
} // namespace emitc
} // namespace mlir

/// Rewrites a `emitc::ClassOp` to generate a reflection map of member fields
/// and a lookup method.
///
/// Fields to be mapped are identified via `includedFieldAttrs` attributes
/// (e.g., `emitc.field_ref`). Fields that do not have a matching attribute
/// are omitted from the reflection map.
///
/// Before:
/// ```mlir
/// emitc.class @foo {
///   emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref =
///   ["another_feature"]}
/// emitc.func @bar() { return }
/// }
/// ```
///
/// After:
/// ```mlir
/// emitc.class @foo {
///   emitc.field @fieldName0 : !emitc.array<1xf32> {emitc.field_ref =
///   ["another_feature"]} emitc.field @reflectionMap : !emitc.opaque<"const
///   std::map<std::string, char*>"> =
///     #emitc.opaque<"{ { \"another_feature\",
///     reinterpret_cast<char*>(&fieldName0) } }">
///   emitc.func @getBufferForName(%name: !emitc.opaque<"std::string">) ->
///   !emitc.ptr<!emitc.opaque<"char">> {
///     %map = get_field @reflectionMap : !emitc.opaque<"const
///     std::map<std::string, char*>"> %ptr = member_call_opaque %map
///     "at"(%name) : ... return %ptr : !emitc.ptr<!emitc.opaque<"char">>
///   }
///   emitc.func @bar() { return }
/// }
/// ```
class MLGOAddReflectionMapClass : public OpRewritePattern<ClassOp> {
public:
  MLGOAddReflectionMapClass(MLIRContext *context,
                            llvm::ArrayRef<std::string> includedFieldAttrs)
      : OpRewritePattern<ClassOp>(context),
        includedFieldAttrs(includedFieldAttrs) {}

  LogicalResult matchAndRewrite(ClassOp classOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = rewriter.getContext();

    emitc::OpaqueType mapType = mlir::emitc::OpaqueType::get(
        context, "const std::map<std::string, char*>");

    // Collect the names of all FieldOps that have one of the attributes in
    // includedFieldAttrs to use the first element of the array attribute
    // as the reflection map key
    std::vector<std::pair<StringRef, StringRef>> fieldNames;
    classOp.walk([&](FieldOp fieldOp) {
      for (const auto &attr : includedFieldAttrs) {
        auto arrayAttr = dyn_cast_if_present<ArrayAttr>(fieldOp->getAttr(attr));

        if (!arrayAttr)
          continue;

        if (!arrayAttr.empty() && isa<StringAttr>(arrayAttr[0])) {
          fieldNames.emplace_back(cast<StringAttr>(arrayAttr[0]).getValue(),
                                  fieldOp.getName());
          return;
        }
      }
    });

    if (fieldNames.empty())
      return failure();

    // Create reflection map contents
    std::string reflectionMapContents;
    reflectionMapContents += "{ ";
    bool first = true;
    for (const auto &[name, value] : fieldNames) {
      if (!first)
        reflectionMapContents += ", ";

      first = false;
      reflectionMapContents += llvm::formatv(
          "{ \"{0}\", reinterpret_cast<char*>(&{1}) }", name, value);
    }
    reflectionMapContents += " }";

    // Set insertion point before the first function or after all fields
    // if there are no functions within the class
    auto funcs = classOp.getBlock().getOps<FuncOp>();
    auto it = funcs.begin();
    if (it != funcs.end())
      rewriter.setInsertionPoint(*it);
    else
      rewriter.setInsertionPointToEnd(&classOp.getBlock());

    // To generate the following C++ code
    // const std::map<std::string, char*> reflectionMap = {
    //  { "another_feature", reinterpret_cast<char*>(&fieldName0) },
    //  { "some_feature", reinterpret_cast<char*>(&fieldName1) },
    //  ...
    // };
    // This can be used to retrieve a pointer to the field's contents given the
    // attribute string identifying the field
    FieldOp reflectionMapField = FieldOp::create(
        rewriter, classOp.getLoc(), rewriter.getStringAttr("reflectionMap"),
        TypeAttr::get(mapType),
        emitc::OpaqueAttr::get(context, reflectionMapContents));

    // Create getBufferForName method
    emitc::OpaqueType nameType =
        emitc::OpaqueType::get(rewriter.getContext(), "std::string");
    emitc::OpaqueType charType =
        emitc::OpaqueType::get(rewriter.getContext(), "char");
    emitc::PointerType valType =
        emitc::PointerType::get(rewriter.getContext(), charType);
    FuncOp getBufferForNameFunc = FuncOp::create(
        rewriter, reflectionMapField->getLoc(), "getBufferForName",
        FunctionType::get(rewriter.getContext(), {nameType}, {valType}));

    Block *body =
        rewriter.createBlock(&getBufferForNameFunc.getBody(), {}, {nameType},
                             {reflectionMapField->getLoc()});
    rewriter.setInsertionPointToStart(body);
    GetFieldOp mapField = GetFieldOp::create(
        rewriter, reflectionMapField->getLoc(), mapType, "reflectionMap");
    Value nameArg = body->getArgument(0);
    MemberCallOpaqueOp lookupCall = MemberCallOpaqueOp::create(
        rewriter, reflectionMapField->getLoc(), valType, mapField.getResult(),
        "at", ArrayAttr{}, ArrayAttr{}, ValueRange{nameArg});
    ReturnOp::create(rewriter, reflectionMapField->getLoc(),
                     lookupCall.getResult(0));

    return success();
  }

private:
  /// The names of the attributes on FieldOps that contain the field name
  /// metadata for the reflection map. The pass matches the first attribute
  /// present in the order they are specified in this list.
  llvm::SmallVector<std::string> includedFieldAttrs;
};

void mlir::emitc::populateMLGOAddReflectionMapPatterns(
    RewritePatternSet &patterns,
    llvm::ArrayRef<std::string> includedFieldAttrs) {
  patterns.add<MLGOAddReflectionMapClass>(patterns.getContext(),
                                          includedFieldAttrs);
}
