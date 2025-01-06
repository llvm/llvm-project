//===- SparseAssembler.cpp - adds wrapper method around sparse types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils/CodegenUtils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorStorageLayout.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensorType.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace sparse_tensor;

//===----------------------------------------------------------------------===//
// Helper methods.
//===----------------------------------------------------------------------===//

// Convert type range to new types range, with sparse tensors externalized.
static void convTypes(bool &hasAnnotation, TypeRange types,
                      SmallVectorImpl<Type> &convTypes,
                      SmallVectorImpl<Type> *extraTypes, bool directOut) {
  for (auto type : types) {
    // All "dense" data passes through unmodified.
    if (!getSparseTensorEncoding(type)) {
      convTypes.push_back(type);
      continue;
    }
    hasAnnotation = true;

    // Convert the external representations of the pos/crd/val arrays.
    const SparseTensorType stt(cast<RankedTensorType>(type));
    foreachFieldAndTypeInSparseTensor(
        stt, [&convTypes, extraTypes, directOut](Type t, FieldIndex,
                                                 SparseTensorFieldKind kind,
                                                 Level, LevelType) {
          if (kind == SparseTensorFieldKind::PosMemRef ||
              kind == SparseTensorFieldKind::CrdMemRef ||
              kind == SparseTensorFieldKind::ValMemRef) {
            auto rtp = cast<ShapedType>(t);
            if (!directOut) {
              rtp = RankedTensorType::get(rtp.getShape(), rtp.getElementType());
              if (extraTypes)
                extraTypes->push_back(rtp);
            }
            convTypes.push_back(rtp);
          }
          return true;
        });
  }
}

// Convert input and output values to [dis]assemble ops for sparse tensors.
static void convVals(OpBuilder &builder, Location loc, TypeRange types,
                     ValueRange fromVals, ValueRange extraVals,
                     SmallVectorImpl<Value> &toVals, unsigned extra, bool isIn,
                     bool directOut) {
  unsigned idx = 0;
  for (auto type : types) {
    // All "dense" data passes through unmodified.
    if (!getSparseTensorEncoding(type)) {
      toVals.push_back(fromVals[idx++]);
      continue;
    }
    // Handle sparse data.
    auto rtp = cast<RankedTensorType>(type);
    const SparseTensorType stt(rtp);
    SmallVector<Value> inputs;
    SmallVector<Type> retTypes;
    SmallVector<Type> cntTypes;
    if (!isIn)
      inputs.push_back(fromVals[idx++]); // The sparse tensor to disassemble

    // Collect the external representations of the pos/crd/val arrays.
    foreachFieldAndTypeInSparseTensor(stt, [&, isIn](Type t, FieldIndex,
                                                     SparseTensorFieldKind kind,
                                                     Level lv, LevelType) {
      if (kind == SparseTensorFieldKind::PosMemRef ||
          kind == SparseTensorFieldKind::CrdMemRef ||
          kind == SparseTensorFieldKind::ValMemRef) {
        if (isIn) {
          inputs.push_back(fromVals[idx++]);
        } else if (directOut) {
          Value mem;
          if (kind == SparseTensorFieldKind::PosMemRef)
            mem = builder.create<sparse_tensor::ToPositionsOp>(loc, inputs[0],
                                                               lv);
          else if (kind == SparseTensorFieldKind::CrdMemRef)
            mem = builder.create<sparse_tensor::ToCoordinatesOp>(loc, inputs[0],
                                                                 lv);
          else
            mem = builder.create<sparse_tensor::ToValuesOp>(loc, inputs[0]);
          toVals.push_back(mem);
        } else {
          ShapedType rtp = cast<ShapedType>(t);
          rtp = RankedTensorType::get(rtp.getShape(), rtp.getElementType());
          inputs.push_back(extraVals[extra++]);
          retTypes.push_back(rtp);
          cntTypes.push_back(builder.getIndexType());
        }
      }
      return true;
    });

    if (isIn) {
      // Assemble multiple inputs into a single sparse tensor.
      auto a = builder.create<sparse_tensor::AssembleOp>(loc, rtp, inputs);
      toVals.push_back(a.getResult());
    } else if (!directOut) {
      // Disassemble a single sparse input into multiple outputs.
      // Note that this includes the counters, which are dropped.
      unsigned len = retTypes.size();
      retTypes.append(cntTypes);
      auto d =
          builder.create<sparse_tensor::DisassembleOp>(loc, retTypes, inputs);
      for (unsigned i = 0; i < len; i++)
        toVals.push_back(d.getResult(i));
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewriting rules.
//===----------------------------------------------------------------------===//

namespace {

// A rewriting rules that converts public entry methods that use sparse tensors
// as input parameters and/or output return values into wrapper methods that
// [dis]assemble the individual tensors that constitute the actual storage used
// externally into MLIR sparse tensors before calling the original method.
//
// In particular, each sparse tensor input
//
// void foo(..., t, ...) { }
//
// makes the original foo() internal and adds the following wrapper method
//
// void foo(..., t1..tn, ...) {
//   t = assemble t1..tn
//   _internal_foo(..., t, ...)
// }
//
// and likewise, each output tensor
//
// ... T ... bar(...) { return ..., t, ...; }
//
// makes the original bar() internal and adds the following wrapper method
//
// ... T1..TN ... bar(..., t1'..tn') {
//   ..., t, ... = _internal_bar(...)
//   t1..tn = disassemble t, t1'..tn'
//   return ..., t1..tn, ...
// }
//
// (with a direct-out variant without the disassemble).
//
struct SparseFuncAssembler : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  SparseFuncAssembler(MLIRContext *context, bool dO)
      : OpRewritePattern(context), directOut(dO) {}

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    // Only rewrite public entry methods.
    if (funcOp.isPrivate())
      return failure();

    // Translate sparse tensor types to external types.
    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes;
    SmallVector<Type> extraTypes;
    bool hasAnnotation = false;
    convTypes(hasAnnotation, funcOp.getArgumentTypes(), inputTypes, nullptr,
              false);
    convTypes(hasAnnotation, funcOp.getResultTypes(), outputTypes, &extraTypes,
              directOut);

    // Only sparse inputs or outputs need a wrapper method.
    if (!hasAnnotation)
      return failure();

    // Modify the original method into an internal, private method.
    auto orgName = funcOp.getName();
    std::string wrapper = llvm::formatv("_internal_{0}", orgName).str();
    funcOp.setName(wrapper);
    funcOp.setPrivate();

    // Start the new public wrapper method with original name.
    Location loc = funcOp.getLoc();
    ModuleOp modOp = funcOp->getParentOfType<ModuleOp>();
    MLIRContext *context = modOp.getContext();
    OpBuilder moduleBuilder(modOp.getBodyRegion());
    unsigned extra = inputTypes.size();
    inputTypes.append(extraTypes);
    auto func = moduleBuilder.create<func::FuncOp>(
        loc, orgName, FunctionType::get(context, inputTypes, outputTypes));
    func.setPublic();

    // Construct new wrapper method body.
    OpBuilder::InsertionGuard insertionGuard(rewriter);
    Block *body = func.addEntryBlock();
    rewriter.setInsertionPointToStart(body);

    // Convert inputs.
    SmallVector<Value> inputs;
    convVals(rewriter, loc, funcOp.getArgumentTypes(), body->getArguments(),
             ValueRange(), inputs, /*extra=*/0, /*isIn=*/true, directOut);

    // Call the original, now private method. A subsequent inlining pass can
    // determine whether cloning the method body in place is worthwhile.
    auto org = SymbolRefAttr::get(context, wrapper);
    auto call = rewriter.create<func::CallOp>(loc, funcOp.getResultTypes(), org,
                                              inputs);

    // Convert outputs and return.
    SmallVector<Value> outputs;
    convVals(rewriter, loc, funcOp.getResultTypes(), call.getResults(),
             body->getArguments(), outputs, extra, /*isIn=*/false, directOut);
    rewriter.create<func::ReturnOp>(loc, outputs);

    // Finally, migrate a potential c-interface property.
    if (funcOp->getAttrOfType<UnitAttr>(
            LLVM::LLVMDialect::getEmitCWrapperAttrName())) {
      func->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                    UnitAttr::get(context));
      funcOp->removeAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName());
    }
    return success();
  }

private:
  const bool directOut;
};

} // namespace

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

void mlir::populateSparseAssembler(RewritePatternSet &patterns,
                                   bool directOut) {
  patterns.add<SparseFuncAssembler>(patterns.getContext(), directOut);
}
