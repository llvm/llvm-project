//===- SparseAssembler.cpp - adds wrapper method around sparse types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils/CodegenUtils.h"

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

// TODO: reuse StorageLayout::foreachField?

// TODO: we need COO AoS and SoA

// Convert type range to new types range, with sparse tensors externalized.
void convTypes(TypeRange types, SmallVectorImpl<Type> &convTypes,
               SmallVectorImpl<Type> *extraTypes = nullptr) {
  for (auto type : types) {
    // All "dense" data passes through unmodified.
    if (!getSparseTensorEncoding(type)) {
      convTypes.push_back(type);
      continue;
    }
    // Convert the external representation of the values array.
    const SparseTensorType stt(cast<RankedTensorType>(type));
    auto shape = {ShapedType::kDynamic};
    auto vtp = RankedTensorType::get(shape, stt.getElementType());
    convTypes.push_back(vtp);
    if (extraTypes)
      extraTypes->push_back(vtp);
    // Convert the external representations of the pos/crd arrays.
    for (Level lvl = 0, lvlRank = stt.getLvlRank(); lvl < lvlRank; lvl++) {
      const auto lt = stt.getLvlType(lvl);
      if (isCompressedLT(lt) || isLooseCompressedLT(lt)) {
        auto ptp = RankedTensorType::get(shape, stt.getPosType());
        auto ctp = RankedTensorType::get(shape, stt.getCrdType());
        convTypes.push_back(ptp);
        convTypes.push_back(ctp);
        if (extraTypes) {
          extraTypes->push_back(ptp);
          extraTypes->push_back(ctp);
        }
      } else {
        assert(isDenseLT(lt)); // TODO: handle other cases
      }
    }
  }
}

// Convert input and output values to [dis[assemble ops for sparse tensors.
void convVals(OpBuilder &builder, Location loc, TypeRange types,
              ValueRange fromVals, ValueRange extraVals,
              SmallVectorImpl<Value> &toVals, unsigned extra, bool isIn) {
  unsigned idx = 0;
  for (auto type : types) {
    // All "dense" data passes through unmodified.
    if (!getSparseTensorEncoding(type)) {
      toVals.push_back(fromVals[idx++]);
      continue;
    }
    // Convert the external representation of the values array.
    auto rtp = cast<RankedTensorType>(type);
    const SparseTensorType stt(rtp);
    auto shape = {ShapedType::kDynamic};
    SmallVector<Value> inputs;
    SmallVector<Type> retTypes;
    SmallVector<Type> cntTypes;
    // Collect the external representation of the values array for
    // input or the outgoing sparse tensor for output.
    inputs.push_back(fromVals[idx++]);
    if (!isIn) {
      inputs.push_back(extraVals[extra++]);
      retTypes.push_back(RankedTensorType::get(shape, stt.getElementType()));
      cntTypes.push_back(builder.getIndexType());
    }
    // Collect the external representations of the pos/crd arrays.
    for (Level lvl = 0, lvlRank = stt.getLvlRank(); lvl < lvlRank; lvl++) {
      const auto lt = stt.getLvlType(lvl);
      if (isCompressedLT(lt) || isLooseCompressedLT(lt)) {
        if (isIn) {
          inputs.push_back(fromVals[idx++]);
          inputs.push_back(fromVals[idx++]);
        } else {
          Type pTp = stt.getPosType();
          Type cTp = stt.getCrdType();
          inputs.push_back(extraVals[extra++]);
          inputs.push_back(extraVals[extra++]);
          retTypes.push_back(RankedTensorType::get(shape, pTp));
          retTypes.push_back(RankedTensorType::get(shape, cTp));
          cntTypes.push_back(pTp);
          cntTypes.push_back(cTp);
        }
      } else {
        assert(isDenseLT(lt)); // TODO: handle other cases
      }
    }
    if (isIn) {
      // Assemble multiple inputs into a single sparse tensor.
      auto a = builder.create<sparse_tensor::AssembleOp>(loc, rtp, inputs);
      toVals.push_back(a.getResult());
    } else {
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
// TODO: refine output sparse tensors to work well with external framework
//
// TODO: use "inlining" instead of a wrapper?
//
struct SparseFuncAssembler : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    // Only rewrite public entry methods.
    if (funcOp.isPrivate())
      return failure();

    // Translate sparse tensor types to external types.
    SmallVector<Type> inputTypes;
    SmallVector<Type> outputTypes;
    SmallVector<Type> extraTypes;
    convTypes(funcOp.getArgumentTypes(), inputTypes);
    convTypes(funcOp.getResultTypes(), outputTypes, &extraTypes);

    // Only sparse inputs or outputs need a wrapper method.
    if (inputTypes.size() == funcOp.getArgumentTypes().size() &&
        outputTypes.size() == funcOp.getResultTypes().size())
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
             ValueRange(), inputs, 0, /*isIn=*/true);

    // Call original, now internal method.
    auto org = SymbolRefAttr::get(context, wrapper);
    auto call = rewriter.create<func::CallOp>(loc, funcOp.getResultTypes(), org,
                                              inputs);

    // Convert outputs and return.
    SmallVector<Value> outputs;
    convVals(rewriter, loc, funcOp.getResultTypes(), call.getResults(),
             body->getArguments(), outputs, extra, /*isIn=*/false);
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
};

} // namespace

//===----------------------------------------------------------------------===//
// Public method for populating conversion rules.
//===----------------------------------------------------------------------===//

void mlir::populateSparseAssembler(RewritePatternSet &patterns) {
  patterns.add<SparseFuncAssembler>(patterns.getContext());
}
