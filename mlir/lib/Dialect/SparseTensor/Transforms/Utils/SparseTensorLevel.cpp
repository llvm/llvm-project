//===- SparseTensorLevel.cpp - Tensor management class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SparseTensorLevel.h"
#include "CodegenUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::sparse_tensor;

std::unique_ptr<SparseTensorLevel>
sparse_tensor::makeSparseTensorLevel(OpBuilder &builder, Location loc, Value t,
                                     Level l) {
  auto stt = getSparseTensorType(t);

  LevelType lt = stt.getLvlType(l);
  Value lvlSz = stt.hasEncoding()
                    ? builder.create<LvlOp>(loc, t, l).getResult()
                    : builder.create<tensor::DimOp>(loc, t, l).getResult();

  switch (*getLevelFormat(lt)) {
  case LevelFormat::Dense:
    return std::make_unique<DenseLevel>(lvlSz);
  case LevelFormat::Compressed: {
    Value posBuf = genToPositions(builder, loc, t, l);
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<CompressedLevel>(lt, lvlSz, posBuf, crdBuf);
  }
  case LevelFormat::LooseCompressed: {
    Value posBuf = genToPositions(builder, loc, t, l);
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<LooseCompressedLevel>(lt, lvlSz, posBuf, crdBuf);
  }
  case LevelFormat::Singleton: {
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<SingletonLevel>(lt, lvlSz, crdBuf);
  }
  case LevelFormat::TwoOutOfFour: {
    Value crdBuf = genToCoordinates(builder, loc, t, l);
    return std::make_unique<TwoOutFourLevel>(lt, lvlSz, crdBuf);
  }
  }
  llvm_unreachable("unrecognizable level format");
}

Value SparseLevel::peekCrdAt(OpBuilder &b, Location l, Value pos) const {
  return genIndexLoad(b, l, crdBuffer, pos);
}
