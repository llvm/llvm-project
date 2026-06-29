#ifndef POLYGEISTOPS_H
#define POLYGEISTOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

bool collectEffects(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    bool ignoreBarriers);

bool getEffectsBefore(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    bool stopAtBarrier);

bool getEffectsAfter(
    mlir::Operation *op,
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects,
    bool stopAtBarrier);

bool isReadOnly(mlir::Operation *);
bool isReadNone(mlir::Operation *);

bool mayReadFrom(mlir::Operation *, mlir::Value);
bool mayWriteTo(mlir::Operation *, mlir::Value, bool ignoreBarrier = false);

bool mayAlias(mlir::MemoryEffects::EffectInstance a,
              mlir::MemoryEffects::EffectInstance b);

bool mayAlias(mlir::MemoryEffects::EffectInstance a, mlir::Value b);

struct ValueOrInt {
  bool isValue;
  mlir::Value vVal;
  int64_t iVal;
  ValueOrInt(mlir::Value v) { initValue(v); }
  void initValue(mlir::Value v) {
    using namespace mlir;
    if (v) {
      IntegerAttr iattr;
      if (matchPattern(v, m_Constant(&iattr))) {
        iVal = iattr.getValue().getSExtValue();
        vVal = nullptr;
        isValue = false;
        return;
      }
    }
    isValue = true;
    vVal = v;
  }

  ValueOrInt(size_t i) : isValue(false), vVal(), iVal(i) {}

  bool operator>=(int64_t v) {
    if (isValue)
      return false;
    return iVal >= v;
  }
  bool operator>(int64_t v) {
    if (isValue)
      return false;
    return iVal > v;
  }
  bool operator==(int64_t v) {
    if (isValue)
      return false;
    return iVal == v;
  }
  bool operator<(int64_t v) {
    if (isValue)
      return false;
    return iVal < v;
  }
  bool operator<=(int64_t v) {
    if (isValue)
      return false;
    return iVal <= v;
  }
  bool operator>=(const llvm::APInt &v) {
    if (isValue)
      return false;
    return iVal >= v.getSExtValue();
  }
  bool operator>(const llvm::APInt &v) {
    if (isValue)
      return false;
    return iVal > v.getSExtValue();
  }
  bool operator==(const llvm::APInt &v) {
    if (isValue)
      return false;
    return iVal == v.getSExtValue();
  }
  bool operator<(const llvm::APInt &v) {
    if (isValue)
      return false;
    return iVal < v.getSExtValue();
  }
  bool operator<=(const llvm::APInt &v) {
    if (isValue)
      return false;
    return iVal <= v.getSExtValue();
  }
};

enum class Cmp { EQ, LT, LE, GT, GE };

bool valueCmp(Cmp cmp, mlir::AffineExpr expr, size_t numDim,
              mlir::ValueRange operands, ValueOrInt val);

bool valueCmp(Cmp cmp, mlir::Value bval, ValueOrInt val);
#endif
