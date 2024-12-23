//===- OpenACCUtils.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Utils/OpenACCUtils.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "llvm/ADT/TypeSwitch.h"

mlir::Value mlir::acc::getVarPtr(mlir::Operation *accDataClauseOp) {
  auto varPtr{llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
                  .Case<ACC_DATA_ENTRY_OPS, mlir::acc::CopyoutOp,
                        mlir::acc::UpdateHostOp>([&](auto dataClauseOp) {
                    return dataClauseOp.getVarPtr();
                  })
                  .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return varPtr;
}

bool mlir::acc::setVarPtr(mlir::Operation *accDataClauseOp,
                          mlir::Value varPtr) {
  bool res{llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
               .Case<ACC_DATA_ENTRY_OPS, mlir::acc::CopyoutOp,
                     mlir::acc::UpdateHostOp>([&](auto dataClauseOp) {
                 dataClauseOp.getVarPtrMutable().assign(varPtr);
                 return true;
               })
               .Default([&](mlir::Operation *) { return false; })};
  return res;
}

mlir::Value mlir::acc::getAccPtr(mlir::Operation *accDataClauseOp) {
  auto accPtr{
      llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
              [&](auto dataClauseOp) { return dataClauseOp.getAccPtr(); })
          .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return accPtr;
}

bool mlir::acc::setAccPtr(mlir::Operation *accDataClauseOp,
                          mlir::Value accPtr) {
  bool res{llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
               .Case<ACC_DATA_ENTRY_OPS>([&](auto dataClauseOp) {
                 // Cannot set the result of an existing operation and
                 // data entry ops produce `accPtr` as a result.
                 return false;
               })
               .Case<ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
                 dataClauseOp.getAccPtrMutable().assign(accPtr);
                 return true;
               })
               .Default([&](mlir::Operation *) { return false; })};
  return res;
}

mlir::Value mlir::acc::getVarPtrPtr(mlir::Operation *accDataClauseOp) {
  auto varPtrPtr{
      llvm::TypeSwitch<mlir::Operation *, mlir::Value>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS>(
              [&](auto dataClauseOp) { return dataClauseOp.getVarPtrPtr(); })
          .Default([&](mlir::Operation *) { return mlir::Value(); })};
  return varPtrPtr;
}

bool mlir::acc::setVarPtrPtr(mlir::Operation *accDataClauseOp,
                             mlir::Value varPtrPtr) {
  bool res{llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
               .Case<ACC_DATA_ENTRY_OPS>([&](auto dataClauseOp) {
                 dataClauseOp.getVarPtrPtrMutable().assign(varPtrPtr);
                 return true;
               })
               .Default([&](mlir::Operation *) { return false; })};
  return res;
}

mlir::SmallVector<mlir::Value>
mlir::acc::getBounds(mlir::Operation *accDataClauseOp) {
  mlir::SmallVector<mlir::Value> bounds{
      llvm::TypeSwitch<mlir::Operation *, mlir::SmallVector<mlir::Value>>(
          accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
            return mlir::SmallVector<mlir::Value>(
                dataClauseOp.getBounds().begin(),
                dataClauseOp.getBounds().end());
          })
          .Default([&](mlir::Operation *) {
            return mlir::SmallVector<mlir::Value, 0>();
          })};
  return bounds;
}

bool mlir::acc::setBounds(mlir::Operation *accDataClauseOp,
                          mlir::SmallVector<mlir::Value> &bounds) {
  bool res{
      llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
            dataClauseOp.getBoundsMutable().assign(bounds);
            return true;
          })
          .Default([&](mlir::Operation *) { return false; })};
  return res;
}

bool mlir::acc::setBounds(mlir::Operation *accDataClauseOp, mlir::Value bound) {
  mlir::SmallVector<mlir::Value> bounds({bound});
  return setBounds(accDataClauseOp, bounds);
}

std::optional<llvm::StringRef>
mlir::acc::getVarName(mlir::Operation *accDataClauseOp) {
  auto name{
      llvm::TypeSwitch<mlir::Operation *, std::optional<llvm::StringRef>>(
          accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
              [&](auto dataClauseOp) { return dataClauseOp.getName(); })
          .Default([&](mlir::Operation *) -> std::optional<llvm::StringRef> {
            return {};
          })};
  return name;
}

std::optional<mlir::acc::DataClause>
mlir::acc::getDataClause(mlir::Operation *accDataClauseOp) {
  auto dataClause{
      llvm::TypeSwitch<mlir::Operation *, std::optional<mlir::acc::DataClause>>(
          accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
              [&](auto dataClauseOp) { return dataClauseOp.getDataClause(); })
          .Default([&](mlir::Operation *) { return std::nullopt; })};
  return dataClause;
}

bool mlir::acc::setDataClause(mlir::Operation *accDataClauseOp,
                              mlir::acc::DataClause dataClause) {
  bool res{
      llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
            dataClauseOp.setDataClause(dataClause);
            return true;
          })
          .Default([&](mlir::Operation *) { return false; })};
  return res;
}

bool mlir::acc::getStructuredFlag(mlir::Operation *accDataClauseOp) {
  auto structured{
      llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
              [&](auto dataClauseOp) { return dataClauseOp.getStructured(); })
          .Default([&](mlir::Operation *) { return false; })};
  return structured;
}

bool mlir::acc::setStructuredFlag(mlir::Operation *accDataClauseOp,
                                  bool structured) {
  auto res{
      llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
            dataClauseOp.setStructured(structured);
            return true;
          })
          .Default([&](mlir::Operation *) { return false; })};
  return res;
}

bool mlir::acc::getImplicitFlag(mlir::Operation *accDataClauseOp) {
  auto implicit{
      llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
              [&](auto dataClauseOp) { return dataClauseOp.getImplicit(); })
          .Default([&](mlir::Operation *) { return false; })};
  return implicit;
}

bool mlir::acc::setImplicitFlag(mlir::Operation *accDataClauseOp,
                                bool implicit) {
  auto res{
      llvm::TypeSwitch<mlir::Operation *, bool>(accDataClauseOp)
          .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
            dataClauseOp.setImplicit(implicit);
            return true;
          })
          .Default([&](mlir::Operation *) { return false; })};
  return res;
}

mlir::SmallVector<mlir::Value>
mlir::acc::getAsyncOperands(mlir::Operation *accDataClauseOp) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::SmallVector<mlir::Value>>(
             accDataClauseOp)
      .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
        return mlir::SmallVector<mlir::Value>(
            dataClauseOp.getAsyncOperands().begin(),
            dataClauseOp.getAsyncOperands().end());
      })
      .Default([&](mlir::Operation *) {
        return mlir::SmallVector<mlir::Value, 0>();
      });
}

mlir::ArrayAttr
mlir::acc::getAsyncOperandsDeviceType(mlir::Operation *accDataClauseOp) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::ArrayAttr>(accDataClauseOp)
      .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>([&](auto dataClauseOp) {
        return dataClauseOp.getAsyncOperandsDeviceTypeAttr();
      })
      .Default([&](mlir::Operation *) { return mlir::ArrayAttr{}; });
}

mlir::ArrayAttr mlir::acc::getAsyncOnly(mlir::Operation *accDataClauseOp) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::ArrayAttr>(accDataClauseOp)
      .Case<ACC_DATA_ENTRY_OPS, ACC_DATA_EXIT_OPS>(
          [&](auto dataClauseOp) { return dataClauseOp.getAsyncOnlyAttr(); })
      .Default([&](mlir::Operation *) { return mlir::ArrayAttr{}; });
}

mlir::ValueRange mlir::acc::getDataOperands(mlir::Operation *accOp) {
  auto dataOperands{
      llvm::TypeSwitch<mlir::Operation *, mlir::ValueRange>(accOp)
          .Case<ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS>([&](auto accConstructOp) {
            return accConstructOp.getDataClauseOperands();
          })
          .Default([&](mlir::Operation *) { return mlir::ValueRange(); })};
  return dataOperands;
}

mlir::MutableOperandRange
mlir::acc::getMutableDataOperands(mlir::Operation *accOp) {
  auto dataOperands{
      llvm::TypeSwitch<mlir::Operation *, mlir::MutableOperandRange>(accOp)
          .Case<ACC_COMPUTE_AND_DATA_CONSTRUCT_OPS>([&](auto accConstructOp) {
            return accConstructOp.getDataClauseOperandsMutable();
          })
          .Default([&](mlir::Operation *) { return nullptr; })};
  return dataOperands;
}
