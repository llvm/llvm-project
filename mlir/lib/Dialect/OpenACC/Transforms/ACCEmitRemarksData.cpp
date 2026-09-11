//===- ACCEmitRemarksData.cpp - Emit OpenACC data-mapping remarks ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass emits optimization remarks describing OpenACC data-mapping clauses
// associated with structured and unstructured data constructs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCEMITREMARKSDATA
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-emit-remarks-data"

using namespace mlir;

namespace {

static bool isStructuredDeclareEnter(acc::DeclareEnterOp op) {
  return !op.getToken().getUsers().empty();
}

static StringRef getDataClauseRemarkPrefix(Operation *dataClauseOp) {
  std::optional<acc::DataClause> clause = acc::getDataClause(dataClauseOp);
  if (!clause)
    return {};

  switch (*clause) {
  case acc::DataClause::acc_copyin:
    return "copyin(";
  case acc::DataClause::acc_copyin_readonly:
    return "copyin(readonly:";
  case acc::DataClause::acc_copy:
    return "copy(";
  case acc::DataClause::acc_copyout:
    return "copyout(";
  case acc::DataClause::acc_copyout_zero:
    return "copyout(zero:";
  case acc::DataClause::acc_present:
    return "present(";
  case acc::DataClause::acc_create:
    return "create(";
  case acc::DataClause::acc_create_zero:
    return "create(zero:";
  case acc::DataClause::acc_delete:
    return "delete(";
  case acc::DataClause::acc_attach:
    return "attach(";
  case acc::DataClause::acc_detach:
    return "detach(";
  case acc::DataClause::acc_no_create:
    return "no_create(";
  case acc::DataClause::acc_private:
    return "private(";
  case acc::DataClause::acc_firstprivate:
    return "firstprivate(";
  case acc::DataClause::acc_deviceptr:
    return "deviceptr(";
  case acc::DataClause::acc_update_host:
    return "update_host(";
  case acc::DataClause::acc_update_self:
    return "update_self(";
  case acc::DataClause::acc_update_device:
    return "update_device(";
  case acc::DataClause::acc_use_device:
    return "use_device(";
  case acc::DataClause::acc_reduction:
    return isa<acc::CopyinOp>(dataClauseOp) ? "copy(" : "reduction(";
  case acc::DataClause::acc_declare_device_resident:
    return "device_resident(";
  case acc::DataClause::acc_declare_link:
    return "link(";
  case acc::DataClause::acc_cache:
    return "cache(";
  case acc::DataClause::acc_cache_readonly:
    return "cache(readonly:";
  case acc::DataClause::acc_getdeviceptr:
    return "";
  }
  llvm_unreachable("Unhandled data clause");
}

static bool shouldReport(Operation *op, acc::OpenACCSupport &accSupport,
                         std::string &varName) {
  if (!isa_and_nonnull<ACC_DATA_CLAUSE_OPS>(op))
    return false;
  if (getDataClauseRemarkPrefix(op).empty())
    return false;
  if (op->getNumResults() == 0)
    return false;
  varName = accSupport.getVariableName(op->getResult(0));
  // Not useful to report if the variable name is empty.
  return !varName.empty();
}

static bool reportOnSameLine(Operation *lhs, Operation *rhs) {
  return acc::getDataClause(lhs) == acc::getDataClause(rhs) &&
         acc::getImplicitFlag(lhs) == acc::getImplicitFlag(rhs);
}

static bool reportIfNotPresent(Operation *op) {
  switch (acc::getDataClause(op).value()) {
  case acc::DataClause::acc_copyin:
  case acc::DataClause::acc_copyin_readonly:
  case acc::DataClause::acc_copyout:
  case acc::DataClause::acc_copyout_zero:
  case acc::DataClause::acc_copy:
  case acc::DataClause::acc_create:
  case acc::DataClause::acc_create_zero:
  case acc::DataClause::acc_no_create:
    return true;
  case acc::DataClause::acc_present:
  case acc::DataClause::acc_delete:
  case acc::DataClause::acc_attach:
  case acc::DataClause::acc_detach:
  case acc::DataClause::acc_private:
  case acc::DataClause::acc_firstprivate:
  case acc::DataClause::acc_deviceptr:
  case acc::DataClause::acc_getdeviceptr:
  case acc::DataClause::acc_update_host:
  case acc::DataClause::acc_update_self:
  case acc::DataClause::acc_update_device:
  case acc::DataClause::acc_use_device:
  case acc::DataClause::acc_declare_device_resident:
  case acc::DataClause::acc_declare_link:
  case acc::DataClause::acc_cache:
  case acc::DataClause::acc_cache_readonly:
    return false;
  case acc::DataClause::acc_reduction:
    return isa<acc::CopyinOp>(op);
  }
  llvm_unreachable("Unhandled data clause");
}

static void emitDataMappingRemarks(ValueRange mappingOperands,
                                   StringRef directivePrefix,
                                   acc::OpenACCSupport &accSupport) {
  if (mappingOperands.empty())
    return;

  // Collect reportable ops with their variable names once so sorting and
  // remark formatting do not recompute them.
  struct MappingInfo {
    Operation *op;
    std::string varName;
  };
  SmallVector<MappingInfo, 8> mappingOps;
  mappingOps.reserve(mappingOperands.size());
  for (Value operand : mappingOperands) {
    Operation *defOp = operand.getDefiningOp();
    std::string varName;
    if (!shouldReport(defOp, accSupport, varName))
      continue;
    mappingOps.push_back({defOp, std::move(varName)});
  }
  if (mappingOps.empty())
    return;

  llvm::sort(mappingOps, [](const MappingInfo &lhs, const MappingInfo &rhs) {
    acc::DataClause lhsClause = acc::getDataClause(lhs.op).value();
    acc::DataClause rhsClause = acc::getDataClause(rhs.op).value();
    if (lhsClause == rhsClause) {
      bool lhsImplicit = acc::getImplicitFlag(lhs.op);
      bool rhsImplicit = acc::getImplicitFlag(rhs.op);
      if (lhsImplicit == rhsImplicit)
        return lhs.varName < rhs.varName;
      return lhsImplicit < rhsImplicit;
    }
    return lhsClause < rhsClause;
  });

  for (auto *it = mappingOps.begin(); it != mappingOps.end(); ++it) {
    Operation *op = it->op;

    SmallVector<MappingInfo *, 4> groupedOps = {it};
    while (std::next(it) != mappingOps.end() &&
           reportOnSameLine(op, std::next(it)->op)) {
      ++it;
      groupedOps.push_back(it);
    }

    // Anchor the remark on the data-clause op so its source location is used.
    accSupport.emitRemark(
        op,
        [&]() {
          std::string message = "Generating ";
          message += directivePrefix.str();
          if (op->getDiscardableAttr(acc::getFromDefaultClauseAttrName()))
            message += "default ";
          else if (acc::getImplicitFlag(op))
            message += "implicit ";
          message += getDataClauseRemarkPrefix(op).str();
          message += groupedOps.front()->varName;
          for (MappingInfo *grouped : llvm::drop_begin(groupedOps)) {
            message += ", ";
            message += grouped->varName;
          }
          message += ")";
          if (reportIfNotPresent(op))
            message += " [if not already present]";
          return message;
        },
        DEBUG_TYPE);
  }
}

class ACCEmitRemarksData
    : public acc::impl::ACCEmitRemarksDataBase<ACCEmitRemarksData> {
public:
  using ACCEmitRemarksDataBase<ACCEmitRemarksData>::ACCEmitRemarksDataBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    auto cachedAnalysis = getCachedParentAnalysis<acc::OpenACCSupport>();
    acc::OpenACCSupport &accSupport = cachedAnalysis
                                          ? cachedAnalysis->get()
                                          : getAnalysis<acc::OpenACCSupport>();

    // Remark emission policy:
    // - update / enter (structured or unstructured): report
    // - exit: only unstructured (structured exit was covered at enter)
    // - declare: only structured declare enter; unstructured declare
    //   registration does not emit mapping remarks.
    func.walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<acc::DataOp, acc::KernelEnvironmentOp>([&](auto dataOp) {
            emitDataMappingRemarks(dataOp.getDataClauseOperands(), "",
                                   accSupport);
          })
          .Case<acc::EnterDataOp>([&](acc::EnterDataOp enterOp) {
            emitDataMappingRemarks(enterOp.getDataClauseOperands(),
                                   "enter data ", accSupport);
          })
          .Case<acc::ExitDataOp>([&](acc::ExitDataOp exitOp) {
            emitDataMappingRemarks(exitOp.getDataClauseOperands(), "exit data ",
                                   accSupport);
          })
          .Case<acc::UpdateOp>([&](acc::UpdateOp updateOp) {
            emitDataMappingRemarks(updateOp.getDataClauseOperands(), "",
                                   accSupport);
          })
          .Case<acc::DeclareEnterOp>([&](acc::DeclareEnterOp declareEnterOp) {
            if (isStructuredDeclareEnter(declareEnterOp))
              emitDataMappingRemarks(declareEnterOp.getDataClauseOperands(), "",
                                     accSupport);
          });
    });
  }
};

} // namespace
