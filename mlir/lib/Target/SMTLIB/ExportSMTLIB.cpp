//===- ExportSMTLIB.cpp - SMT-LIB Emitter -----=---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SMT-LIB emitter implementation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/SMTLIB/ExportSMTLIB.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/SMT/IR/SMTVisitors.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Target/SMTLIB/Namespace.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace smt;

using ValueMap = llvm::ScopedHashTable<mlir::Value, std::string>;

#define DEBUG_TYPE "export-smtlib"

namespace {

/// A visitor to print the SMT dialect types as SMT-LIB formatted sorts.
/// Printing nested types use recursive calls since nestings of a depth that
/// could lead to problems should not occur in practice.
struct TypeVisitor : public smt::SMTTypeVisitor<TypeVisitor, void,
                                                mlir::raw_indented_ostream &> {
  TypeVisitor(const SMTEmissionOptions &options) : options(options) {}

  void visitSMTType(BoolType type, mlir::raw_indented_ostream &stream) {
    stream << "Bool";
  }

  void visitSMTType(IntType type, mlir::raw_indented_ostream &stream) {
    stream << "Int";
  }

  void visitSMTType(BitVectorType type, mlir::raw_indented_ostream &stream) {
    stream << "(_ BitVec " << type.getWidth() << ")";
  }

  void visitSMTType(ArrayType type, mlir::raw_indented_ostream &stream) {
    stream << "(Array ";
    dispatchSMTTypeVisitor(type.getDomainType(), stream);
    stream << " ";
    dispatchSMTTypeVisitor(type.getRangeType(), stream);
    stream << ")";
  }

  void visitSMTType(SMTFuncType type, mlir::raw_indented_ostream &stream) {
    stream << "(";
    StringLiteral nextToken = "";

    for (Type domainTy : type.getDomainTypes()) {
      stream << nextToken;
      dispatchSMTTypeVisitor(domainTy, stream);
      nextToken = " ";
    }

    stream << ") ";
    dispatchSMTTypeVisitor(type.getRangeType(), stream);
  }

  void visitSMTType(SortType type, mlir::raw_indented_ostream &stream) {
    if (!type.getSortParams().empty())
      stream << "(";

    stream << type.getIdentifier().getValue();
    for (Type paramTy : type.getSortParams()) {
      stream << " ";
      dispatchSMTTypeVisitor(paramTy, stream);
    }

    if (!type.getSortParams().empty())
      stream << ")";
  }

private:
  // A reference to the emission options for easy use in the visitor methods.
  [[maybe_unused]] const SMTEmissionOptions &options;
};

/// Contains the informations passed to the ExpressionVisitor methods. Makes it
/// easier to add more information.
struct VisitorInfo {
  VisitorInfo(mlir::raw_indented_ostream &stream, ValueMap &valueMap)
      : stream(stream), valueMap(valueMap) {}
  VisitorInfo(mlir::raw_indented_ostream &stream, ValueMap &valueMap,
              unsigned indentLevel, unsigned openParens)
      : stream(stream), valueMap(valueMap), indentLevel(indentLevel),
        openParens(openParens) {}

  // Stream to print to.
  mlir::raw_indented_ostream &stream;
  // Mapping from SSA values to SMT-LIB expressions.
  ValueMap &valueMap;
  // Total number of spaces currently indented.
  unsigned indentLevel = 0;
  // Number of parentheses that have been opened but not closed yet.
  unsigned openParens = 0;
};

/// A visitor to print SMT dialect operations with exactly one result value as
/// the equivalent operator in SMT-LIB.
struct ExpressionVisitor
    : public smt::SMTOpVisitor<ExpressionVisitor, LogicalResult,
                               VisitorInfo &> {
  using Base =
      smt::SMTOpVisitor<ExpressionVisitor, LogicalResult, VisitorInfo &>;
  using Base::visitSMTOp;

  ExpressionVisitor(const SMTEmissionOptions &options, Namespace &names)
      : options(options), typeVisitor(options), names(names) {}

  LogicalResult dispatchSMTOpVisitor(Operation *op, VisitorInfo &info) {
    assert(op->getNumResults() == 1 &&
           "expression op must have exactly one result value");

    // Print the expression inlined if it is only used once and the
    // corresponding emission option is enabled. This can lead to bad
    // performance for big inputs since the inlined expression is stored as a
    // string in the value mapping where otherwise only the symbol names of free
    // and bound variables are stored, and due to a lot of string concatenation
    // (thus it's off by default and just intended to print small examples in a
    // more human-readable format).
    Value res = op->getResult(0);
    if (res.hasOneUse() && options.inlineSingleUseValues) {
      std::string str;
      llvm::raw_string_ostream sstream(str);
      mlir::raw_indented_ostream indentedStream(sstream);

      VisitorInfo newInfo(indentedStream, info.valueMap, info.indentLevel,
                          info.openParens);
      if (failed(Base::dispatchSMTOpVisitor(op, newInfo)))
        return failure();

      info.valueMap.insert(res, str);
      return success();
    }

    // Generate a let binding for the current expression being processed and
    // store the sybmol in the value map.  Indent the expressions for easier
    // readability.
    auto name = names.newName("tmp");
    info.valueMap.insert(res, name.str());
    info.stream << "(let ((" << name << " ";

    VisitorInfo newInfo(info.stream, info.valueMap,
                        info.indentLevel + 8 + name.size(), 0);
    if (failed(Base::dispatchSMTOpVisitor(op, newInfo)))
      return failure();

    info.stream << "))\n";

    if (options.indentLetBody) {
      // Five spaces to align with the opening parenthesis
      info.indentLevel += 5;
    }
    ++info.openParens;
    info.stream.indent(info.indentLevel);

    return success();
  }

  //===--------------------------------------------------------------------===//
  // Bit-vector theory operation visitors
  //===--------------------------------------------------------------------===//

  template <typename Op>
  LogicalResult printBinaryOp(Op op, StringRef name, VisitorInfo &info) {
    info.stream << "(" << name << " " << info.valueMap.lookup(op.getLhs())
                << " " << info.valueMap.lookup(op.getRhs()) << ")";
    return success();
  }

  template <typename Op>
  LogicalResult printVariadicOp(Op op, StringRef name, VisitorInfo &info) {
    info.stream << "(" << name;
    for (Value val : op.getOperands())
      info.stream << " " << info.valueMap.lookup(val);
    info.stream << ")";
    return success();
  }

  LogicalResult visitSMTOp(BVNegOp op, VisitorInfo &info) {
    info.stream << "(bvneg " << info.valueMap.lookup(op.getInput()) << ")";
    return success();
  }

  LogicalResult visitSMTOp(BVNotOp op, VisitorInfo &info) {
    info.stream << "(bvnot " << info.valueMap.lookup(op.getInput()) << ")";
    return success();
  }

#define HANDLE_OP(OPTYPE, NAME, KIND)                                          \
  LogicalResult visitSMTOp(OPTYPE op, VisitorInfo &info) {                     \
    return print##KIND##Op(op, NAME, info);                                    \
  }

  HANDLE_OP(BVAddOp, "bvadd", Binary);
  HANDLE_OP(BVMulOp, "bvmul", Binary);
  HANDLE_OP(BVURemOp, "bvurem", Binary);
  HANDLE_OP(BVSRemOp, "bvsrem", Binary);
  HANDLE_OP(BVSModOp, "bvsmod", Binary);
  HANDLE_OP(BVShlOp, "bvshl", Binary);
  HANDLE_OP(BVLShrOp, "bvlshr", Binary);
  HANDLE_OP(BVAShrOp, "bvashr", Binary);
  HANDLE_OP(BVUDivOp, "bvudiv", Binary);
  HANDLE_OP(BVSDivOp, "bvsdiv", Binary);
  HANDLE_OP(BVAndOp, "bvand", Binary);
  HANDLE_OP(BVOrOp, "bvor", Binary);
  HANDLE_OP(BVXOrOp, "bvxor", Binary);
  HANDLE_OP(ConcatOp, "concat", Binary);

  LogicalResult visitSMTOp(ExtractOp op, VisitorInfo &info) {
    info.stream << "((_ extract "
                << (op.getLowBit() + op.getType().getWidth() - 1) << " "
                << op.getLowBit() << ") " << info.valueMap.lookup(op.getInput())
                << ")";
    return success();
  }

  LogicalResult visitSMTOp(RepeatOp op, VisitorInfo &info) {
    info.stream << "((_ repeat " << op.getCount() << ") "
                << info.valueMap.lookup(op.getInput()) << ")";
    return success();
  }

  LogicalResult visitSMTOp(BVCmpOp op, VisitorInfo &info) {
    return printBinaryOp(op, "bv" + stringifyBVCmpPredicate(op.getPred()).str(),
                         info);
  }

  //===--------------------------------------------------------------------===//
  // Int theory operation visitors
  //===--------------------------------------------------------------------===//

  HANDLE_OP(IntAddOp, "+", Variadic);
  HANDLE_OP(IntMulOp, "*", Variadic);
  HANDLE_OP(IntSubOp, "-", Binary);
  HANDLE_OP(IntDivOp, "div", Binary);
  HANDLE_OP(IntModOp, "mod", Binary);

  LogicalResult visitSMTOp(IntCmpOp op, VisitorInfo &info) {
    switch (op.getPred()) {
    case IntPredicate::ge:
      return printBinaryOp(op, ">=", info);
    case IntPredicate::le:
      return printBinaryOp(op, "<=", info);
    case IntPredicate::gt:
      return printBinaryOp(op, ">", info);
    case IntPredicate::lt:
      return printBinaryOp(op, "<", info);
    }
    return failure();
  }

  //===--------------------------------------------------------------------===//
  // Core theory operation visitors
  //===--------------------------------------------------------------------===//

  HANDLE_OP(EqOp, "=", Variadic);
  HANDLE_OP(DistinctOp, "distinct", Variadic);

  LogicalResult visitSMTOp(IteOp op, VisitorInfo &info) {
    info.stream << "(ite " << info.valueMap.lookup(op.getCond()) << " "
                << info.valueMap.lookup(op.getThenValue()) << " "
                << info.valueMap.lookup(op.getElseValue()) << ")";
    return success();
  }

  LogicalResult visitSMTOp(ApplyFuncOp op, VisitorInfo &info) {
    info.stream << "(" << info.valueMap.lookup(op.getFunc());
    for (Value arg : op.getArgs())
      info.stream << " " << info.valueMap.lookup(arg);
    info.stream << ")";
    return success();
  }

  template <typename OpTy>
  LogicalResult quantifierHelper(OpTy op, StringRef operatorString,
                                 VisitorInfo &info) {
    auto weight = op.getWeight();
    auto patterns = op.getPatterns();
    // TODO: add support
    if (op.getNoPattern())
      return op.emitError() << "no-pattern attribute not supported yet";

    llvm::ScopedHashTableScope<Value, std::string> scope(info.valueMap);
    info.stream << "(" << operatorString << " (";
    StringLiteral delimiter = "";

    SmallVector<StringRef> argNames;

    for (auto [i, arg] : llvm::enumerate(op.getBody().getArguments())) {
      // Generate and register a new unique name.
      StringRef prefix =
          op.getBoundVarNames()
              ? cast<StringAttr>(op.getBoundVarNames()->getValue()[i])
                    .getValue()
              : "tmp";
      StringRef name = names.newName(prefix);
      argNames.push_back(name);

      info.valueMap.insert(arg, name.str());

      // Print the bound variable declaration.
      info.stream << delimiter << "(" << name << " ";
      typeVisitor.dispatchSMTTypeVisitor(arg.getType(), info.stream);
      info.stream << ")";
      delimiter = " ";
    }

    info.stream << ")\n";

    // Print the quantifier body. This assumes that quantifiers are not deeply
    // nested (at least not enough that recursive calls could become a problem).

    SmallVector<Value> worklist;
    Value yieldedValue = op.getBody().front().getTerminator()->getOperand(0);
    worklist.push_back(yieldedValue);
    unsigned indentExt = operatorString.size() + 2;
    VisitorInfo newInfo(info.stream, info.valueMap,
                        info.indentLevel + indentExt, 0);
    if (weight != 0 || !patterns.empty())
      newInfo.stream.indent(newInfo.indentLevel);
    else
      newInfo.stream.indent(info.indentLevel);

    if (weight != 0 || !patterns.empty())
      info.stream << "( ! ";

    if (failed(printExpression(worklist, newInfo)))
      return failure();

    info.stream << info.valueMap.lookup(yieldedValue);

    for (unsigned j = 0; j < newInfo.openParens; ++j)
      info.stream << ")";

    if (weight != 0)
      info.stream << " :weight " << weight;
    if (!patterns.empty()) {
      bool first = true;
      info.stream << "\n:pattern (";
      for (auto &p : patterns) {

        if (!first)
          info.stream << " ";

        // retrieve argument name from the body region
        for (auto [i, arg] : llvm::enumerate(p.getArguments()))
          info.valueMap.insert(arg, argNames[i].str());

        SmallVector<Value> worklist;

        // retrieve all yielded operands in pattern region
        for (auto yieldedValue : p.front().getTerminator()->getOperands()) {

          worklist.push_back(yieldedValue);
          unsigned indentExt = operatorString.size() + 2;

          VisitorInfo newInfo2(info.stream, info.valueMap,
                               info.indentLevel + indentExt, 0);

          info.stream.indent(0);

          if (failed(printExpression(worklist, newInfo2)))
            return failure();

          info.stream << info.valueMap.lookup(yieldedValue);
          for (unsigned j = 0; j < newInfo2.openParens; ++j)
            info.stream << ")";
        }

        first = false;
      }
      info.stream << ")";
    }

    if (weight != 0 || !patterns.empty())
      info.stream << ")";

    info.stream << ")";

    return success();
  }

  LogicalResult visitSMTOp(ForallOp op, VisitorInfo &info) {
    return quantifierHelper(op, "forall", info);
  }

  LogicalResult visitSMTOp(ExistsOp op, VisitorInfo &info) {
    return quantifierHelper(op, "exists", info);
  }

  LogicalResult visitSMTOp(NotOp op, VisitorInfo &info) {
    info.stream << "(not " << info.valueMap.lookup(op.getInput()) << ")";
    return success();
  }

  HANDLE_OP(AndOp, "and", Variadic);
  HANDLE_OP(OrOp, "or", Variadic);
  HANDLE_OP(XOrOp, "xor", Variadic);
  HANDLE_OP(ImpliesOp, "=>", Binary);

  //===--------------------------------------------------------------------===//
  // Array theory operation visitors
  //===--------------------------------------------------------------------===//

  LogicalResult visitSMTOp(ArrayStoreOp op, VisitorInfo &info) {
    info.stream << "(store " << info.valueMap.lookup(op.getArray()) << " "
                << info.valueMap.lookup(op.getIndex()) << " "
                << info.valueMap.lookup(op.getValue()) << ")";
    return success();
  }

  LogicalResult visitSMTOp(ArraySelectOp op, VisitorInfo &info) {
    info.stream << "(select " << info.valueMap.lookup(op.getArray()) << " "
                << info.valueMap.lookup(op.getIndex()) << ")";
    return success();
  }

  LogicalResult visitSMTOp(ArrayBroadcastOp op, VisitorInfo &info) {
    info.stream << "((as const ";
    typeVisitor.dispatchSMTTypeVisitor(op.getType(), info.stream);
    info.stream << ") " << info.valueMap.lookup(op.getValue()) << ")";
    return success();
  }

  LogicalResult visitUnhandledSMTOp(Operation *op, VisitorInfo &info) {
    return success();
  }

#undef HANDLE_OP

  /// Print an expression transitively. The root node should be added to the
  /// 'worklist' before calling.
  LogicalResult printExpression(SmallVector<Value> &worklist,
                                VisitorInfo &info) {
    while (!worklist.empty()) {
      Value curr = worklist.back();

      // If we already have a let-binding for the value, just print it.
      if (info.valueMap.count(curr)) {
        worklist.pop_back();
        continue;
      }

      // Traverse until we reach a value/operation that has all operands
      // available and can thus be printed.
      bool allAvailable = true;
      Operation *defOp = curr.getDefiningOp();
      assert(defOp != nullptr &&
             "block arguments must already be in the valueMap");

      for (Value val : defOp->getOperands()) {
        if (!info.valueMap.count(val)) {
          worklist.push_back(val);
          allAvailable = false;
        }
      }

      if (!allAvailable)
        continue;

      if (failed(dispatchSMTOpVisitor(curr.getDefiningOp(), info)))
        return failure();

      worklist.pop_back();
    }

    return success();
  }

private:
  // A reference to the emission options for easy use in the visitor methods.
  [[maybe_unused]] const SMTEmissionOptions &options;
  TypeVisitor typeVisitor;
  Namespace &names;
};

/// A visitor to print SMT dialect operations with zero result values or
/// ones that have to initialize some global state.
struct StatementVisitor
    : public smt::SMTOpVisitor<StatementVisitor, LogicalResult,
                               mlir::raw_indented_ostream &, ValueMap &> {
  using smt::SMTOpVisitor<StatementVisitor, LogicalResult,
                          mlir::raw_indented_ostream &, ValueMap &>::visitSMTOp;

  StatementVisitor(const SMTEmissionOptions &options, Namespace &names)
      : options(options), typeVisitor(options), names(names),
        exprVisitor(options, names) {}

  LogicalResult visitSMTOp(BVConstantOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    valueMap.insert(op.getResult(), op.getValue().getValueAsString());
    return success();
  }

  LogicalResult visitSMTOp(BoolConstantOp op,
                           mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    valueMap.insert(op.getResult(), op.getValue() ? "true" : "false");
    return success();
  }

  LogicalResult visitSMTOp(IntConstantOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    SmallString<16> str;
    op.getValue().toStringSigned(str);
    valueMap.insert(op.getResult(), str.str().str());
    return success();
  }

  LogicalResult visitSMTOp(DeclareFunOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    StringRef name =
        names.newName(op.getNamePrefix() ? *op.getNamePrefix() : "tmp");
    valueMap.insert(op.getResult(), name.str());
    stream << "("
           << (isa<SMTFuncType>(op.getType()) ? "declare-fun "
                                              : "declare-const ")
           << name << " ";
    typeVisitor.dispatchSMTTypeVisitor(op.getType(), stream);
    stream << ")\n";
    return success();
  }

  LogicalResult visitSMTOp(AssertOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    llvm::ScopedHashTableScope<Value, std::string> scope1(valueMap);
    SmallVector<Value> worklist;
    worklist.push_back(op.getInput());
    stream << "(assert ";
    VisitorInfo info(stream, valueMap, 8, 0);
    if (failed(exprVisitor.printExpression(worklist, info)))
      return failure();
    stream << valueMap.lookup(op.getInput());
    for (unsigned i = 0; i < info.openParens + 1; ++i)
      stream << ")";
    stream << "\n";
    stream.indent(0);
    return success();
  }

  LogicalResult visitSMTOp(ResetOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    stream << "(reset)\n";
    return success();
  }

  LogicalResult visitSMTOp(PushOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    stream << "(push " << op.getCount() << ")\n";
    return success();
  }

  LogicalResult visitSMTOp(PopOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    stream << "(pop " << op.getCount() << ")\n";
    return success();
  }

  LogicalResult visitSMTOp(CheckOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    if (op->getNumResults() != 0)
      return op.emitError() << "must not have any result values";

    if (op.getSatRegion().front().getOperations().size() != 1)
      return op->emitError() << "'sat' region must be empty";
    if (op.getUnknownRegion().front().getOperations().size() != 1)
      return op->emitError() << "'unknown' region must be empty";
    if (op.getUnsatRegion().front().getOperations().size() != 1)
      return op->emitError() << "'unsat' region must be empty";

    stream << "(check-sat)\n";
    return success();
  }

  LogicalResult visitSMTOp(SetLogicOp op, mlir::raw_indented_ostream &stream,
                           ValueMap &valueMap) {
    stream << "(set-logic " << op.getLogic() << ")\n";
    return success();
  }

  LogicalResult visitUnhandledSMTOp(Operation *op,
                                    mlir::raw_indented_ostream &stream,
                                    ValueMap &valueMap) {
    // Ignore operations which are handled in the Expression Visitor.
    if (isa<smt::Int2BVOp, BV2IntOp>(op))
      return op->emitError("operation not supported for SMTLIB emission");

    return success();
  }

private:
  // A reference to the emission options for easy use in the visitor methods.
  [[maybe_unused]] const SMTEmissionOptions &options;
  TypeVisitor typeVisitor;
  Namespace &names;
  ExpressionVisitor exprVisitor;
};

} // namespace

//===----------------------------------------------------------------------===//
// Unified Emitter implementation
//===----------------------------------------------------------------------===//

/// Emit the SMT operations in the given 'solver' to the 'stream'.
static LogicalResult emit(SolverOp solver, const SMTEmissionOptions &options,
                          mlir::raw_indented_ostream &stream) {
  if (!solver.getInputs().empty() || solver->getNumResults() != 0)
    return solver->emitError()
           << "solver scopes with inputs or results are not supported";

  Block *block = solver.getBody();

  // Declare uninterpreted sorts.
  DenseMap<StringAttr, unsigned> declaredSorts;
  auto result = block->walk([&](Operation *op) -> WalkResult {
    if (!isa<SMTDialect>(op->getDialect()))
      return op->emitError()
             << "solver must not contain any non-SMT operations";

    for (Type resTy : op->getResultTypes()) {
      auto sortTy = dyn_cast<SortType>(resTy);
      if (!sortTy)
        continue;

      unsigned arity = sortTy.getSortParams().size();
      if (declaredSorts.contains(sortTy.getIdentifier())) {
        if (declaredSorts[sortTy.getIdentifier()] != arity)
          return op->emitError("uninterpreted sorts with same identifier but "
                               "different arity found");

        continue;
      }

      declaredSorts[sortTy.getIdentifier()] = arity;
      stream << "(declare-sort " << sortTy.getIdentifier().getValue() << " "
             << arity << ")\n";
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  ValueMap valueMap;
  llvm::ScopedHashTableScope<Value, std::string> scope0(valueMap);
  Namespace names;
  StatementVisitor visitor(options, names);

  // Collect all statement operations (ops with no result value).
  // Declare constants and then only refer to them by identifier later on.
  result = block->walk([&](Operation *op) {
    if (failed(visitor.dispatchSMTOpVisitor(op, stream, valueMap)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  stream << "(reset)\n";
  return success();
}

LogicalResult smt::exportSMTLIB(Operation *module, llvm::raw_ostream &os,
                                const SMTEmissionOptions &options) {
  if (module->getNumRegions() != 1)
    return module->emitError("must have exactly one region");
  if (!module->getRegion(0).hasOneBlock())
    return module->emitError("op region must have exactly one block");

  mlir::raw_indented_ostream ios(os);
  unsigned solverIdx = 0;
  auto result = module->walk([&](SolverOp solver) {
    ios << "; solver scope " << solverIdx << "\n";
    if (failed(emit(solver, options, ios)))
      return WalkResult::interrupt();
    ++solverIdx;
    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// mlir-translate registration
//===----------------------------------------------------------------------===//

void smt::registerExportSMTLIBTranslation() {
  static llvm::cl::opt<bool> inlineSingleUseValues(
      "smtlibexport-inline-single-use-values",
      llvm::cl::desc("Inline expressions that are used only once rather than "
                     "generating a let-binding"),
      llvm::cl::init(false));

  auto getOptions = [] {
    SMTEmissionOptions opts;
    opts.inlineSingleUseValues = inlineSingleUseValues;
    return opts;
  };

  static mlir::TranslateFromMLIRRegistration toSMTLIB(
      "export-smtlib", "export SMT-LIB",
      [=](Operation *module, raw_ostream &output) {
        return smt::exportSMTLIB(module, output, getOptions());
      },
      [](mlir::DialectRegistry &registry) {
        // Register the 'func' and 'HW' dialects to support printing solver
        // scopes nested in functions and modules.
        registry.insert<mlir::func::FuncDialect, arith::ArithDialect,
                        smt::SMTDialect>();
      });
}
