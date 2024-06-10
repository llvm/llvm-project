//===- RewriterGen.cpp - MLIR pattern rewriter generator ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RewriterGen uses pattern rewrite definitions to generate rewriter matchers.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/IndentedOstream.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/CodeGenHelpers.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pattern.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Property.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace mlir;
using namespace mlir::tblgen;

using llvm::formatv;
using llvm::Record;
using llvm::RecordKeeper;

#define DEBUG_TYPE "mlir-tblgen-rewritergen"

namespace llvm {
template <>
struct format_provider<mlir::tblgen::Pattern::IdentifierLine> {
  static void format(const mlir::tblgen::Pattern::IdentifierLine &v,
                     raw_ostream &os, StringRef style) {
    os << v.first << ":" << v.second;
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// PatternEmitter
//===----------------------------------------------------------------------===//

namespace {

class StaticMatcherHelper;

class PatternEmitter {
public:
  PatternEmitter(Record *pat, RecordOperatorMap *mapper, raw_ostream &os,
                 StaticMatcherHelper &helper);

  // Emits the mlir::RewritePattern struct named `rewriteName`.
  void emit(StringRef rewriteName);

  // Emits the static function of DAG matcher.
  void emitStaticMatcher(DagNode tree, std::string funcName);

private:
  // Emits the code for matching ops.
  void emitMatchLogic(DagNode tree, StringRef opName);

  // Emits the code for rewriting ops.
  void emitRewriteLogic();

  //===--------------------------------------------------------------------===//
  // Match utilities
  //===--------------------------------------------------------------------===//

  // Emits C++ statements for matching the DAG structure.
  void emitMatch(DagNode tree, StringRef name, int depth);

  // Emit C++ function call to static DAG matcher.
  void emitStaticMatchCall(DagNode tree, StringRef name);

  // Emit C++ function call to static type/attribute constraint function.
  void emitStaticVerifierCall(StringRef funcName, StringRef opName,
                              StringRef arg, StringRef failureStr);

  // Emits C++ statements for matching using a native code call.
  void emitNativeCodeMatch(DagNode tree, StringRef name, int depth);

  // Emits C++ statements for matching the op constrained by the given DAG
  // `tree` returning the op's variable name.
  void emitOpMatch(DagNode tree, StringRef opName, int depth);

  // Emits C++ statements for matching the `argIndex`-th argument of the given
  // DAG `tree` as an operand. `operandName` and `operandMatcher` indicate the
  // bound name and the constraint of the operand respectively.
  void emitOperandMatch(DagNode tree, StringRef opName, StringRef operandName,
                        int operandIndex, DagLeaf operandMatcher,
                        StringRef argName, int argIndex,
                        std::optional<int> variadicSubIndex);

  // Emits C++ statements for matching the operands which can be matched in
  // either order.
  void emitEitherOperandMatch(DagNode tree, DagNode eitherArgTree,
                              StringRef opName, int argIndex, int &operandIndex,
                              int depth);

  // Emits C++ statements for matching a variadic operand.
  void emitVariadicOperandMatch(DagNode tree, DagNode variadicArgTree,
                                StringRef opName, int argIndex,
                                int &operandIndex, int depth);

  // Emits C++ statements for matching the `argIndex`-th argument of the given
  // DAG `tree` as an attribute.
  void emitAttributeMatch(DagNode tree, StringRef opName, int argIndex,
                          int depth);

  // Emits C++ for checking a match with a corresponding match failure
  // diagnostic.
  void emitMatchCheck(StringRef opName, const FmtObjectBase &matchFmt,
                      const llvm::formatv_object_base &failureFmt);

  // Emits C++ for checking a match with a corresponding match failure
  // diagnostics.
  void emitMatchCheck(StringRef opName, const std::string &matchStr,
                      const std::string &failureStr);

  //===--------------------------------------------------------------------===//
  // Rewrite utilities
  //===--------------------------------------------------------------------===//

  // The entry point for handling a result pattern rooted at `resultTree`. This
  // method dispatches to concrete handlers according to `resultTree`'s kind and
  // returns a symbol representing the whole value pack. Callers are expected to
  // further resolve the symbol according to the specific use case.
  //
  // `depth` is the nesting level of `resultTree`; 0 means top-level result
  // pattern. For top-level result pattern, `resultIndex` indicates which result
  // of the matched root op this pattern is intended to replace, which can be
  // used to deduce the result type of the op generated from this result
  // pattern.
  std::string handleResultPattern(DagNode resultTree, int resultIndex,
                                  int depth);

  // Emits the C++ statement to replace the matched DAG with a value built via
  // calling native C++ code.
  std::string handleReplaceWithNativeCodeCall(DagNode resultTree, int depth);

  // Returns the symbol of the old value serving as the replacement.
  StringRef handleReplaceWithValue(DagNode tree);

  // Emits the C++ statement to replace the matched DAG with an array of
  // matched values.
  std::string handleVariadic(DagNode tree, int depth);

  // Trailing directives are used at the end of DAG node argument lists to
  // specify additional behaviour for op matchers and creators, etc.
  struct TrailingDirectives {
    // DAG node containing the `location` directive. Null if there is none.
    DagNode location;

    // DAG node containing the `returnType` directive. Null if there is none.
    DagNode returnType;

    // Number of found trailing directives.
    int numDirectives;
  };

  // Collect any trailing directives.
  TrailingDirectives getTrailingDirectives(DagNode tree);

  // Returns the location value to use.
  std::string getLocation(TrailingDirectives &tail);

  // Returns the location value to use.
  std::string handleLocationDirective(DagNode tree);

  // Emit return type argument.
  std::string handleReturnTypeArg(DagNode returnType, int i, int depth);

  // Emits the C++ statement to build a new op out of the given DAG `tree` and
  // returns the variable name that this op is assigned to. If the root op in
  // DAG `tree` has a specified name, the created op will be assigned to a
  // variable of the given name. Otherwise, a unique name will be used as the
  // result value name.
  std::string handleOpCreation(DagNode tree, int resultIndex, int depth);

  using ChildNodeIndexNameMap = DenseMap<unsigned, std::string>;

  // Emits a local variable for each value and attribute to be used for creating
  // an op.
  void createSeparateLocalVarsForOpArgs(DagNode node,
                                        ChildNodeIndexNameMap &childNodeNames);

  // Emits the concrete arguments used to call an op's builder.
  void supplyValuesForOpArgs(DagNode node,
                             const ChildNodeIndexNameMap &childNodeNames,
                             int depth);

  // Emits the local variables for holding all values as a whole and all named
  // attributes as a whole to be used for creating an op.
  void createAggregateLocalVarsForOpArgs(
      DagNode node, const ChildNodeIndexNameMap &childNodeNames, int depth);

  // Returns the C++ expression to construct a constant attribute of the given
  // `value` for the given attribute kind `attr`.
  std::string handleConstantAttr(Attribute attr, const Twine &value);

  // Returns the C++ expression to build an argument from the given DAG `leaf`.
  // `patArgName` is used to bound the argument to the source pattern.
  std::string handleOpArgument(DagLeaf leaf, StringRef patArgName);

  //===--------------------------------------------------------------------===//
  // General utilities
  //===--------------------------------------------------------------------===//

  // Collects all of the operations within the given dag tree.
  void collectOps(DagNode tree, llvm::SmallPtrSetImpl<const Operator *> &ops);

  // Returns a unique symbol for a local variable of the given `op`.
  std::string getUniqueSymbol(const Operator *op);

  //===--------------------------------------------------------------------===//
  // Symbol utilities
  //===--------------------------------------------------------------------===//

  // Returns how many static values the given DAG `node` correspond to.
  int getNodeValueCount(DagNode node);

private:
  // Pattern instantiation location followed by the location of multiclass
  // prototypes used. This is intended to be used as a whole to
  // PrintFatalError() on errors.
  ArrayRef<SMLoc> loc;

  // Op's TableGen Record to wrapper object.
  RecordOperatorMap *opMap;

  // Handy wrapper for pattern being emitted.
  Pattern pattern;

  // Map for all bound symbols' info.
  SymbolInfoMap symbolInfoMap;

  StaticMatcherHelper &staticMatcherHelper;

  // The next unused ID for newly created values.
  unsigned nextValueId = 0;

  raw_indented_ostream os;

  // Format contexts containing placeholder substitutions.
  FmtContext fmtCtx;
};

// Tracks DagNode's reference multiple times across patterns. Enables generating
// static matcher functions for DagNode's referenced multiple times rather than
// inlining them.
class StaticMatcherHelper {
public:
  StaticMatcherHelper(raw_ostream &os, const RecordKeeper &recordKeeper,
                      RecordOperatorMap &mapper);

  // Determine if we should inline the match logic or delegate to a static
  // function.
  bool useStaticMatcher(DagNode node) {
    // either/variadic node must be associated to the parentOp, thus we can't
    // emit a static matcher rooted at them.
    if (node.isEither() || node.isVariadic())
      return false;

    return refStats[node] > kStaticMatcherThreshold;
  }

  // Get the name of the static DAG matcher function corresponding to the node.
  std::string getMatcherName(DagNode node) {
    assert(useStaticMatcher(node));
    return matcherNames[node];
  }

  // Get the name of static type/attribute verification function.
  StringRef getVerifierName(DagLeaf leaf);

  // Collect the `Record`s, i.e., the DRR, so that we can get the information of
  // the duplicated DAGs.
  void addPattern(Record *record);

  // Emit all static functions of DAG Matcher.
  void populateStaticMatchers(raw_ostream &os);

  // Emit all static functions for Constraints.
  void populateStaticConstraintFunctions(raw_ostream &os);

private:
  static constexpr unsigned kStaticMatcherThreshold = 1;

  // Consider two patterns as down below,
  //   DagNode_Root_A    DagNode_Root_B
  //       \                 \
  //     DagNode_C         DagNode_C
  //         \                 \
  //       DagNode_D         DagNode_D
  //
  // DagNode_Root_A and DagNode_Root_B share the same subtree which consists of
  // DagNode_C and DagNode_D. Both DagNode_C and DagNode_D are referenced
  // multiple times so we'll have static matchers for both of them. When we're
  // emitting the match logic for DagNode_C, we will check if DagNode_D has the
  // static matcher generated. If so, then we'll generate a call to the
  // function, inline otherwise. In this case, inlining is not what we want. As
  // a result, generate the static matcher in topological order to ensure all
  // the dependent static matchers are generated and we can avoid accidentally
  // inlining.
  //
  // The topological order of all the DagNodes among all patterns.
  SmallVector<std::pair<DagNode, Record *>> topologicalOrder;

  RecordOperatorMap &opMap;

  // Records of the static function name of each DagNode
  DenseMap<DagNode, std::string> matcherNames;

  // After collecting all the DagNode in each pattern, `refStats` records the
  // number of users for each DagNode. We will generate the static matcher for a
  // DagNode while the number of users exceeds a certain threshold.
  DenseMap<DagNode, unsigned> refStats;

  // Number of static matcher generated. This is used to generate a unique name
  // for each DagNode.
  int staticMatcherCounter = 0;

  // The DagLeaf which contains type or attr constraint.
  SetVector<DagLeaf> constraints;

  // Static type/attribute verification function emitter.
  StaticVerifierFunctionEmitter staticVerifierEmitter;
};

} // namespace

PatternEmitter::PatternEmitter(Record *pat, RecordOperatorMap *mapper,
                               raw_ostream &os, StaticMatcherHelper &helper)
    : loc(pat->getLoc()), opMap(mapper), pattern(pat, mapper),
      symbolInfoMap(pat->getLoc()), staticMatcherHelper(helper), os(os) {
  fmtCtx.withBuilder("rewriter");
}

std::string PatternEmitter::handleConstantAttr(Attribute attr,
                                               const Twine &value) {
  if (!attr.isConstBuildable())
    PrintFatalError(loc, "Attribute " + attr.getAttrDefName() +
                             " does not have the 'constBuilderCall' field");

  // TODO: Verify the constants here
  return std::string(tgfmt(attr.getConstBuilderTemplate(), &fmtCtx, value));
}

void PatternEmitter::emitStaticMatcher(DagNode tree, std::string funcName) {
  os << formatv(
      "static ::mlir::LogicalResult {0}(::mlir::PatternRewriter &rewriter, "
      "::mlir::Operation *op0, ::llvm::SmallVector<::mlir::Operation "
      "*, 4> &tblgen_ops",
      funcName);

  // We pass the reference of the variables that need to be captured. Hence we
  // need to collect all the symbols in the tree first.
  pattern.collectBoundSymbols(tree, symbolInfoMap, /*isSrcPattern=*/true);
  symbolInfoMap.assignUniqueAlternativeNames();
  for (const auto &info : symbolInfoMap)
    os << formatv(", {0}", info.second.getArgDecl(info.first));

  os << ") {\n";
  os.indent();
  os << "(void)tblgen_ops;\n";

  // Note that a static matcher is considered at least one step from the match
  // entry.
  emitMatch(tree, "op0", /*depth=*/1);

  os << "return ::mlir::success();\n";
  os.unindent();
  os << "}\n\n";
}

// Helper function to match patterns.
void PatternEmitter::emitMatch(DagNode tree, StringRef name, int depth) {
  if (tree.isNativeCodeCall()) {
    emitNativeCodeMatch(tree, name, depth);
    return;
  }

  if (tree.isOperation()) {
    emitOpMatch(tree, name, depth);
    return;
  }

  PrintFatalError(loc, "encountered non-op, non-NativeCodeCall match.");
}

void PatternEmitter::emitStaticMatchCall(DagNode tree, StringRef opName) {
  std::string funcName = staticMatcherHelper.getMatcherName(tree);
  os << formatv("if(::mlir::failed({0}(rewriter, {1}, tblgen_ops", funcName,
                opName);

  // TODO(chiahungduan): Add a lookupBoundSymbols() to do the subtree lookup in
  // one pass.

  // In general, bound symbol should have the unique name in the pattern but
  // for the operand, binding same symbol to multiple operands imply a
  // constraint at the same time. In this case, we will rename those operands
  // with different names. As a result, we need to collect all the symbolInfos
  // from the DagNode then get the updated name of the local variables from the
  // global symbolInfoMap.

  // Collect all the bound symbols in the Dag
  SymbolInfoMap localSymbolMap(loc);
  pattern.collectBoundSymbols(tree, localSymbolMap, /*isSrcPattern=*/true);

  for (const auto &info : localSymbolMap) {
    auto name = info.first;
    auto symboInfo = info.second;
    auto ret = symbolInfoMap.findBoundSymbol(name, symboInfo);
    os << formatv(", {0}", ret->second.getVarName(name));
  }

  os << "))) {\n";
  os.scope().os << "return ::mlir::failure();\n";
  os << "}\n";
}

void PatternEmitter::emitStaticVerifierCall(StringRef funcName,
                                            StringRef opName, StringRef arg,
                                            StringRef failureStr) {
  os << formatv("if(::mlir::failed({0}(rewriter, {1}, {2}, {3}))) {{\n",
                funcName, opName, arg, failureStr);
  os.scope().os << "return ::mlir::failure();\n";
  os << "}\n";
}

// Helper function to match patterns.
void PatternEmitter::emitNativeCodeMatch(DagNode tree, StringRef opName,
                                         int depth) {
  LLVM_DEBUG(llvm::dbgs() << "handle NativeCodeCall matcher pattern: ");
  LLVM_DEBUG(tree.print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');

  // The order of generating static matcher follows the topological order so
  // that for every dependent DagNode already have their static matcher
  // generated if needed. The reason we check if `getMatcherName(tree).empty()`
  // is when we are generating the static matcher for a DagNode itself. In this
  // case, we need to emit the function body rather than a function call.
  if (staticMatcherHelper.useStaticMatcher(tree) &&
      !staticMatcherHelper.getMatcherName(tree).empty()) {
    emitStaticMatchCall(tree, opName);

    // NativeCodeCall will never be at depth 0 so that we don't need to catch
    // the root operation as emitOpMatch();

    return;
  }

  // TODO(suderman): iterate through arguments, determine their types, output
  // names.
  SmallVector<std::string, 8> capture;

  raw_indented_ostream::DelimitedScope scope(os);

  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    std::string argName = formatv("arg{0}_{1}", depth, i);
    if (DagNode argTree = tree.getArgAsNestedDag(i)) {
      if (argTree.isEither())
        PrintFatalError(loc, "NativeCodeCall cannot have `either` operands");
      if (argTree.isVariadic())
        PrintFatalError(loc, "NativeCodeCall cannot have `variadic` operands");

      os << "::mlir::Value " << argName << ";\n";
    } else {
      auto leaf = tree.getArgAsLeaf(i);
      if (leaf.isAttrMatcher() || leaf.isConstantAttr()) {
        os << "::mlir::Attribute " << argName << ";\n";
      } else {
        os << "::mlir::Value " << argName << ";\n";
      }
    }

    capture.push_back(std::move(argName));
  }

  auto tail = getTrailingDirectives(tree);
  if (tail.returnType)
    PrintFatalError(loc, "`NativeCodeCall` cannot have return type specifier");
  auto locToUse = getLocation(tail);

  auto fmt = tree.getNativeCodeTemplate();
  if (fmt.count("$_self") != 1)
    PrintFatalError(loc, "NativeCodeCall must have $_self as argument for "
                         "passing the defining Operation");

  auto nativeCodeCall = std::string(
      tgfmt(fmt, &fmtCtx.addSubst("_loc", locToUse).withSelf(opName.str()),
            static_cast<ArrayRef<std::string>>(capture)));

  emitMatchCheck(opName, formatv("!::mlir::failed({0})", nativeCodeCall),
                 formatv("\"{0} return ::mlir::failure\"", nativeCodeCall));

  for (int i = 0, e = tree.getNumArgs() - tail.numDirectives; i != e; ++i) {
    auto name = tree.getArgName(i);
    if (!name.empty() && name != "_") {
      os << formatv("{0} = {1};\n", name, capture[i]);
    }
  }

  for (int i = 0, e = tree.getNumArgs() - tail.numDirectives; i != e; ++i) {
    std::string argName = capture[i];

    // Handle nested DAG construct first
    if (tree.getArgAsNestedDag(i)) {
      PrintFatalError(
          loc, formatv("Matching nested tree in NativeCodecall not support for "
                       "{0} as arg {1}",
                       argName, i));
    }

    DagLeaf leaf = tree.getArgAsLeaf(i);

    // The parameter for native function doesn't bind any constraints.
    if (leaf.isUnspecified())
      continue;

    auto constraint = leaf.getAsConstraint();

    std::string self;
    if (leaf.isAttrMatcher() || leaf.isConstantAttr())
      self = argName;
    else
      self = formatv("{0}.getType()", argName);
    StringRef verifier = staticMatcherHelper.getVerifierName(leaf);
    emitStaticVerifierCall(
        verifier, opName, self,
        formatv("\"operand {0} of native code call '{1}' failed to satisfy "
                "constraint: "
                "'{2}'\"",
                i, tree.getNativeCodeTemplate(),
                escapeString(constraint.getSummary()))
            .str());
  }

  LLVM_DEBUG(llvm::dbgs() << "done emitting match for native code call\n");
}

// Helper function to match patterns.
void PatternEmitter::emitOpMatch(DagNode tree, StringRef opName, int depth) {
  Operator &op = tree.getDialectOp(opMap);
  LLVM_DEBUG(llvm::dbgs() << "start emitting match for op '"
                          << op.getOperationName() << "' at depth " << depth
                          << '\n');

  auto getCastedName = [depth]() -> std::string {
    return formatv("castedOp{0}", depth);
  };

  // The order of generating static matcher follows the topological order so
  // that for every dependent DagNode already have their static matcher
  // generated if needed. The reason we check if `getMatcherName(tree).empty()`
  // is when we are generating the static matcher for a DagNode itself. In this
  // case, we need to emit the function body rather than a function call.
  if (staticMatcherHelper.useStaticMatcher(tree) &&
      !staticMatcherHelper.getMatcherName(tree).empty()) {
    emitStaticMatchCall(tree, opName);
    // In the codegen of rewriter, we suppose that castedOp0 will capture the
    // root operation. Manually add it if the root DagNode is a static matcher.
    if (depth == 0)
      os << formatv("auto {2} = ::llvm::dyn_cast_or_null<{1}>({0}); "
                    "(void){2};\n",
                    opName, op.getQualCppClassName(), getCastedName());
    return;
  }

  std::string castedName = getCastedName();
  os << formatv("auto {0} = ::llvm::dyn_cast<{2}>({1}); "
                "(void){0};\n",
                castedName, opName, op.getQualCppClassName());

  // Skip the operand matching at depth 0 as the pattern rewriter already does.
  if (depth != 0)
    emitMatchCheck(opName, /*matchStr=*/castedName,
                   formatv("\"{0} is not {1} type\"", castedName,
                           op.getQualCppClassName()));

  // If the operand's name is set, set to that variable.
  auto name = tree.getSymbol();
  if (!name.empty())
    os << formatv("{0} = {1};\n", name, castedName);

  for (int i = 0, opArgIdx = 0, e = tree.getNumArgs(), nextOperand = 0; i != e;
       ++i, ++opArgIdx) {
    auto opArg = op.getArg(opArgIdx);
    std::string argName = formatv("op{0}", depth + 1);

    // Handle nested DAG construct first
    if (DagNode argTree = tree.getArgAsNestedDag(i)) {
      if (argTree.isEither()) {
        emitEitherOperandMatch(tree, argTree, castedName, opArgIdx, nextOperand,
                               depth);
        ++opArgIdx;
        continue;
      }
      if (auto *operand = llvm::dyn_cast_if_present<NamedTypeConstraint *>(opArg)) {
        if (argTree.isVariadic()) {
          if (!operand->isVariadic()) {
            auto error = formatv("variadic DAG construct can't match op {0}'s "
                                 "non-variadic operand #{1}",
                                 op.getOperationName(), opArgIdx);
            PrintFatalError(loc, error);
          }
          emitVariadicOperandMatch(tree, argTree, castedName, opArgIdx,
                                   nextOperand, depth);
          ++nextOperand;
          continue;
        }
        if (operand->isVariableLength()) {
          auto error = formatv("use nested DAG construct to match op {0}'s "
                               "variadic operand #{1} unsupported now",
                               op.getOperationName(), opArgIdx);
          PrintFatalError(loc, error);
        }
      }

      os << "{\n";

      // Attributes don't count for getODSOperands.
      // TODO: Operand is a Value, check if we should remove `getDefiningOp()`.
      os.indent() << formatv(
          "auto *{0} = "
          "(*{1}.getODSOperands({2}).begin()).getDefiningOp();\n",
          argName, castedName, nextOperand);
      // Null check of operand's definingOp
      emitMatchCheck(
          castedName, /*matchStr=*/argName,
          formatv("\"There's no operation that defines operand {0} of {1}\"",
                  nextOperand++, castedName));
      emitMatch(argTree, argName, depth + 1);
      os << formatv("tblgen_ops.push_back({0});\n", argName);
      os.unindent() << "}\n";
      continue;
    }

    // Next handle DAG leaf: operand or attribute
    if (opArg.is<NamedTypeConstraint *>()) {
      auto operandName =
          formatv("{0}.getODSOperands({1})", castedName, nextOperand);
      emitOperandMatch(tree, castedName, operandName.str(), opArgIdx,
                       /*operandMatcher=*/tree.getArgAsLeaf(i),
                       /*argName=*/tree.getArgName(i), opArgIdx,
                       /*variadicSubIndex=*/std::nullopt);
      ++nextOperand;
    } else if (opArg.is<NamedAttribute *>()) {
      emitAttributeMatch(tree, opName, opArgIdx, depth);
    } else {
      PrintFatalError(loc, "unhandled case when matching op");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "done emitting match for op '"
                          << op.getOperationName() << "' at depth " << depth
                          << '\n');
}

void PatternEmitter::emitOperandMatch(DagNode tree, StringRef opName,
                                      StringRef operandName, int operandIndex,
                                      DagLeaf operandMatcher, StringRef argName,
                                      int argIndex,
                                      std::optional<int> variadicSubIndex) {
  Operator &op = tree.getDialectOp(opMap);
  auto *operand = op.getArg(operandIndex).get<NamedTypeConstraint *>();

  // If a constraint is specified, we need to generate C++ statements to
  // check the constraint.
  if (!operandMatcher.isUnspecified()) {
    if (!operandMatcher.isOperandMatcher())
      PrintFatalError(
          loc, formatv("the {1}-th argument of op '{0}' should be an operand",
                       op.getOperationName(), argIndex + 1));

    // Only need to verify if the matcher's type is different from the one
    // of op definition.
    Constraint constraint = operandMatcher.getAsConstraint();
    if (operand->constraint != constraint) {
      if (operand->isVariableLength()) {
        auto error = formatv(
            "further constrain op {0}'s variadic operand #{1} unsupported now",
            op.getOperationName(), argIndex);
        PrintFatalError(loc, error);
      }
      auto self = formatv("(*{0}.begin()).getType()", operandName);
      StringRef verifier = staticMatcherHelper.getVerifierName(operandMatcher);
      emitStaticVerifierCall(
          verifier, opName, self.str(),
          formatv(
              "\"operand {0} of op '{1}' failed to satisfy constraint: '{2}'\"",
              operand - op.operand_begin(), op.getOperationName(),
              escapeString(constraint.getSummary()))
              .str());
    }
  }

  // Capture the value
  // `$_` is a special symbol to ignore op argument matching.
  if (!argName.empty() && argName != "_") {
    auto res = symbolInfoMap.findBoundSymbol(argName, tree, op, operandIndex,
                                             variadicSubIndex);
    if (res == symbolInfoMap.end())
      PrintFatalError(loc, formatv("symbol not found: {0}", argName));

    os << formatv("{0} = {1};\n", res->second.getVarName(argName), operandName);
  }
}

void PatternEmitter::emitEitherOperandMatch(DagNode tree, DagNode eitherArgTree,
                                            StringRef opName, int argIndex,
                                            int &operandIndex, int depth) {
  constexpr int numEitherArgs = 2;
  if (eitherArgTree.getNumArgs() != numEitherArgs)
    PrintFatalError(loc, "`either` only supports grouping two operands");

  Operator &op = tree.getDialectOp(opMap);

  std::string codeBuffer;
  llvm::raw_string_ostream tblgenOps(codeBuffer);

  std::string lambda = formatv("eitherLambda{0}", depth);
  os << formatv(
      "auto {0} = [&](::mlir::OperandRange v0, ::mlir::OperandRange v1) {{\n",
      lambda);

  os.indent();

  for (int i = 0; i < numEitherArgs; ++i, ++argIndex) {
    if (DagNode argTree = eitherArgTree.getArgAsNestedDag(i)) {
      if (argTree.isEither())
        PrintFatalError(loc, "either cannot be nested");

      std::string argName = formatv("local_op_{0}", i).str();

      os << formatv("auto {0} = (*v{1}.begin()).getDefiningOp();\n", argName,
                    i);

      // Indent emitMatchCheck and emitMatch because they declare local
      // variables.
      os << "{\n";
      os.indent();

      emitMatchCheck(
          opName, /*matchStr=*/argName,
          formatv("\"There's no operation that defines operand {0} of {1}\"",
                  operandIndex++, opName));
      emitMatch(argTree, argName, depth + 1);

      os.unindent() << "}\n";

      // `tblgen_ops` is used to collect the matched operations. In either, we
      // need to queue the operation only if the matching success. Thus we emit
      // the code at the end.
      tblgenOps << formatv("tblgen_ops.push_back({0});\n", argName);
    } else if (op.getArg(argIndex).is<NamedTypeConstraint *>()) {
      emitOperandMatch(tree, opName, /*operandName=*/formatv("v{0}", i).str(),
                       operandIndex,
                       /*operandMatcher=*/eitherArgTree.getArgAsLeaf(i),
                       /*argName=*/eitherArgTree.getArgName(i), argIndex,
                       /*variadicSubIndex=*/std::nullopt);
      ++operandIndex;
    } else {
      PrintFatalError(loc, "either can only be applied on operand");
    }
  }

  os << tblgenOps.str();
  os << "return ::mlir::success();\n";
  os.unindent() << "};\n";

  os << "{\n";
  os.indent();

  os << formatv("auto eitherOperand0 = {0}.getODSOperands({1});\n", opName,
                operandIndex - 2);
  os << formatv("auto eitherOperand1 = {0}.getODSOperands({1});\n", opName,
                operandIndex - 1);

  os << formatv("if(::mlir::failed({0}(eitherOperand0, eitherOperand1)) && "
                "::mlir::failed({0}(eitherOperand1, "
                "eitherOperand0)))\n",
                lambda);
  os.indent() << "return ::mlir::failure();\n";

  os.unindent().unindent() << "}\n";
}

void PatternEmitter::emitVariadicOperandMatch(DagNode tree,
                                              DagNode variadicArgTree,
                                              StringRef opName, int argIndex,
                                              int &operandIndex, int depth) {
  Operator &op = tree.getDialectOp(opMap);

  os << "{\n";
  os.indent();

  os << formatv("auto variadic_operand_range = {0}.getODSOperands({1});\n",
                opName, operandIndex);
  os << formatv("if (variadic_operand_range.size() != {0}) "
                "return ::mlir::failure();\n",
                variadicArgTree.getNumArgs());

  StringRef variadicTreeName = variadicArgTree.getSymbol();
  if (!variadicTreeName.empty()) {
    auto res =
        symbolInfoMap.findBoundSymbol(variadicTreeName, tree, op, operandIndex,
                                      /*variadicSubIndex=*/std::nullopt);
    if (res == symbolInfoMap.end())
      PrintFatalError(loc, formatv("symbol not found: {0}", variadicTreeName));

    os << formatv("{0} = variadic_operand_range;\n",
                  res->second.getVarName(variadicTreeName));
  }

  for (int i = 0; i < variadicArgTree.getNumArgs(); ++i) {
    if (DagNode argTree = variadicArgTree.getArgAsNestedDag(i)) {
      if (!argTree.isOperation())
        PrintFatalError(loc, "variadic only accepts operation sub-dags");

      os << "{\n";
      os.indent();

      std::string argName = formatv("local_op_{0}", i).str();
      os << formatv("auto *{0} = "
                    "variadic_operand_range[{1}].getDefiningOp();\n",
                    argName, i);
      emitMatchCheck(
          opName, /*matchStr=*/argName,
          formatv("\"There's no operation that defines variadic operand "
                  "{0} (variadic sub-opearnd #{1}) of {2}\"",
                  operandIndex, i, opName));
      emitMatch(argTree, argName, depth + 1);
      os << formatv("tblgen_ops.push_back({0});\n", argName);

      os.unindent() << "}\n";
    } else if (op.getArg(argIndex).is<NamedTypeConstraint *>()) {
      auto operandName = formatv("variadic_operand_range.slice({0}, 1)", i);
      emitOperandMatch(tree, opName, operandName.str(), operandIndex,
                       /*operandMatcher=*/variadicArgTree.getArgAsLeaf(i),
                       /*argName=*/variadicArgTree.getArgName(i), argIndex, i);
    } else {
      PrintFatalError(loc, "variadic can only be applied on operand");
    }
  }

  os.unindent() << "}\n";
}

void PatternEmitter::emitAttributeMatch(DagNode tree, StringRef opName,
                                        int argIndex, int depth) {
  Operator &op = tree.getDialectOp(opMap);
  auto *namedAttr = op.getArg(argIndex).get<NamedAttribute *>();
  const auto &attr = namedAttr->attr;

  os << "{\n";
  os.indent() << formatv("auto tblgen_attr = {0}->getAttrOfType<{1}>(\"{2}\");"
                         "(void)tblgen_attr;\n",
                         opName, attr.getStorageType(), namedAttr->name);

  // TODO: This should use getter method to avoid duplication.
  if (attr.hasDefaultValue()) {
    os << "if (!tblgen_attr) tblgen_attr = "
       << std::string(tgfmt(attr.getConstBuilderTemplate(), &fmtCtx,
                            attr.getDefaultValue()))
       << ";\n";
  } else if (attr.isOptional()) {
    // For a missing attribute that is optional according to definition, we
    // should just capture a mlir::Attribute() to signal the missing state.
    // That is precisely what getDiscardableAttr() returns on missing
    // attributes.
  } else {
    emitMatchCheck(opName, tgfmt("tblgen_attr", &fmtCtx),
                   formatv("\"expected op '{0}' to have attribute '{1}' "
                           "of type '{2}'\"",
                           op.getOperationName(), namedAttr->name,
                           attr.getStorageType()));
  }

  auto matcher = tree.getArgAsLeaf(argIndex);
  if (!matcher.isUnspecified()) {
    if (!matcher.isAttrMatcher()) {
      PrintFatalError(
          loc, formatv("the {1}-th argument of op '{0}' should be an attribute",
                       op.getOperationName(), argIndex + 1));
    }

    // If a constraint is specified, we need to generate function call to its
    // static verifier.
    StringRef verifier = staticMatcherHelper.getVerifierName(matcher);
    if (attr.isOptional()) {
      // Avoid dereferencing null attribute. This is using a simple heuristic to
      // avoid common cases of attempting to dereference null attribute. This
      // will return where there is no check if attribute is null unless the
      // attribute's value is not used.
      // FIXME: This could be improved as some null dereferences could slip
      // through.
      if (!StringRef(matcher.getConditionTemplate()).contains("!$_self") &&
          StringRef(matcher.getConditionTemplate()).contains("$_self")) {
        os << "if (!tblgen_attr) return ::mlir::failure();\n";
      }
    }
    emitStaticVerifierCall(
        verifier, opName, "tblgen_attr",
        formatv("\"op '{0}' attribute '{1}' failed to satisfy constraint: "
                "'{2}'\"",
                op.getOperationName(), namedAttr->name,
                escapeString(matcher.getAsConstraint().getSummary()))
            .str());
  }

  // Capture the value
  auto name = tree.getArgName(argIndex);
  // `$_` is a special symbol to ignore op argument matching.
  if (!name.empty() && name != "_") {
    os << formatv("{0} = tblgen_attr;\n", name);
  }

  os.unindent() << "}\n";
}

void PatternEmitter::emitMatchCheck(
    StringRef opName, const FmtObjectBase &matchFmt,
    const llvm::formatv_object_base &failureFmt) {
  emitMatchCheck(opName, matchFmt.str(), failureFmt.str());
}

void PatternEmitter::emitMatchCheck(StringRef opName,
                                    const std::string &matchStr,
                                    const std::string &failureStr) {

  os << "if (!(" << matchStr << "))";
  os.scope("{\n", "\n}\n").os << "return rewriter.notifyMatchFailure(" << opName
                              << ", [&](::mlir::Diagnostic &diag) {\n  diag << "
                              << failureStr << ";\n});";
}

void PatternEmitter::emitMatchLogic(DagNode tree, StringRef opName) {
  LLVM_DEBUG(llvm::dbgs() << "--- start emitting match logic ---\n");
  int depth = 0;
  emitMatch(tree, opName, depth);

  for (auto &appliedConstraint : pattern.getConstraints()) {
    auto &constraint = appliedConstraint.constraint;
    auto &entities = appliedConstraint.entities;

    auto condition = constraint.getConditionTemplate();
    if (isa<TypeConstraint>(constraint)) {
      if (entities.size() != 1)
        PrintFatalError(loc, "type constraint requires exactly one argument");

      auto self = formatv("({0}.getType())",
                          symbolInfoMap.getValueAndRangeUse(entities.front()));
      emitMatchCheck(
          opName, tgfmt(condition, &fmtCtx.withSelf(self.str())),
          formatv("\"value entity '{0}' failed to satisfy constraint: '{1}'\"",
                  entities.front(), escapeString(constraint.getSummary())));

    } else if (isa<AttrConstraint>(constraint)) {
      PrintFatalError(
          loc, "cannot use AttrConstraint in Pattern multi-entity constraints");
    } else {
      // TODO: replace formatv arguments with the exact specified
      // args.
      if (entities.size() > 4) {
        PrintFatalError(loc, "only support up to 4-entity constraints now");
      }
      SmallVector<std::string, 4> names;
      int i = 0;
      for (int e = entities.size(); i < e; ++i)
        names.push_back(symbolInfoMap.getValueAndRangeUse(entities[i]));
      std::string self = appliedConstraint.self;
      if (!self.empty())
        self = symbolInfoMap.getValueAndRangeUse(self);
      for (; i < 4; ++i)
        names.push_back("<unused>");
      emitMatchCheck(opName,
                     tgfmt(condition, &fmtCtx.withSelf(self), names[0],
                           names[1], names[2], names[3]),
                     formatv("\"entities '{0}' failed to satisfy constraint: "
                             "'{1}'\"",
                             llvm::join(entities, ", "),
                             escapeString(constraint.getSummary())));
    }
  }

  // Some of the operands could be bound to the same symbol name, we need
  // to enforce equality constraint on those.
  // TODO: we should be able to emit equality checks early
  // and short circuit unnecessary work if vars are not equal.
  for (auto symbolInfoIt = symbolInfoMap.begin();
       symbolInfoIt != symbolInfoMap.end();) {
    auto range = symbolInfoMap.getRangeOfEqualElements(symbolInfoIt->first);
    auto startRange = range.first;
    auto endRange = range.second;

    auto firstOperand = symbolInfoIt->second.getVarName(symbolInfoIt->first);
    for (++startRange; startRange != endRange; ++startRange) {
      auto secondOperand = startRange->second.getVarName(symbolInfoIt->first);
      emitMatchCheck(
          opName,
          formatv("*{0}.begin() == *{1}.begin()", firstOperand, secondOperand),
          formatv("\"Operands '{0}' and '{1}' must be equal\"", firstOperand,
                  secondOperand));
    }

    symbolInfoIt = endRange;
  }

  LLVM_DEBUG(llvm::dbgs() << "--- done emitting match logic ---\n");
}

void PatternEmitter::collectOps(DagNode tree,
                                llvm::SmallPtrSetImpl<const Operator *> &ops) {
  // Check if this tree is an operation.
  if (tree.isOperation()) {
    const Operator &op = tree.getDialectOp(opMap);
    LLVM_DEBUG(llvm::dbgs()
               << "found operation " << op.getOperationName() << '\n');
    ops.insert(&op);
  }

  // Recurse the arguments of the tree.
  for (unsigned i = 0, e = tree.getNumArgs(); i != e; ++i)
    if (auto child = tree.getArgAsNestedDag(i))
      collectOps(child, ops);
}

void PatternEmitter::emit(StringRef rewriteName) {
  // Get the DAG tree for the source pattern.
  DagNode sourceTree = pattern.getSourcePattern();

  const Operator &rootOp = pattern.getSourceRootOp();
  auto rootName = rootOp.getOperationName();

  // Collect the set of result operations.
  llvm::SmallPtrSet<const Operator *, 4> resultOps;
  LLVM_DEBUG(llvm::dbgs() << "start collecting ops used in result patterns\n");
  for (unsigned i = 0, e = pattern.getNumResultPatterns(); i != e; ++i) {
    collectOps(pattern.getResultPattern(i), resultOps);
  }
  LLVM_DEBUG(llvm::dbgs() << "done collecting ops used in result patterns\n");

  // Emit RewritePattern for Pattern.
  auto locs = pattern.getLocation();
  os << formatv("/* Generated from:\n    {0:$[ instantiating\n    ]}\n*/\n",
                make_range(locs.rbegin(), locs.rend()));
  os << formatv(R"(struct {0} : public ::mlir::RewritePattern {
  {0}(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("{1}", {2}, context, {{)",
                rewriteName, rootName, pattern.getBenefit());
  // Sort result operators by name.
  llvm::SmallVector<const Operator *, 4> sortedResultOps(resultOps.begin(),
                                                         resultOps.end());
  llvm::sort(sortedResultOps, [&](const Operator *lhs, const Operator *rhs) {
    return lhs->getOperationName() < rhs->getOperationName();
  });
  llvm::interleaveComma(sortedResultOps, os, [&](const Operator *op) {
    os << '"' << op->getOperationName() << '"';
  });
  os << "}) {}\n";

  // Emit matchAndRewrite() function.
  {
    auto classScope = os.scope();
    os.printReindented(R"(
    ::mlir::LogicalResult matchAndRewrite(::mlir::Operation *op0,
        ::mlir::PatternRewriter &rewriter) const override {)")
        << '\n';
    {
      auto functionScope = os.scope();

      // Register all symbols bound in the source pattern.
      pattern.collectSourcePatternBoundSymbols(symbolInfoMap);

      LLVM_DEBUG(llvm::dbgs()
                 << "start creating local variables for capturing matches\n");
      os << "// Variables for capturing values and attributes used while "
            "creating ops\n";
      // Create local variables for storing the arguments and results bound
      // to symbols.
      for (const auto &symbolInfoPair : symbolInfoMap) {
        const auto &symbol = symbolInfoPair.first;
        const auto &info = symbolInfoPair.second;

        os << info.getVarDecl(symbol);
      }
      // TODO: capture ops with consistent numbering so that it can be
      // reused for fused loc.
      os << "::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;\n\n";
      LLVM_DEBUG(llvm::dbgs()
                 << "done creating local variables for capturing matches\n");

      os << "// Match\n";
      os << "tblgen_ops.push_back(op0);\n";
      emitMatchLogic(sourceTree, "op0");

      os << "\n// Rewrite\n";
      emitRewriteLogic();

      os << "return ::mlir::success();\n";
    }
    os << "};\n";
  }
  os << "};\n\n";
}

void PatternEmitter::emitRewriteLogic() {
  LLVM_DEBUG(llvm::dbgs() << "--- start emitting rewrite logic ---\n");
  const Operator &rootOp = pattern.getSourceRootOp();
  int numExpectedResults = rootOp.getNumResults();
  int numResultPatterns = pattern.getNumResultPatterns();

  // First register all symbols bound to ops generated in result patterns.
  pattern.collectResultPatternBoundSymbols(symbolInfoMap);

  // Only the last N static values generated are used to replace the matched
  // root N-result op. We need to calculate the starting index (of the results
  // of the matched op) each result pattern is to replace.
  SmallVector<int, 4> offsets(numResultPatterns + 1, numExpectedResults);
  // If we don't need to replace any value at all, set the replacement starting
  // index as the number of result patterns so we skip all of them when trying
  // to replace the matched op's results.
  int replStartIndex = numExpectedResults == 0 ? numResultPatterns : -1;
  for (int i = numResultPatterns - 1; i >= 0; --i) {
    auto numValues = getNodeValueCount(pattern.getResultPattern(i));
    offsets[i] = offsets[i + 1] - numValues;
    if (offsets[i] == 0) {
      if (replStartIndex == -1)
        replStartIndex = i;
    } else if (offsets[i] < 0 && offsets[i + 1] > 0) {
      auto error = formatv(
          "cannot use the same multi-result op '{0}' to generate both "
          "auxiliary values and values to be used for replacing the matched op",
          pattern.getResultPattern(i).getSymbol());
      PrintFatalError(loc, error);
    }
  }

  if (offsets.front() > 0) {
    const char error[] =
        "not enough values generated to replace the matched op";
    PrintFatalError(loc, error);
  }

  os << "auto odsLoc = rewriter.getFusedLoc({";
  for (int i = 0, e = pattern.getSourcePattern().getNumOps(); i != e; ++i) {
    os << (i ? ", " : "") << "tblgen_ops[" << i << "]->getLoc()";
  }
  os << "}); (void)odsLoc;\n";

  // Process auxiliary result patterns.
  for (int i = 0; i < replStartIndex; ++i) {
    DagNode resultTree = pattern.getResultPattern(i);
    auto val = handleResultPattern(resultTree, offsets[i], 0);
    // Normal op creation will be streamed to `os` by the above call; but
    // NativeCodeCall will only be materialized to `os` if it is used. Here
    // we are handling auxiliary patterns so we want the side effect even if
    // NativeCodeCall is not replacing matched root op's results.
    if (resultTree.isNativeCodeCall() &&
        resultTree.getNumReturnsOfNativeCode() == 0)
      os << val << ";\n";
  }

  auto processSupplementalPatterns = [&]() {
    int numSupplementalPatterns = pattern.getNumSupplementalPatterns();
    for (int i = 0, offset = -numSupplementalPatterns;
         i < numSupplementalPatterns; ++i) {
      DagNode resultTree = pattern.getSupplementalPattern(i);
      auto val = handleResultPattern(resultTree, offset++, 0);
      if (resultTree.isNativeCodeCall() &&
          resultTree.getNumReturnsOfNativeCode() == 0)
        os << val << ";\n";
    }
  };

  if (numExpectedResults == 0) {
    assert(replStartIndex >= numResultPatterns &&
           "invalid auxiliary vs. replacement pattern division!");
    processSupplementalPatterns();
    // No result to replace. Just erase the op.
    os << "rewriter.eraseOp(op0);\n";
  } else {
    // Process replacement result patterns.
    os << "::llvm::SmallVector<::mlir::Value, 4> tblgen_repl_values;\n";
    for (int i = replStartIndex; i < numResultPatterns; ++i) {
      DagNode resultTree = pattern.getResultPattern(i);
      auto val = handleResultPattern(resultTree, offsets[i], 0);
      os << "\n";
      // Resolve each symbol for all range use so that we can loop over them.
      // We need an explicit cast to `SmallVector` to capture the cases where
      // `{0}` resolves to an `Operation::result_range` as well as cases that
      // are not iterable (e.g. vector that gets wrapped in additional braces by
      // RewriterGen).
      // TODO: Revisit the need for materializing a vector.
      os << symbolInfoMap.getAllRangeUse(
          val,
          "for (auto v: ::llvm::SmallVector<::mlir::Value, 4>{ {0} }) {{\n"
          "  tblgen_repl_values.push_back(v);\n}\n",
          "\n");
    }
    processSupplementalPatterns();
    os << "\nrewriter.replaceOp(op0, tblgen_repl_values);\n";
  }

  LLVM_DEBUG(llvm::dbgs() << "--- done emitting rewrite logic ---\n");
}

std::string PatternEmitter::getUniqueSymbol(const Operator *op) {
  return std::string(
      formatv("tblgen_{0}_{1}", op->getCppClassName(), nextValueId++));
}

std::string PatternEmitter::handleResultPattern(DagNode resultTree,
                                                int resultIndex, int depth) {
  LLVM_DEBUG(llvm::dbgs() << "handle result pattern: ");
  LLVM_DEBUG(resultTree.print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');

  if (resultTree.isLocationDirective()) {
    PrintFatalError(loc,
                    "location directive can only be used with op creation");
  }

  if (resultTree.isNativeCodeCall())
    return handleReplaceWithNativeCodeCall(resultTree, depth);

  if (resultTree.isReplaceWithValue())
    return handleReplaceWithValue(resultTree).str();

  if (resultTree.isVariadic())
    return handleVariadic(resultTree, depth);

  // Normal op creation.
  auto symbol = handleOpCreation(resultTree, resultIndex, depth);
  if (resultTree.getSymbol().empty()) {
    // This is an op not explicitly bound to a symbol in the rewrite rule.
    // Register the auto-generated symbol for it.
    symbolInfoMap.bindOpResult(symbol, pattern.getDialectOp(resultTree));
  }
  return symbol;
}

std::string PatternEmitter::handleVariadic(DagNode tree, int depth) {
  assert(tree.isVariadic());

  auto name = std::string(formatv("tblgen_variadic_values_{0}", nextValueId++));
  symbolInfoMap.bindValue(name);
  os << "::llvm::SmallVector<::mlir::Value, 4> " << name << ";\n";
  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    if (auto child = tree.getArgAsNestedDag(i)) {
      os << name << ".push_back(" << handleResultPattern(child, i, depth + 1)
         << ");\n";
    } else {
      os << name << ".push_back("
         << handleOpArgument(tree.getArgAsLeaf(i), tree.getArgName(i))
         << ");\n";
    }
  }

  return name;
}

StringRef PatternEmitter::handleReplaceWithValue(DagNode tree) {
  assert(tree.isReplaceWithValue());

  if (tree.getNumArgs() != 1) {
    PrintFatalError(
        loc, "replaceWithValue directive must take exactly one argument");
  }

  if (!tree.getSymbol().empty()) {
    PrintFatalError(loc, "cannot bind symbol to replaceWithValue");
  }

  return tree.getArgName(0);
}

std::string PatternEmitter::handleLocationDirective(DagNode tree) {
  assert(tree.isLocationDirective());
  auto lookUpArgLoc = [this, &tree](int idx) {
    const auto *const lookupFmt = "{0}.getLoc()";
    return symbolInfoMap.getValueAndRangeUse(tree.getArgName(idx), lookupFmt);
  };

  if (tree.getNumArgs() == 0)
    llvm::PrintFatalError(
        "At least one argument to location directive required");

  if (!tree.getSymbol().empty())
    PrintFatalError(loc, "cannot bind symbol to location");

  if (tree.getNumArgs() == 1) {
    DagLeaf leaf = tree.getArgAsLeaf(0);
    if (leaf.isStringAttr())
      return formatv("::mlir::NameLoc::get(rewriter.getStringAttr(\"{0}\"))",
                     leaf.getStringAttr())
          .str();
    return lookUpArgLoc(0);
  }

  std::string ret;
  llvm::raw_string_ostream os(ret);
  std::string strAttr;
  os << "rewriter.getFusedLoc({";
  bool first = true;
  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    DagLeaf leaf = tree.getArgAsLeaf(i);
    // Handle the optional string value.
    if (leaf.isStringAttr()) {
      if (!strAttr.empty())
        llvm::PrintFatalError("Only one string attribute may be specified");
      strAttr = leaf.getStringAttr();
      continue;
    }
    os << (first ? "" : ", ") << lookUpArgLoc(i);
    first = false;
  }
  os << "}";
  if (!strAttr.empty()) {
    os << ", rewriter.getStringAttr(\"" << strAttr << "\")";
  }
  os << ")";
  return os.str();
}

std::string PatternEmitter::handleReturnTypeArg(DagNode returnType, int i,
                                                int depth) {
  // Nested NativeCodeCall.
  if (auto dagNode = returnType.getArgAsNestedDag(i)) {
    if (!dagNode.isNativeCodeCall())
      PrintFatalError(loc, "nested DAG in `returnType` must be a native code "
                           "call");
    return handleReplaceWithNativeCodeCall(dagNode, depth);
  }
  // String literal.
  auto dagLeaf = returnType.getArgAsLeaf(i);
  if (dagLeaf.isStringAttr())
    return tgfmt(dagLeaf.getStringAttr(), &fmtCtx);
  return tgfmt(
      "$0.getType()", &fmtCtx,
      handleOpArgument(returnType.getArgAsLeaf(i), returnType.getArgName(i)));
}

std::string PatternEmitter::handleOpArgument(DagLeaf leaf,
                                             StringRef patArgName) {
  if (leaf.isStringAttr())
    PrintFatalError(loc, "raw string not supported as argument");
  if (leaf.isConstantAttr()) {
    auto constAttr = leaf.getAsConstantAttr();
    return handleConstantAttr(constAttr.getAttribute(),
                              constAttr.getConstantValue());
  }
  if (leaf.isEnumAttrCase()) {
    auto enumCase = leaf.getAsEnumAttrCase();
    // This is an enum case backed by an IntegerAttr. We need to get its value
    // to build the constant.
    std::string val = std::to_string(enumCase.getValue());
    return handleConstantAttr(enumCase, val);
  }

  LLVM_DEBUG(llvm::dbgs() << "handle argument '" << patArgName << "'\n");
  auto argName = symbolInfoMap.getValueAndRangeUse(patArgName);
  if (leaf.isUnspecified() || leaf.isOperandMatcher()) {
    LLVM_DEBUG(llvm::dbgs() << "replace " << patArgName << " with '" << argName
                            << "' (via symbol ref)\n");
    return argName;
  }
  if (leaf.isNativeCodeCall()) {
    auto repl = tgfmt(leaf.getNativeCodeTemplate(), &fmtCtx.withSelf(argName));
    LLVM_DEBUG(llvm::dbgs() << "replace " << patArgName << " with '" << repl
                            << "' (via NativeCodeCall)\n");
    return std::string(repl);
  }
  PrintFatalError(loc, "unhandled case when rewriting op");
}

std::string PatternEmitter::handleReplaceWithNativeCodeCall(DagNode tree,
                                                            int depth) {
  LLVM_DEBUG(llvm::dbgs() << "handle NativeCodeCall pattern: ");
  LLVM_DEBUG(tree.print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');

  auto fmt = tree.getNativeCodeTemplate();

  SmallVector<std::string, 16> attrs;

  auto tail = getTrailingDirectives(tree);
  if (tail.returnType)
    PrintFatalError(loc, "`NativeCodeCall` cannot have return type specifier");
  auto locToUse = getLocation(tail);

  for (int i = 0, e = tree.getNumArgs() - tail.numDirectives; i != e; ++i) {
    if (tree.isNestedDagArg(i)) {
      attrs.push_back(
          handleResultPattern(tree.getArgAsNestedDag(i), i, depth + 1));
    } else {
      attrs.push_back(
          handleOpArgument(tree.getArgAsLeaf(i), tree.getArgName(i)));
    }
    LLVM_DEBUG(llvm::dbgs() << "NativeCodeCall argument #" << i
                            << " replacement: " << attrs[i] << "\n");
  }

  std::string symbol = tgfmt(fmt, &fmtCtx.addSubst("_loc", locToUse),
                             static_cast<ArrayRef<std::string>>(attrs));

  // In general, NativeCodeCall without naming binding don't need this. To
  // ensure void helper function has been correctly labeled, i.e., use
  // NativeCodeCallVoid, we cache the result to a local variable so that we will
  // get a compilation error in the auto-generated file.
  // Example.
  //   // In the td file
  //   Pat<(...), (NativeCodeCall<Foo> ...)>
  //
  //   ---
  //
  //   // In the auto-generated .cpp
  //   ...
  //   // Causes compilation error if Foo() returns void.
  //   auto nativeVar = Foo();
  //   ...
  if (tree.getNumReturnsOfNativeCode() != 0) {
    // Determine the local variable name for return value.
    std::string varName =
        SymbolInfoMap::getValuePackName(tree.getSymbol()).str();
    if (varName.empty()) {
      varName = formatv("nativeVar_{0}", nextValueId++);
      // Register the local variable for later uses.
      symbolInfoMap.bindValues(varName, tree.getNumReturnsOfNativeCode());
    }

    // Catch the return value of helper function.
    os << formatv("auto {0} = {1}; (void){0};\n", varName, symbol);

    if (!tree.getSymbol().empty())
      symbol = tree.getSymbol().str();
    else
      symbol = varName;
  }

  return symbol;
}

int PatternEmitter::getNodeValueCount(DagNode node) {
  if (node.isOperation()) {
    // If the op is bound to a symbol in the rewrite rule, query its result
    // count from the symbol info map.
    auto symbol = node.getSymbol();
    if (!symbol.empty()) {
      return symbolInfoMap.getStaticValueCount(symbol);
    }
    // Otherwise this is an unbound op; we will use all its results.
    return pattern.getDialectOp(node).getNumResults();
  }

  if (node.isNativeCodeCall())
    return node.getNumReturnsOfNativeCode();

  return 1;
}

PatternEmitter::TrailingDirectives
PatternEmitter::getTrailingDirectives(DagNode tree) {
  TrailingDirectives tail = {DagNode(nullptr), DagNode(nullptr), 0};

  // Look backwards through the arguments.
  auto numPatArgs = tree.getNumArgs();
  for (int i = numPatArgs - 1; i >= 0; --i) {
    auto dagArg = tree.getArgAsNestedDag(i);
    // A leaf is not a directive. Stop looking.
    if (!dagArg)
      break;

    auto isLocation = dagArg.isLocationDirective();
    auto isReturnType = dagArg.isReturnTypeDirective();
    // If encountered a DAG node that isn't a trailing directive, stop looking.
    if (!(isLocation || isReturnType))
      break;
    // Save the directive, but error if one of the same type was already
    // found.
    ++tail.numDirectives;
    if (isLocation) {
      if (tail.location)
        PrintFatalError(loc, "`location` directive can only be specified "
                             "once");
      tail.location = dagArg;
    } else if (isReturnType) {
      if (tail.returnType)
        PrintFatalError(loc, "`returnType` directive can only be specified "
                             "once");
      tail.returnType = dagArg;
    }
  }

  return tail;
}

std::string
PatternEmitter::getLocation(PatternEmitter::TrailingDirectives &tail) {
  if (tail.location)
    return handleLocationDirective(tail.location);

  // If no explicit location is given, use the default, all fused, location.
  return "odsLoc";
}

std::string PatternEmitter::handleOpCreation(DagNode tree, int resultIndex,
                                             int depth) {
  LLVM_DEBUG(llvm::dbgs() << "create op for pattern: ");
  LLVM_DEBUG(tree.print(llvm::dbgs()));
  LLVM_DEBUG(llvm::dbgs() << '\n');

  Operator &resultOp = tree.getDialectOp(opMap);
  auto numOpArgs = resultOp.getNumArgs();
  auto numPatArgs = tree.getNumArgs();

  auto tail = getTrailingDirectives(tree);
  auto locToUse = getLocation(tail);

  auto inPattern = numPatArgs - tail.numDirectives;
  if (numOpArgs != inPattern) {
    PrintFatalError(loc,
                    formatv("resultant op '{0}' argument number mismatch: "
                            "{1} in pattern vs. {2} in definition",
                            resultOp.getOperationName(), inPattern, numOpArgs));
  }

  // A map to collect all nested DAG child nodes' names, with operand index as
  // the key. This includes both bound and unbound child nodes.
  ChildNodeIndexNameMap childNodeNames;

  // If the argument is a type constraint, then its an operand. Check if the
  // op's argument is variadic that the argument in the pattern is too.
  auto checkIfMatchedVariadic = [&](int i) {
    // FIXME: This does not yet check for variable/leaf case.
    // FIXME: Change so that native code call can be handled.
    const auto *operand =
        llvm::dyn_cast_if_present<NamedTypeConstraint *>(resultOp.getArg(i));
    if (!operand || !operand->isVariadic())
      return;

    auto child = tree.getArgAsNestedDag(i);
    if (!child)
      return;

    // Skip over replaceWithValues.
    while (child.isReplaceWithValue()) {
      if (!(child = child.getArgAsNestedDag(0)))
        return;
    }
    if (!child.isNativeCodeCall() && !child.isVariadic())
      PrintFatalError(loc, formatv("op expects variadic operand `{0}`, while "
                                   "provided is non-variadic",
                                   resultOp.getArgName(i)));
  };

  // First go through all the child nodes who are nested DAG constructs to
  // create ops for them and remember the symbol names for them, so that we can
  // use the results in the current node. This happens in a recursive manner.
  for (int i = 0, e = tree.getNumArgs() - tail.numDirectives; i != e; ++i) {
    checkIfMatchedVariadic(i);
    if (auto child = tree.getArgAsNestedDag(i))
      childNodeNames[i] = handleResultPattern(child, i, depth + 1);
  }

  // The name of the local variable holding this op.
  std::string valuePackName;
  // The symbol for holding the result of this pattern. Note that the result of
  // this pattern is not necessarily the same as the variable created by this
  // pattern because we can use `__N` suffix to refer only a specific result if
  // the generated op is a multi-result op.
  std::string resultValue;
  if (tree.getSymbol().empty()) {
    // No symbol is explicitly bound to this op in the pattern. Generate a
    // unique name.
    valuePackName = resultValue = getUniqueSymbol(&resultOp);
  } else {
    resultValue = std::string(tree.getSymbol());
    // Strip the index to get the name for the value pack and use it to name the
    // local variable for the op.
    valuePackName = std::string(SymbolInfoMap::getValuePackName(resultValue));
  }

  // Create the local variable for this op.
  os << formatv("{0} {1};\n{{\n", resultOp.getQualCppClassName(),
                valuePackName);

  // Right now ODS don't have general type inference support. Except a few
  // special cases listed below, DRR needs to supply types for all results
  // when building an op.
  bool isSameOperandsAndResultType =
      resultOp.getTrait("::mlir::OpTrait::SameOperandsAndResultType");
  bool useFirstAttr =
      resultOp.getTrait("::mlir::OpTrait::FirstAttrDerivedResultType");

  if (!tail.returnType && (isSameOperandsAndResultType || useFirstAttr)) {
    // We know how to deduce the result type for ops with these traits and we've
    // generated builders taking aggregate parameters. Use those builders to
    // create the ops.

    // First prepare local variables for op arguments used in builder call.
    createAggregateLocalVarsForOpArgs(tree, childNodeNames, depth);

    // Then create the op.
    os.scope("", "\n}\n").os << formatv(
        "{0} = rewriter.create<{1}>({2}, tblgen_values, tblgen_attrs);",
        valuePackName, resultOp.getQualCppClassName(), locToUse);
    return resultValue;
  }

  bool usePartialResults = valuePackName != resultValue;

  if (!tail.returnType && (usePartialResults || depth > 0 || resultIndex < 0)) {
    // For these cases (broadcastable ops, op results used both as auxiliary
    // values and replacement values, ops in nested patterns, auxiliary ops), we
    // still need to supply the result types when building the op. But because
    // we don't generate a builder automatically with ODS for them, it's the
    // developer's responsibility to make sure such a builder (with result type
    // deduction ability) exists. We go through the separate-parameter builder
    // here given that it's easier for developers to write compared to
    // aggregate-parameter builders.
    createSeparateLocalVarsForOpArgs(tree, childNodeNames);

    os.scope().os << formatv("{0} = rewriter.create<{1}>({2}", valuePackName,
                             resultOp.getQualCppClassName(), locToUse);
    supplyValuesForOpArgs(tree, childNodeNames, depth);
    os << "\n  );\n}\n";
    return resultValue;
  }

  // If we are provided explicit return types, use them to build the op.
  // However, if depth == 0 and resultIndex >= 0, it means we are replacing
  // the values generated from the source pattern root op. Then we must use the
  // source pattern's value types to determine the value type of the generated
  // op here.
  if (depth == 0 && resultIndex >= 0 && tail.returnType)
    PrintFatalError(loc, "Cannot specify explicit return types in an op whose "
                         "return values replace the source pattern's root op");

  // First prepare local variables for op arguments used in builder call.
  createAggregateLocalVarsForOpArgs(tree, childNodeNames, depth);

  // Then prepare the result types. We need to specify the types for all
  // results.
  os.indent() << formatv("::llvm::SmallVector<::mlir::Type, 4> tblgen_types; "
                         "(void)tblgen_types;\n");
  int numResults = resultOp.getNumResults();
  if (tail.returnType) {
    auto numRetTys = tail.returnType.getNumArgs();
    for (int i = 0; i < numRetTys; ++i) {
      auto varName = handleReturnTypeArg(tail.returnType, i, depth + 1);
      os << "tblgen_types.push_back(" << varName << ");\n";
    }
  } else {
    if (numResults != 0) {
      // Copy the result types from the source pattern.
      for (int i = 0; i < numResults; ++i)
        os << formatv("for (auto v: castedOp0.getODSResults({0})) {{\n"
                      "  tblgen_types.push_back(v.getType());\n}\n",
                      resultIndex + i);
    }
  }
  os << formatv("{0} = rewriter.create<{1}>({2}, tblgen_types, "
                "tblgen_values, tblgen_attrs);\n",
                valuePackName, resultOp.getQualCppClassName(), locToUse);
  os.unindent() << "}\n";
  return resultValue;
}

void PatternEmitter::createSeparateLocalVarsForOpArgs(
    DagNode node, ChildNodeIndexNameMap &childNodeNames) {
  Operator &resultOp = node.getDialectOp(opMap);

  // Now prepare operands used for building this op:
  // * If the operand is non-variadic, we create a `Value` local variable.
  // * If the operand is variadic, we create a `SmallVector<Value>` local
  //   variable.

  int valueIndex = 0; // An index for uniquing local variable names.
  for (int argIndex = 0, e = resultOp.getNumArgs(); argIndex < e; ++argIndex) {
    const auto *operand =
        llvm::dyn_cast_if_present<NamedTypeConstraint *>(resultOp.getArg(argIndex));
    // We do not need special handling for attributes.
    if (!operand)
      continue;

    raw_indented_ostream::DelimitedScope scope(os);
    std::string varName;
    if (operand->isVariadic()) {
      varName = std::string(formatv("tblgen_values_{0}", valueIndex++));
      os << formatv("::llvm::SmallVector<::mlir::Value, 4> {0};\n", varName);
      std::string range;
      if (node.isNestedDagArg(argIndex)) {
        range = childNodeNames[argIndex];
      } else {
        range = std::string(node.getArgName(argIndex));
      }
      // Resolve the symbol for all range use so that we have a uniform way of
      // capturing the values.
      range = symbolInfoMap.getValueAndRangeUse(range);
      os << formatv("for (auto v: {0}) {{\n  {1}.push_back(v);\n}\n", range,
                    varName);
    } else {
      varName = std::string(formatv("tblgen_value_{0}", valueIndex++));
      os << formatv("::mlir::Value {0} = ", varName);
      if (node.isNestedDagArg(argIndex)) {
        os << symbolInfoMap.getValueAndRangeUse(childNodeNames[argIndex]);
      } else {
        DagLeaf leaf = node.getArgAsLeaf(argIndex);
        auto symbol =
            symbolInfoMap.getValueAndRangeUse(node.getArgName(argIndex));
        if (leaf.isNativeCodeCall()) {
          os << std::string(
              tgfmt(leaf.getNativeCodeTemplate(), &fmtCtx.withSelf(symbol)));
        } else {
          os << symbol;
        }
      }
      os << ";\n";
    }

    // Update to use the newly created local variable for building the op later.
    childNodeNames[argIndex] = varName;
  }
}

void PatternEmitter::supplyValuesForOpArgs(
    DagNode node, const ChildNodeIndexNameMap &childNodeNames, int depth) {
  Operator &resultOp = node.getDialectOp(opMap);
  for (int argIndex = 0, numOpArgs = resultOp.getNumArgs();
       argIndex != numOpArgs; ++argIndex) {
    // Start each argument on its own line.
    os << ",\n    ";

    Argument opArg = resultOp.getArg(argIndex);
    // Handle the case of operand first.
    if (auto *operand = llvm::dyn_cast_if_present<NamedTypeConstraint *>(opArg)) {
      if (!operand->name.empty())
        os << "/*" << operand->name << "=*/";
      os << childNodeNames.lookup(argIndex);
      continue;
    }

    // The argument in the op definition.
    auto opArgName = resultOp.getArgName(argIndex);
    if (auto subTree = node.getArgAsNestedDag(argIndex)) {
      if (!subTree.isNativeCodeCall())
        PrintFatalError(loc, "only NativeCodeCall allowed in nested dag node "
                             "for creating attribute");
      os << formatv("/*{0}=*/{1}", opArgName, childNodeNames.lookup(argIndex));
    } else {
      auto leaf = node.getArgAsLeaf(argIndex);
      // The argument in the result DAG pattern.
      auto patArgName = node.getArgName(argIndex);
      if (leaf.isConstantAttr() || leaf.isEnumAttrCase()) {
        // TODO: Refactor out into map to avoid recomputing these.
        if (!opArg.is<NamedAttribute *>())
          PrintFatalError(loc, Twine("expected attribute ") + Twine(argIndex));
        if (!patArgName.empty())
          os << "/*" << patArgName << "=*/";
      } else {
        os << "/*" << opArgName << "=*/";
      }
      os << handleOpArgument(leaf, patArgName);
    }
  }
}

void PatternEmitter::createAggregateLocalVarsForOpArgs(
    DagNode node, const ChildNodeIndexNameMap &childNodeNames, int depth) {
  Operator &resultOp = node.getDialectOp(opMap);

  auto scope = os.scope();
  os << formatv("::llvm::SmallVector<::mlir::Value, 4> "
                "tblgen_values; (void)tblgen_values;\n");
  os << formatv("::llvm::SmallVector<::mlir::NamedAttribute, 4> "
                "tblgen_attrs; (void)tblgen_attrs;\n");

  const char *addAttrCmd =
      "if (auto tmpAttr = {1}) {\n"
      "  tblgen_attrs.emplace_back(rewriter.getStringAttr(\"{0}\"), "
      "tmpAttr);\n}\n";
  int numVariadic = 0;
  bool hasOperandSegmentSizes = false;
  std::vector<std::string> sizes;
  for (int argIndex = 0, e = resultOp.getNumArgs(); argIndex < e; ++argIndex) {
    if (resultOp.getArg(argIndex).is<NamedAttribute *>()) {
      // The argument in the op definition.
      auto opArgName = resultOp.getArgName(argIndex);
      hasOperandSegmentSizes =
          hasOperandSegmentSizes || opArgName == "operandSegmentSizes";
      if (auto subTree = node.getArgAsNestedDag(argIndex)) {
        if (!subTree.isNativeCodeCall())
          PrintFatalError(loc, "only NativeCodeCall allowed in nested dag node "
                               "for creating attribute");
        os << formatv(addAttrCmd, opArgName, childNodeNames.lookup(argIndex));
      } else {
        auto leaf = node.getArgAsLeaf(argIndex);
        // The argument in the result DAG pattern.
        auto patArgName = node.getArgName(argIndex);
        os << formatv(addAttrCmd, opArgName,
                      handleOpArgument(leaf, patArgName));
      }
      continue;
    }

    const auto *operand =
        resultOp.getArg(argIndex).get<NamedTypeConstraint *>();
    std::string varName;
    if (operand->isVariadic()) {
      ++numVariadic;
      std::string range;
      if (node.isNestedDagArg(argIndex)) {
        range = childNodeNames.lookup(argIndex);
      } else {
        range = std::string(node.getArgName(argIndex));
      }
      // Resolve the symbol for all range use so that we have a uniform way of
      // capturing the values.
      range = symbolInfoMap.getValueAndRangeUse(range);
      os << formatv("for (auto v: {0}) {{\n  tblgen_values.push_back(v);\n}\n",
                    range);
      sizes.push_back(formatv("static_cast<int32_t>({0}.size())", range));
    } else {
      sizes.emplace_back("1");
      os << formatv("tblgen_values.push_back(");
      if (node.isNestedDagArg(argIndex)) {
        os << symbolInfoMap.getValueAndRangeUse(
            childNodeNames.lookup(argIndex));
      } else {
        DagLeaf leaf = node.getArgAsLeaf(argIndex);
        if (leaf.isConstantAttr())
          // TODO: Use better location
          PrintFatalError(
              loc,
              "attribute found where value was expected, if attempting to use "
              "constant value, construct a constant op with given attribute "
              "instead");

        auto symbol =
            symbolInfoMap.getValueAndRangeUse(node.getArgName(argIndex));
        if (leaf.isNativeCodeCall()) {
          os << std::string(
              tgfmt(leaf.getNativeCodeTemplate(), &fmtCtx.withSelf(symbol)));
        } else {
          os << symbol;
        }
      }
      os << ");\n";
    }
  }

  if (numVariadic > 1 && !hasOperandSegmentSizes) {
    // Only set size if it can't be computed.
    const auto *sameVariadicSize =
        resultOp.getTrait("::mlir::OpTrait::SameVariadicOperandSize");
    if (!sameVariadicSize) {
      const char *setSizes = R"(
        tblgen_attrs.emplace_back(rewriter.getStringAttr("operandSegmentSizes"),
          rewriter.getDenseI32ArrayAttr({{ {0} }));
          )";
      os.printReindented(formatv(setSizes, llvm::join(sizes, ", ")).str());
    }
  }
}

StaticMatcherHelper::StaticMatcherHelper(raw_ostream &os,
                                         const RecordKeeper &recordKeeper,
                                         RecordOperatorMap &mapper)
    : opMap(mapper), staticVerifierEmitter(os, recordKeeper) {}

void StaticMatcherHelper::populateStaticMatchers(raw_ostream &os) {
  // PatternEmitter will use the static matcher if there's one generated. To
  // ensure that all the dependent static matchers are generated before emitting
  // the matching logic of the DagNode, we use topological order to achieve it.
  for (auto &dagInfo : topologicalOrder) {
    DagNode node = dagInfo.first;
    if (!useStaticMatcher(node))
      continue;

    std::string funcName =
        formatv("static_dag_matcher_{0}", staticMatcherCounter++);
    assert(!matcherNames.contains(node));
    PatternEmitter(dagInfo.second, &opMap, os, *this)
        .emitStaticMatcher(node, funcName);
    matcherNames[node] = funcName;
  }
}

void StaticMatcherHelper::populateStaticConstraintFunctions(raw_ostream &os) {
  staticVerifierEmitter.emitPatternConstraints(constraints.getArrayRef());
}

void StaticMatcherHelper::addPattern(Record *record) {
  Pattern pat(record, &opMap);

  // While generating the function body of the DAG matcher, it may depends on
  // other DAG matchers. To ensure the dependent matchers are ready, we compute
  // the topological order for all the DAGs and emit the DAG matchers in this
  // order.
  llvm::unique_function<void(DagNode)> dfs = [&](DagNode node) {
    ++refStats[node];

    if (refStats[node] != 1)
      return;

    for (unsigned i = 0, e = node.getNumArgs(); i < e; ++i)
      if (DagNode sibling = node.getArgAsNestedDag(i))
        dfs(sibling);
      else {
        DagLeaf leaf = node.getArgAsLeaf(i);
        if (!leaf.isUnspecified())
          constraints.insert(leaf);
      }

    topologicalOrder.push_back(std::make_pair(node, record));
  };

  dfs(pat.getSourcePattern());
}

StringRef StaticMatcherHelper::getVerifierName(DagLeaf leaf) {
  if (leaf.isAttrMatcher()) {
    std::optional<StringRef> constraint =
        staticVerifierEmitter.getAttrConstraintFn(leaf.getAsConstraint());
    assert(constraint && "attribute constraint was not uniqued");
    return *constraint;
  }
  assert(leaf.isOperandMatcher());
  return staticVerifierEmitter.getTypeConstraintFn(leaf.getAsConstraint());
}

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os, recordKeeper);

  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");

  // We put the map here because it can be shared among multiple patterns.
  RecordOperatorMap recordOpMap;

  // Exam all the patterns and generate static matcher for the duplicated
  // DagNode.
  StaticMatcherHelper staticMatcher(os, recordKeeper, recordOpMap);
  for (Record *p : patterns)
    staticMatcher.addPattern(p);
  staticMatcher.populateStaticConstraintFunctions(os);
  staticMatcher.populateStaticMatchers(os);

  std::vector<std::string> rewriterNames;
  rewriterNames.reserve(patterns.size());

  std::string baseRewriterName = "GeneratedConvert";
  int rewriterIndex = 0;

  for (Record *p : patterns) {
    std::string name;
    if (p->isAnonymous()) {
      // If no name is provided, ensure unique rewriter names simply by
      // appending unique suffix.
      name = baseRewriterName + llvm::utostr(rewriterIndex++);
    } else {
      name = std::string(p->getName());
    }
    LLVM_DEBUG(llvm::dbgs()
               << "=== start generating pattern '" << name << "' ===\n");
    PatternEmitter(p, &recordOpMap, os, staticMatcher).emit(name);
    LLVM_DEBUG(llvm::dbgs()
               << "=== done generating pattern '" << name << "' ===\n");
    rewriterNames.push_back(std::move(name));
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void LLVM_ATTRIBUTE_UNUSED populateWithGenerated("
        "::mlir::RewritePatternSet &patterns) {\n";
  for (const auto &name : rewriterNames) {
    os << "  patterns.add<" << name << ">(patterns.getContext());\n";
  }
  os << "}\n";
}

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });
