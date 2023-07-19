//===- ModuleImport.cpp - LLVM to MLIR conversion ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the import of an LLVM IR module into an LLVM dialect
// module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "AttrKindDetail.h"
#include "DataLayoutImporter.h"
#include "DebugImporter.h"
#include "LoopAnnotationImporter.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ModRef.h"

using namespace mlir;
using namespace mlir::LLVM;
using namespace mlir::LLVM::detail;

#include "mlir/Dialect/LLVMIR/LLVMConversionEnumsFromLLVM.inc"

// Utility to print an LLVM value as a string for passing to emitError().
// FIXME: Diagnostic should be able to natively handle types that have
// operator << (raw_ostream&) defined.
static std::string diag(const llvm::Value &value) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << value;
  return os.str();
}

// Utility to print an LLVM metadata node as a string for passing
// to emitError(). The module argument is needed to print the nodes
// canonically numbered.
static std::string diagMD(const llvm::Metadata *node,
                          const llvm::Module *module) {
  std::string str;
  llvm::raw_string_ostream os(str);
  node->print(os, module, /*IsForDebug=*/true);
  return os.str();
}

/// Returns the name of the global_ctors global variables.
static constexpr StringRef getGlobalCtorsVarName() {
  return "llvm.global_ctors";
}

/// Returns the name of the global_dtors global variables.
static constexpr StringRef getGlobalDtorsVarName() {
  return "llvm.global_dtors";
}

/// Returns the symbol name for the module-level comdat operation. It must not
/// conflict with the user namespace.
static constexpr StringRef getGlobalComdatOpName() {
  return "__llvm_global_comdat";
}

/// Converts the sync scope identifier of `inst` to the string representation
/// necessary to build an atomic LLVM dialect operation. Returns the empty
/// string if the operation has either no sync scope or the default system-level
/// sync scope attached. The atomic operations only set their sync scope
/// attribute if they have a non-default sync scope attached.
static StringRef getLLVMSyncScope(llvm::Instruction *inst) {
  std::optional<llvm::SyncScope::ID> syncScopeID =
      llvm::getAtomicSyncScopeID(inst);
  if (!syncScopeID)
    return "";

  // Search the sync scope name for the given identifier. The default
  // system-level sync scope thereby maps to the empty string.
  SmallVector<StringRef> syncScopeName;
  llvm::LLVMContext &llvmContext = inst->getContext();
  llvmContext.getSyncScopeNames(syncScopeName);
  auto *it = llvm::find_if(syncScopeName, [&](StringRef name) {
    return *syncScopeID == llvmContext.getOrInsertSyncScopeID(name);
  });
  if (it != syncScopeName.end())
    return *it;
  llvm_unreachable("incorrect sync scope identifier");
}

/// Converts an array of unsigned indices to a signed integer position array.
static SmallVector<int64_t> getPositionFromIndices(ArrayRef<unsigned> indices) {
  SmallVector<int64_t> position;
  llvm::append_range(position, indices);
  return position;
}

/// Converts the LLVM instructions that have a generated MLIR builder. Using a
/// static implementation method called from the module import ensures the
/// builders have to use the `moduleImport` argument and cannot directly call
/// import methods. As a result, both the intrinsic and the instruction MLIR
/// builders have to use the `moduleImport` argument and none of them has direct
/// access to the private module import methods.
static LogicalResult convertInstructionImpl(OpBuilder &odsBuilder,
                                            llvm::Instruction *inst,
                                            ModuleImport &moduleImport) {
  // Copy the operands to an LLVM operands array reference for conversion.
  SmallVector<llvm::Value *> operands(inst->operands());
  ArrayRef<llvm::Value *> llvmOperands(operands);

  // Convert all instructions that provide an MLIR builder.
#include "mlir/Dialect/LLVMIR/LLVMOpFromLLVMIRConversions.inc"
  return failure();
}

/// Get a topologically sorted list of blocks for the given function.
static SetVector<llvm::BasicBlock *>
getTopologicallySortedBlocks(llvm::Function *func) {
  SetVector<llvm::BasicBlock *> blocks;
  for (llvm::BasicBlock &bb : *func) {
    if (!blocks.contains(&bb)) {
      llvm::ReversePostOrderTraversal<llvm::BasicBlock *> traversal(&bb);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }
  assert(blocks.size() == func->size() && "some blocks are not sorted");

  return blocks;
}

ModuleImport::ModuleImport(ModuleOp mlirModule,
                           std::unique_ptr<llvm::Module> llvmModule,
                           bool emitExpensiveWarnings)
    : builder(mlirModule->getContext()), context(mlirModule->getContext()),
      mlirModule(mlirModule), llvmModule(std::move(llvmModule)),
      iface(mlirModule->getContext()),
      typeTranslator(*mlirModule->getContext()),
      debugImporter(std::make_unique<DebugImporter>(mlirModule)),
      loopAnnotationImporter(
          std::make_unique<LoopAnnotationImporter>(*this, builder)),
      emitExpensiveWarnings(emitExpensiveWarnings) {
  builder.setInsertionPointToStart(mlirModule.getBody());
}

ComdatOp ModuleImport::getGlobalComdatOp() {
  if (globalComdatOp)
    return globalComdatOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());
  globalComdatOp =
      builder.create<ComdatOp>(mlirModule.getLoc(), getGlobalComdatOpName());
  globalInsertionOp = globalComdatOp;
  return globalComdatOp;
}

LogicalResult ModuleImport::processTBAAMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();

  // If `node` is a valid TBAA root node, then return its optional identity
  // string, otherwise return failure.
  auto getIdentityIfRootNode =
      [&](const llvm::MDNode *node) -> FailureOr<std::optional<StringRef>> {
    // Root node, e.g.:
    //   !0 = !{!"Simple C/C++ TBAA"}
    //   !1 = !{}
    if (node->getNumOperands() > 1)
      return failure();
    // If the operand is MDString, then assume that this is a root node.
    if (node->getNumOperands() == 1)
      if (const auto *op0 = dyn_cast<const llvm::MDString>(node->getOperand(0)))
        return std::optional<StringRef>{op0->getString()};
    return std::optional<StringRef>{};
  };

  // If `node` looks like a TBAA type descriptor metadata,
  // then return true, if it is a valid node, and false otherwise.
  // If it does not look like a TBAA type descriptor metadata, then
  // return std::nullopt.
  // If `identity` and `memberTypes/Offsets` are non-null, then they will
  // contain the converted metadata operands for a valid TBAA node (i.e. when
  // true is returned).
  auto isTypeDescriptorNode = [&](const llvm::MDNode *node,
                                  StringRef *identity = nullptr,
                                  SmallVectorImpl<TBAAMemberAttr> *members =
                                      nullptr) -> std::optional<bool> {
    unsigned numOperands = node->getNumOperands();
    // Type descriptor, e.g.:
    //   !1 = !{!"int", !0, /*optional*/i64 0} /* scalar int type */
    //   !2 = !{!"agg_t", !1, i64 0} /* struct agg_t { int x; } */
    if (numOperands < 2)
      return std::nullopt;

    // TODO: support "new" format (D41501) for type descriptors,
    //       where the first operand is an MDNode.
    const auto *identityNode =
        dyn_cast<const llvm::MDString>(node->getOperand(0));
    if (!identityNode)
      return std::nullopt;

    // This should be a type descriptor node.
    if (identity)
      *identity = identityNode->getString();

    for (unsigned pairNum = 0, e = numOperands / 2; pairNum < e; ++pairNum) {
      const auto *memberNode =
          dyn_cast<const llvm::MDNode>(node->getOperand(2 * pairNum + 1));
      if (!memberNode) {
        emitError(loc) << "operand '" << 2 * pairNum + 1 << "' must be MDNode: "
                       << diagMD(node, llvmModule.get());
        return false;
      }
      int64_t offset = 0;
      if (2 * pairNum + 2 >= numOperands) {
        // Allow for optional 0 offset in 2-operand nodes.
        if (numOperands != 2) {
          emitError(loc) << "missing member offset: "
                         << diagMD(node, llvmModule.get());
          return false;
        }
      } else {
        auto *offsetCI = llvm::mdconst::dyn_extract<llvm::ConstantInt>(
            node->getOperand(2 * pairNum + 2));
        if (!offsetCI) {
          emitError(loc) << "operand '" << 2 * pairNum + 2
                         << "' must be ConstantInt: "
                         << diagMD(node, llvmModule.get());
          return false;
        }
        offset = offsetCI->getZExtValue();
      }

      if (members)
        members->push_back(TBAAMemberAttr::get(
            cast<TBAANodeAttr>(tbaaMapping.lookup(memberNode)), offset));
    }

    return true;
  };

  // If `node` looks like a TBAA access tag metadata,
  // then return true, if it is a valid node, and false otherwise.
  // If it does not look like a TBAA access tag metadata, then
  // return std::nullopt.
  // If the other arguments are non-null, then they will contain
  // the converted metadata operands for a valid TBAA node (i.e. when true is
  // returned).
  auto isTagNode = [&](const llvm::MDNode *node,
                       TBAATypeDescriptorAttr *baseAttr = nullptr,
                       TBAATypeDescriptorAttr *accessAttr = nullptr,
                       int64_t *offset = nullptr,
                       bool *isConstant = nullptr) -> std::optional<bool> {
    // Access tag, e.g.:
    //   !3 = !{!1, !1, i64 0} /* scalar int access */
    //   !4 = !{!2, !1, i64 0} /* agg_t::x access */
    //
    // Optional 4th argument is ConstantInt 0/1 identifying whether
    // the location being accessed is "constant" (see for details:
    // https://llvm.org/docs/LangRef.html#representation).
    unsigned numOperands = node->getNumOperands();
    if (numOperands != 3 && numOperands != 4)
      return std::nullopt;
    const auto *baseMD = dyn_cast<const llvm::MDNode>(node->getOperand(0));
    const auto *accessMD = dyn_cast<const llvm::MDNode>(node->getOperand(1));
    auto *offsetCI =
        llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(2));
    if (!baseMD || !accessMD || !offsetCI)
      return std::nullopt;
    // TODO: support "new" TBAA format, if needed (see D41501).
    // In the "old" format the first operand of the access type
    // metadata is MDString. We have to distinguish the formats,
    // because access tags have the same structure, but different
    // meaning for the operands.
    if (accessMD->getNumOperands() < 1 ||
        !isa<llvm::MDString>(accessMD->getOperand(0)))
      return std::nullopt;
    bool isConst = false;
    if (numOperands == 4) {
      auto *isConstantCI =
          llvm::mdconst::dyn_extract<llvm::ConstantInt>(node->getOperand(3));
      if (!isConstantCI) {
        emitError(loc) << "operand '3' must be ConstantInt: "
                       << diagMD(node, llvmModule.get());
        return false;
      }
      isConst = isConstantCI->getValue()[0];
    }
    if (baseAttr)
      *baseAttr = cast<TBAATypeDescriptorAttr>(tbaaMapping.lookup(baseMD));
    if (accessAttr)
      *accessAttr = cast<TBAATypeDescriptorAttr>(tbaaMapping.lookup(accessMD));
    if (offset)
      *offset = offsetCI->getZExtValue();
    if (isConstant)
      *isConstant = isConst;
    return true;
  };

  // Do a post-order walk over the TBAA Graph. Since a correct TBAA Graph is a
  // DAG, a post-order walk guarantees that we convert any metadata node we
  // depend on, prior to converting the current node.
  DenseSet<const llvm::MDNode *> seen;
  SmallVector<const llvm::MDNode *> workList;
  workList.push_back(node);
  while (!workList.empty()) {
    const llvm::MDNode *current = workList.back();
    if (tbaaMapping.contains(current)) {
      // Already converted. Just pop from the worklist.
      workList.pop_back();
      continue;
    }

    // If any child of this node is not yet converted, don't pop the current
    // node from the worklist but push the not-yet-converted children in the
    // front of the worklist.
    bool anyChildNotConverted = false;
    for (const llvm::MDOperand &operand : current->operands())
      if (auto *childNode = dyn_cast_or_null<const llvm::MDNode>(operand.get()))
        if (!tbaaMapping.contains(childNode)) {
          workList.push_back(childNode);
          anyChildNotConverted = true;
        }

    if (anyChildNotConverted) {
      // If this is the second time we failed to convert an element in the
      // worklist it must be because a child is dependent on it being converted
      // and we have a cycle in the graph. Cycles are not allowed in TBAA
      // graphs.
      if (!seen.insert(current).second)
        return emitError(loc) << "has cycle in TBAA graph: "
                              << diagMD(current, llvmModule.get());

      continue;
    }

    // Otherwise simply import the current node.
    workList.pop_back();

    FailureOr<std::optional<StringRef>> rootNodeIdentity =
        getIdentityIfRootNode(current);
    if (succeeded(rootNodeIdentity)) {
      StringAttr stringAttr = *rootNodeIdentity
                                  ? builder.getStringAttr(**rootNodeIdentity)
                                  : nullptr;
      // The root nodes do not have operands, so we can create
      // the TBAARootMetadataOp on the first walk.
      tbaaMapping.insert({current, builder.getAttr<TBAARootAttr>(stringAttr)});
      continue;
    }

    StringRef identity;
    SmallVector<TBAAMemberAttr> members;
    if (std::optional<bool> isValid =
            isTypeDescriptorNode(current, &identity, &members)) {
      assert(isValid.value() && "type descriptor node must be valid");

      tbaaMapping.insert({current, builder.getAttr<TBAATypeDescriptorAttr>(
                                       identity, members)});
      continue;
    }

    TBAATypeDescriptorAttr baseAttr, accessAttr;
    int64_t offset;
    bool isConstant;
    if (std::optional<bool> isValid =
            isTagNode(current, &baseAttr, &accessAttr, &offset, &isConstant)) {
      assert(isValid.value() && "access tag node must be valid");
      tbaaMapping.insert(
          {current, builder.getAttr<TBAATagAttr>(baseAttr, accessAttr, offset,
                                                 isConstant)});
      continue;
    }

    return emitError(loc) << "unsupported TBAA node format: "
                          << diagMD(current, llvmModule.get());
  }
  return success();
}

LogicalResult
ModuleImport::processAccessGroupMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();
  if (failed(loopAnnotationImporter->translateAccessGroup(node, loc)))
    return emitError(loc) << "unsupported access group node: "
                          << diagMD(node, llvmModule.get());
  return success();
}

LogicalResult
ModuleImport::processAliasScopeMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();
  // Helper that verifies the node has a self reference operand.
  auto verifySelfRef = [](const llvm::MDNode *node) {
    return node->getNumOperands() != 0 &&
           node == dyn_cast<llvm::MDNode>(node->getOperand(0));
  };
  // Helper that verifies the given operand is a string or does not exist.
  auto verifyDescription = [](const llvm::MDNode *node, unsigned idx) {
    return idx >= node->getNumOperands() ||
           isa<llvm::MDString>(node->getOperand(idx));
  };
  // Helper that creates an alias scope domain attribute.
  auto createAliasScopeDomainOp = [&](const llvm::MDNode *aliasDomain) {
    StringAttr description = nullptr;
    if (aliasDomain->getNumOperands() >= 2)
      if (auto *operand = dyn_cast<llvm::MDString>(aliasDomain->getOperand(1)))
        description = builder.getStringAttr(operand->getString());
    return builder.getAttr<AliasScopeDomainAttr>(
        DistinctAttr::create(builder.getUnitAttr()), description);
  };

  // Collect the alias scopes and domains to translate them.
  for (const llvm::MDOperand &operand : node->operands()) {
    if (const auto *scope = dyn_cast<llvm::MDNode>(operand)) {
      llvm::AliasScopeNode aliasScope(scope);
      const llvm::MDNode *domain = aliasScope.getDomain();

      // Verify the scope node points to valid scope metadata which includes
      // verifying its domain. Perform the verification before looking it up in
      // the alias scope mapping since it could have been inserted as a domain
      // node before.
      if (!verifySelfRef(scope) || !domain || !verifyDescription(scope, 2))
        return emitError(loc) << "unsupported alias scope node: "
                              << diagMD(scope, llvmModule.get());
      if (!verifySelfRef(domain) || !verifyDescription(domain, 1))
        return emitError(loc) << "unsupported alias domain node: "
                              << diagMD(domain, llvmModule.get());

      if (aliasScopeMapping.contains(scope))
        continue;

      // Convert the domain metadata node if it has not been translated before.
      auto it = aliasScopeMapping.find(aliasScope.getDomain());
      if (it == aliasScopeMapping.end()) {
        auto aliasScopeDomainOp = createAliasScopeDomainOp(domain);
        it = aliasScopeMapping.try_emplace(domain, aliasScopeDomainOp).first;
      }

      // Convert the scope metadata node if it has not been converted before.
      StringAttr description = nullptr;
      if (!aliasScope.getName().empty())
        description = builder.getStringAttr(aliasScope.getName());
      auto aliasScopeOp = builder.getAttr<AliasScopeAttr>(
          DistinctAttr::create(builder.getUnitAttr()),
          cast<AliasScopeDomainAttr>(it->second), description);
      aliasScopeMapping.try_emplace(aliasScope.getNode(), aliasScopeOp);
    }
  }
  return success();
}

FailureOr<SmallVector<AliasScopeAttr>>
ModuleImport::lookupAliasScopeAttrs(const llvm::MDNode *node) const {
  SmallVector<AliasScopeAttr> aliasScopes;
  aliasScopes.reserve(node->getNumOperands());
  for (const llvm::MDOperand &operand : node->operands()) {
    auto *node = cast<llvm::MDNode>(operand.get());
    aliasScopes.push_back(
        dyn_cast_or_null<AliasScopeAttr>(aliasScopeMapping.lookup(node)));
  }
  // Return failure if one of the alias scope lookups failed.
  if (llvm::is_contained(aliasScopes, nullptr))
    return failure();
  return aliasScopes;
}

LogicalResult ModuleImport::convertMetadata() {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());
  for (const llvm::Function &func : llvmModule->functions()) {
    for (const llvm::Instruction &inst : llvm::instructions(func)) {
      // Convert access group metadata nodes.
      if (llvm::MDNode *node =
              inst.getMetadata(llvm::LLVMContext::MD_access_group))
        if (failed(processAccessGroupMetadata(node)))
          return failure();

      // Convert alias analysis metadata nodes.
      llvm::AAMDNodes aliasAnalysisNodes = inst.getAAMetadata();
      if (!aliasAnalysisNodes)
        continue;
      if (aliasAnalysisNodes.TBAA)
        if (failed(processTBAAMetadata(aliasAnalysisNodes.TBAA)))
          return failure();
      if (aliasAnalysisNodes.Scope)
        if (failed(processAliasScopeMetadata(aliasAnalysisNodes.Scope)))
          return failure();
      if (aliasAnalysisNodes.NoAlias)
        if (failed(processAliasScopeMetadata(aliasAnalysisNodes.NoAlias)))
          return failure();
    }
  }
  return success();
}

void ModuleImport::processComdat(const llvm::Comdat *comdat) {
  if (comdatMapping.contains(comdat))
    return;

  ComdatOp comdatOp = getGlobalComdatOp();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&comdatOp.getBody().back());
  auto selectorOp = builder.create<ComdatSelectorOp>(
      mlirModule.getLoc(), comdat->getName(),
      convertComdatFromLLVM(comdat->getSelectionKind()));
  auto symbolRef =
      SymbolRefAttr::get(builder.getContext(), getGlobalComdatOpName(),
                         FlatSymbolRefAttr::get(selectorOp.getSymNameAttr()));
  comdatMapping.try_emplace(comdat, symbolRef);
}

LogicalResult ModuleImport::convertComdats() {
  for (llvm::GlobalVariable &globalVar : llvmModule->globals())
    if (globalVar.hasComdat())
      processComdat(globalVar.getComdat());
  for (llvm::Function &func : llvmModule->functions())
    if (func.hasComdat())
      processComdat(func.getComdat());
  return success();
}

LogicalResult ModuleImport::convertGlobals() {
  for (llvm::GlobalVariable &globalVar : llvmModule->globals()) {
    if (globalVar.getName() == getGlobalCtorsVarName() ||
        globalVar.getName() == getGlobalDtorsVarName()) {
      if (failed(convertGlobalCtorsAndDtors(&globalVar))) {
        return emitError(UnknownLoc::get(context))
               << "unhandled global variable: " << diag(globalVar);
      }
      continue;
    }
    if (failed(convertGlobal(&globalVar))) {
      return emitError(UnknownLoc::get(context))
             << "unhandled global variable: " << diag(globalVar);
    }
  }
  return success();
}

LogicalResult ModuleImport::convertDataLayout() {
  Location loc = mlirModule.getLoc();
  DataLayoutImporter dataLayoutImporter(context, llvmModule->getDataLayout());
  if (!dataLayoutImporter.getDataLayout())
    return emitError(loc, "cannot translate data layout: ")
           << dataLayoutImporter.getLastToken();

  for (StringRef token : dataLayoutImporter.getUnhandledTokens())
    emitWarning(loc, "unhandled data layout token: ") << token;

  mlirModule->setAttr(DLTIDialect::kDataLayoutAttrName,
                      dataLayoutImporter.getDataLayout());
  return success();
}

LogicalResult ModuleImport::convertFunctions() {
  for (llvm::Function &func : llvmModule->functions())
    if (failed(processFunction(&func)))
      return failure();
  return success();
}

void ModuleImport::setNonDebugMetadataAttrs(llvm::Instruction *inst,
                                            Operation *op) {
  SmallVector<std::pair<unsigned, llvm::MDNode *>> allMetadata;
  inst->getAllMetadataOtherThanDebugLoc(allMetadata);
  for (auto &[kind, node] : allMetadata) {
    if (!iface.isConvertibleMetadata(kind))
      continue;
    if (failed(iface.setMetadataAttrs(builder, kind, node, op, *this))) {
      if (emitExpensiveWarnings) {
        Location loc = debugImporter->translateLoc(inst->getDebugLoc());
        emitWarning(loc) << "unhandled metadata: "
                         << diagMD(node, llvmModule.get()) << " on "
                         << diag(*inst);
      }
    }
  }
}

void ModuleImport::setFastmathFlagsAttr(llvm::Instruction *inst,
                                        Operation *op) const {
  auto iface = cast<FastmathFlagsInterface>(op);

  // Even if the imported operation implements the fastmath interface, the
  // original instruction may not have fastmath flags set. Exit if an
  // instruction, such as a non floating-point function call, does not have
  // fastmath flags.
  if (!isa<llvm::FPMathOperator>(inst))
    return;
  llvm::FastMathFlags flags = inst->getFastMathFlags();

  // Set the fastmath bits flag-by-flag.
  FastmathFlags value = {};
  value = bitEnumSet(value, FastmathFlags::nnan, flags.noNaNs());
  value = bitEnumSet(value, FastmathFlags::ninf, flags.noInfs());
  value = bitEnumSet(value, FastmathFlags::nsz, flags.noSignedZeros());
  value = bitEnumSet(value, FastmathFlags::arcp, flags.allowReciprocal());
  value = bitEnumSet(value, FastmathFlags::contract, flags.allowContract());
  value = bitEnumSet(value, FastmathFlags::afn, flags.approxFunc());
  value = bitEnumSet(value, FastmathFlags::reassoc, flags.allowReassoc());
  FastmathFlagsAttr attr = FastmathFlagsAttr::get(builder.getContext(), value);
  iface->setAttr(iface.getFastmathAttrName(), attr);
}

/// Returns if `type` is a scalar integer or floating-point type.
static bool isScalarType(Type type) {
  return isa<IntegerType, FloatType>(type);
}

/// Returns `type` if it is a builtin integer or floating-point vector type that
/// can be used to create an attribute or nullptr otherwise. If provided,
/// `arrayShape` is added to the shape of the vector to create an attribute that
/// matches an array of vectors.
static Type getVectorTypeForAttr(Type type, ArrayRef<int64_t> arrayShape = {}) {
  if (!LLVM::isCompatibleVectorType(type))
    return {};

  llvm::ElementCount numElements = LLVM::getVectorNumElements(type);
  if (numElements.isScalable()) {
    emitError(UnknownLoc::get(type.getContext()))
        << "scalable vectors not supported";
    return {};
  }

  // An LLVM dialect vector can only contain scalars.
  Type elementType = LLVM::getVectorElementType(type);
  if (!isScalarType(elementType))
    return {};

  SmallVector<int64_t> shape(arrayShape.begin(), arrayShape.end());
  shape.push_back(numElements.getKnownMinValue());
  return VectorType::get(shape, elementType);
}

Type ModuleImport::getBuiltinTypeForAttr(Type type) {
  if (!type)
    return {};

  // Return builtin integer and floating-point types as is.
  if (isScalarType(type))
    return type;

  // Return builtin vectors of integer and floating-point types as is.
  if (Type vectorType = getVectorTypeForAttr(type))
    return vectorType;

  // Multi-dimensional array types are converted to tensors or vectors,
  // depending on the innermost type being a scalar or a vector.
  SmallVector<int64_t> arrayShape;
  while (auto arrayType = dyn_cast<LLVMArrayType>(type)) {
    arrayShape.push_back(arrayType.getNumElements());
    type = arrayType.getElementType();
  }
  if (isScalarType(type))
    return RankedTensorType::get(arrayShape, type);
  return getVectorTypeForAttr(type, arrayShape);
}

/// Returns an integer or float attribute for the provided scalar constant
/// `constScalar` or nullptr if the conversion fails.
static Attribute getScalarConstantAsAttr(OpBuilder &builder,
                                         llvm::Constant *constScalar) {
  MLIRContext *context = builder.getContext();

  // Convert scalar intergers.
  if (auto *constInt = dyn_cast<llvm::ConstantInt>(constScalar)) {
    return builder.getIntegerAttr(
        IntegerType::get(context, constInt->getType()->getBitWidth()),
        constInt->getValue());
  }

  // Convert scalar floats.
  if (auto *constFloat = dyn_cast<llvm::ConstantFP>(constScalar)) {
    llvm::Type *type = constFloat->getType();
    FloatType floatType =
        type->isBFloatTy()
            ? FloatType::getBF16(context)
            : LLVM::detail::getFloatType(context, type->getScalarSizeInBits());
    if (!floatType) {
      emitError(UnknownLoc::get(builder.getContext()))
          << "unexpected floating-point type";
      return {};
    }
    return builder.getFloatAttr(floatType, constFloat->getValueAPF());
  }
  return {};
}

/// Returns an integer or float attribute array for the provided constant
/// sequence `constSequence` or nullptr if the conversion fails.
static SmallVector<Attribute>
getSequenceConstantAsAttrs(OpBuilder &builder,
                           llvm::ConstantDataSequential *constSequence) {
  SmallVector<Attribute> elementAttrs;
  elementAttrs.reserve(constSequence->getNumElements());
  for (auto idx : llvm::seq<int64_t>(0, constSequence->getNumElements())) {
    llvm::Constant *constElement = constSequence->getElementAsConstant(idx);
    elementAttrs.push_back(getScalarConstantAsAttr(builder, constElement));
  }
  return elementAttrs;
}

Attribute ModuleImport::getConstantAsAttr(llvm::Constant *constant) {
  // Convert scalar constants.
  if (Attribute scalarAttr = getScalarConstantAsAttr(builder, constant))
    return scalarAttr;

  // Convert function references.
  if (auto *func = dyn_cast<llvm::Function>(constant))
    return SymbolRefAttr::get(builder.getContext(), func->getName());

  // Returns the static shape of the provided type if possible.
  auto getConstantShape = [&](llvm::Type *type) {
    return llvm::dyn_cast_if_present<ShapedType>(
        getBuiltinTypeForAttr(convertType(type)));
  };

  // Convert one-dimensional constant arrays or vectors that store 1/2/4/8-byte
  // integer or half/bfloat/float/double values.
  if (auto *constArray = dyn_cast<llvm::ConstantDataSequential>(constant)) {
    if (constArray->isString())
      return builder.getStringAttr(constArray->getAsString());
    auto shape = getConstantShape(constArray->getType());
    if (!shape)
      return {};
    // Convert splat constants to splat elements attributes.
    auto *constVector = dyn_cast<llvm::ConstantDataVector>(constant);
    if (constVector && constVector->isSplat()) {
      // A vector is guaranteed to have at least size one.
      Attribute splatAttr = getScalarConstantAsAttr(
          builder, constVector->getElementAsConstant(0));
      return SplatElementsAttr::get(shape, splatAttr);
    }
    // Convert non-splat constants to dense elements attributes.
    SmallVector<Attribute> elementAttrs =
        getSequenceConstantAsAttrs(builder, constArray);
    return DenseElementsAttr::get(shape, elementAttrs);
  }

  // Convert multi-dimensional constant aggregates that store all kinds of
  // integer and floating-point types.
  if (auto *constAggregate = dyn_cast<llvm::ConstantAggregate>(constant)) {
    auto shape = getConstantShape(constAggregate->getType());
    if (!shape)
      return {};
    // Collect the aggregate elements in depths first order.
    SmallVector<Attribute> elementAttrs;
    SmallVector<llvm::Constant *> workList = {constAggregate};
    while (!workList.empty()) {
      llvm::Constant *current = workList.pop_back_val();
      // Append any nested aggregates in reverse order to ensure the head
      // element of the nested aggregates is at the back of the work list.
      if (auto *constAggregate = dyn_cast<llvm::ConstantAggregate>(current)) {
        for (auto idx :
             reverse(llvm::seq<int64_t>(0, constAggregate->getNumOperands())))
          workList.push_back(constAggregate->getAggregateElement(idx));
        continue;
      }
      // Append the elements of nested constant arrays or vectors that store
      // 1/2/4/8-byte integer or half/bfloat/float/double values.
      if (auto *constArray = dyn_cast<llvm::ConstantDataSequential>(current)) {
        SmallVector<Attribute> attrs =
            getSequenceConstantAsAttrs(builder, constArray);
        elementAttrs.append(attrs.begin(), attrs.end());
        continue;
      }
      // Append nested scalar constants that store all kinds of integer and
      // floating-point types.
      if (Attribute scalarAttr = getScalarConstantAsAttr(builder, current)) {
        elementAttrs.push_back(scalarAttr);
        continue;
      }
      // Bail if the aggregate contains a unsupported constant type such as a
      // constant expression.
      return {};
    }
    return DenseElementsAttr::get(shape, elementAttrs);
  }

  // Convert zero aggregates.
  if (auto *constZero = dyn_cast<llvm::ConstantAggregateZero>(constant)) {
    auto shape = llvm::dyn_cast_if_present<ShapedType>(
        getBuiltinTypeForAttr(convertType(constZero->getType())));
    if (!shape)
      return {};
    // Convert zero aggregates with a static shape to splat elements attributes.
    Attribute splatAttr = builder.getZeroAttr(shape.getElementType());
    assert(splatAttr && "expected non-null zero attribute for scalar types");
    return SplatElementsAttr::get(shape, splatAttr);
  }
  return {};
}

LogicalResult ModuleImport::convertGlobal(llvm::GlobalVariable *globalVar) {
  // Insert the global after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard(builder);
  if (!globalInsertionOp)
    builder.setInsertionPointToStart(mlirModule.getBody());
  else
    builder.setInsertionPointAfter(globalInsertionOp);

  Attribute valueAttr;
  if (globalVar->hasInitializer())
    valueAttr = getConstantAsAttr(globalVar->getInitializer());
  Type type = convertType(globalVar->getValueType());

  uint64_t alignment = 0;
  llvm::MaybeAlign maybeAlign = globalVar->getAlign();
  if (maybeAlign.has_value()) {
    llvm::Align align = *maybeAlign;
    alignment = align.value();
  }

  GlobalOp globalOp = builder.create<GlobalOp>(
      mlirModule.getLoc(), type, globalVar->isConstant(),
      convertLinkageFromLLVM(globalVar->getLinkage()), globalVar->getName(),
      valueAttr, alignment, /*addr_space=*/globalVar->getAddressSpace(),
      /*dso_local=*/globalVar->isDSOLocal(),
      /*thread_local=*/globalVar->isThreadLocal());
  globalInsertionOp = globalOp;

  if (globalVar->hasInitializer() && !valueAttr) {
    clearBlockAndValueMapping();
    Block *block = builder.createBlock(&globalOp.getInitializerRegion());
    setConstantInsertionPointToStart(block);
    FailureOr<Value> initializer =
        convertConstantExpr(globalVar->getInitializer());
    if (failed(initializer))
      return failure();
    builder.create<ReturnOp>(globalOp.getLoc(), *initializer);
  }
  if (globalVar->hasAtLeastLocalUnnamedAddr()) {
    globalOp.setUnnamedAddr(
        convertUnnamedAddrFromLLVM(globalVar->getUnnamedAddr()));
  }
  if (globalVar->hasSection())
    globalOp.setSection(globalVar->getSection());
  globalOp.setVisibility_(
      convertVisibilityFromLLVM(globalVar->getVisibility()));

  if (globalVar->hasComdat())
    globalOp.setComdatAttr(comdatMapping.lookup(globalVar->getComdat()));

  return success();
}

LogicalResult
ModuleImport::convertGlobalCtorsAndDtors(llvm::GlobalVariable *globalVar) {
  if (!globalVar->hasInitializer() || !globalVar->hasAppendingLinkage())
    return failure();
  auto *initializer =
      dyn_cast<llvm::ConstantArray>(globalVar->getInitializer());
  if (!initializer)
    return failure();

  SmallVector<Attribute> funcs;
  SmallVector<int32_t> priorities;
  for (llvm::Value *operand : initializer->operands()) {
    auto *aggregate = dyn_cast<llvm::ConstantAggregate>(operand);
    if (!aggregate || aggregate->getNumOperands() != 3)
      return failure();

    auto *priority = dyn_cast<llvm::ConstantInt>(aggregate->getOperand(0));
    auto *func = dyn_cast<llvm::Function>(aggregate->getOperand(1));
    auto *data = dyn_cast<llvm::Constant>(aggregate->getOperand(2));
    if (!priority || !func || !data)
      return failure();

    // GlobalCtorsOps and GlobalDtorsOps do not support non-null data fields.
    if (!data->isNullValue())
      return failure();

    funcs.push_back(FlatSymbolRefAttr::get(context, func->getName()));
    priorities.push_back(priority->getValue().getZExtValue());
  }

  OpBuilder::InsertionGuard guard(builder);
  if (!globalInsertionOp)
    builder.setInsertionPointToStart(mlirModule.getBody());
  else
    builder.setInsertionPointAfter(globalInsertionOp);

  if (globalVar->getName() == getGlobalCtorsVarName()) {
    globalInsertionOp = builder.create<LLVM::GlobalCtorsOp>(
        mlirModule.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities));
    return success();
  }
  globalInsertionOp = builder.create<LLVM::GlobalDtorsOp>(
      mlirModule.getLoc(), builder.getArrayAttr(funcs),
      builder.getI32ArrayAttr(priorities));
  return success();
}

SetVector<llvm::Constant *>
ModuleImport::getConstantsToConvert(llvm::Constant *constant) {
  // Return the empty set if the constant has been translated before.
  if (valueMapping.contains(constant))
    return {};

  // Traverse the constants in post-order and stop the traversal if a constant
  // already has a `valueMapping` from an earlier constant translation or if the
  // constant is traversed a second time.
  SetVector<llvm::Constant *> orderedSet;
  SetVector<llvm::Constant *> workList;
  DenseMap<llvm::Constant *, SmallVector<llvm::Constant *>> adjacencyLists;
  workList.insert(constant);
  while (!workList.empty()) {
    llvm::Constant *current = workList.back();
    // Collect all dependencies of the current constant and add them to the
    // adjacency list if none has been computed before.
    auto adjacencyIt = adjacencyLists.find(current);
    if (adjacencyIt == adjacencyLists.end()) {
      adjacencyIt = adjacencyLists.try_emplace(current).first;
      // Add all constant operands to the adjacency list and skip any other
      // values such as basic block addresses.
      for (llvm::Value *operand : current->operands())
        if (auto *constDependency = dyn_cast<llvm::Constant>(operand))
          adjacencyIt->getSecond().push_back(constDependency);
      // Use the getElementValue method to add the dependencies of zero
      // initialized aggregate constants since they do not take any operands.
      if (auto *constAgg = dyn_cast<llvm::ConstantAggregateZero>(current)) {
        unsigned numElements = constAgg->getElementCount().getFixedValue();
        for (unsigned i = 0, e = numElements; i != e; ++i)
          adjacencyIt->getSecond().push_back(constAgg->getElementValue(i));
      }
    }
    // Add the current constant to the `orderedSet` of the traversed nodes if
    // all its dependencies have been traversed before. Additionally, remove the
    // constant from the `workList` and continue the traversal.
    if (adjacencyIt->getSecond().empty()) {
      orderedSet.insert(current);
      workList.pop_back();
      continue;
    }
    // Add the next dependency from the adjacency list to the `workList` and
    // continue the traversal. Remove the dependency from the adjacency list to
    // mark that it has been processed. Only enqueue the dependency if it has no
    // `valueMapping` from an earlier translation and if it has not been
    // enqueued before.
    llvm::Constant *dependency = adjacencyIt->getSecond().pop_back_val();
    if (valueMapping.contains(dependency) || workList.contains(dependency) ||
        orderedSet.contains(dependency))
      continue;
    workList.insert(dependency);
  }

  return orderedSet;
}

FailureOr<Value> ModuleImport::convertConstant(llvm::Constant *constant) {
  Location loc = UnknownLoc::get(context);

  // Convert constants that can be represented as attributes.
  if (Attribute attr = getConstantAsAttr(constant)) {
    Type type = convertType(constant->getType());
    if (auto symbolRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
      return builder.create<AddressOfOp>(loc, type, symbolRef.getValue())
          .getResult();
    }
    return builder.create<ConstantOp>(loc, type, attr).getResult();
  }

  // Convert null pointer constants.
  if (auto *nullPtr = dyn_cast<llvm::ConstantPointerNull>(constant)) {
    Type type = convertType(nullPtr->getType());
    return builder.create<NullOp>(loc, type).getResult();
  }

  // Convert poison.
  if (auto *poisonVal = dyn_cast<llvm::PoisonValue>(constant)) {
    Type type = convertType(poisonVal->getType());
    return builder.create<PoisonOp>(loc, type).getResult();
  }

  // Convert undef.
  if (auto *undefVal = dyn_cast<llvm::UndefValue>(constant)) {
    Type type = convertType(undefVal->getType());
    return builder.create<UndefOp>(loc, type).getResult();
  }

  // Convert global variable accesses.
  if (auto *globalVar = dyn_cast<llvm::GlobalVariable>(constant)) {
    Type type = convertType(globalVar->getType());
    auto symbolRef = FlatSymbolRefAttr::get(context, globalVar->getName());
    return builder.create<AddressOfOp>(loc, type, symbolRef).getResult();
  }

  // Convert constant expressions.
  if (auto *constExpr = dyn_cast<llvm::ConstantExpr>(constant)) {
    // Convert the constant expression to a temporary LLVM instruction and
    // translate it using the `processInstruction` method. Delete the
    // instruction after the translation and remove it from `valueMapping`,
    // since later calls to `getAsInstruction` may return the same address
    // resulting in a conflicting `valueMapping` entry.
    llvm::Instruction *inst = constExpr->getAsInstruction();
    auto guard = llvm::make_scope_exit([&]() {
      assert(!noResultOpMapping.contains(inst) &&
             "expected constant expression to return a result");
      valueMapping.erase(inst);
      inst->deleteValue();
    });
    // Note: `processInstruction` does not call `convertConstant` recursively
    // since all constant dependencies have been converted before.
    assert(llvm::all_of(inst->operands(), [&](llvm::Value *value) {
      return valueMapping.contains(value);
    }));
    if (failed(processInstruction(inst)))
      return failure();
    return lookupValue(inst);
  }

  // Convert aggregate constants.
  if (isa<llvm::ConstantAggregate>(constant) ||
      isa<llvm::ConstantAggregateZero>(constant)) {
    // Lookup the aggregate elements that have been converted before.
    SmallVector<Value> elementValues;
    if (auto *constAgg = dyn_cast<llvm::ConstantAggregate>(constant)) {
      elementValues.reserve(constAgg->getNumOperands());
      for (llvm::Value *operand : constAgg->operands())
        elementValues.push_back(lookupValue(operand));
    }
    if (auto *constAgg = dyn_cast<llvm::ConstantAggregateZero>(constant)) {
      unsigned numElements = constAgg->getElementCount().getFixedValue();
      elementValues.reserve(numElements);
      for (unsigned i = 0, e = numElements; i != e; ++i)
        elementValues.push_back(lookupValue(constAgg->getElementValue(i)));
    }
    assert(llvm::count(elementValues, nullptr) == 0 &&
           "expected all elements have been converted before");

    // Generate an UndefOp as root value and insert the aggregate elements.
    Type rootType = convertType(constant->getType());
    bool isArrayOrStruct = isa<LLVMArrayType, LLVMStructType>(rootType);
    assert((isArrayOrStruct || LLVM::isCompatibleVectorType(rootType)) &&
           "unrecognized aggregate type");
    Value root = builder.create<UndefOp>(loc, rootType);
    for (const auto &it : llvm::enumerate(elementValues)) {
      if (isArrayOrStruct) {
        root = builder.create<InsertValueOp>(loc, root, it.value(), it.index());
      } else {
        Attribute indexAttr = builder.getI32IntegerAttr(it.index());
        Value indexValue =
            builder.create<ConstantOp>(loc, builder.getI32Type(), indexAttr);
        root = builder.create<InsertElementOp>(loc, rootType, root, it.value(),
                                               indexValue);
      }
    }
    return root;
  }

  if (auto *constTargetNone = dyn_cast<llvm::ConstantTargetNone>(constant)) {
    LLVMTargetExtType targetExtType =
        cast<LLVMTargetExtType>(convertType(constTargetNone->getType()));
    assert(targetExtType.hasProperty(LLVMTargetExtType::HasZeroInit) &&
           "target extension type does not support zero-initialization");
    // As the number of values needed for initialization is target-specific and
    // opaque to the compiler, use a single i64 zero-valued attribute to
    // represent the 'zeroinitializer', which is the only constant value allowed
    // for target extension types (besides poison and undef).
    // TODO: Replace with 'zeroinitializer' once there is a dedicated
    // zeroinitializer operation in the LLVM dialect.
    return builder
        .create<LLVM::ConstantOp>(loc, targetExtType,
                                  builder.getI64IntegerAttr(0))
        .getRes();
  }

  StringRef error = "";
  if (isa<llvm::BlockAddress>(constant))
    error = " since blockaddress(...) is unsupported";

  return emitError(loc) << "unhandled constant: " << diag(*constant) << error;
}

FailureOr<Value> ModuleImport::convertConstantExpr(llvm::Constant *constant) {
  // Only call the function for constants that have not been translated before
  // since it updates the constant insertion point assuming the converted
  // constant has been introduced at the end of the constant section.
  assert(!valueMapping.contains(constant) &&
         "expected constant has not been converted before");
  assert(constantInsertionBlock &&
         "expected the constant insertion block to be non-null");

  // Insert the constant after the last one or at the start of the entry block.
  OpBuilder::InsertionGuard guard(builder);
  if (!constantInsertionOp)
    builder.setInsertionPointToStart(constantInsertionBlock);
  else
    builder.setInsertionPointAfter(constantInsertionOp);

  // Convert all constants of the expression and add them to `valueMapping`.
  SetVector<llvm::Constant *> constantsToConvert =
      getConstantsToConvert(constant);
  for (llvm::Constant *constantToConvert : constantsToConvert) {
    FailureOr<Value> converted = convertConstant(constantToConvert);
    if (failed(converted))
      return failure();
    mapValue(constantToConvert, *converted);
  }

  // Update the constant insertion point and return the converted constant.
  Value result = lookupValue(constant);
  constantInsertionOp = result.getDefiningOp();
  return result;
}

FailureOr<Value> ModuleImport::convertValue(llvm::Value *value) {
  assert(!isa<llvm::MetadataAsValue>(value) &&
         "expected value to not be metadata");

  // Return the mapped value if it has been converted before.
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->getSecond();

  // Convert constants such as immediate values that have no mapping yet.
  if (auto *constant = dyn_cast<llvm::Constant>(value))
    return convertConstantExpr(constant);

  Location loc = UnknownLoc::get(context);
  if (auto *inst = dyn_cast<llvm::Instruction>(value))
    loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled value: " << diag(*value);
}

FailureOr<Value> ModuleImport::convertMetadataValue(llvm::Value *value) {
  // A value may be wrapped as metadata, for example, when passed to a debug
  // intrinsic. Unwrap these values before the conversion.
  auto *nodeAsVal = dyn_cast<llvm::MetadataAsValue>(value);
  if (!nodeAsVal)
    return failure();
  auto *node = dyn_cast<llvm::ValueAsMetadata>(nodeAsVal->getMetadata());
  if (!node)
    return failure();
  value = node->getValue();

  // Return the mapped value if it has been converted before.
  auto it = valueMapping.find(value);
  if (it != valueMapping.end())
    return it->getSecond();

  // Convert constants such as immediate values that have no mapping yet.
  if (auto *constant = dyn_cast<llvm::Constant>(value))
    return convertConstantExpr(constant);
  return failure();
}

FailureOr<SmallVector<Value>>
ModuleImport::convertValues(ArrayRef<llvm::Value *> values) {
  SmallVector<Value> remapped;
  remapped.reserve(values.size());
  for (llvm::Value *value : values) {
    FailureOr<Value> converted = convertValue(value);
    if (failed(converted))
      return failure();
    remapped.push_back(*converted);
  }
  return remapped;
}

IntegerAttr ModuleImport::matchIntegerAttr(llvm::Value *value) {
  IntegerAttr integerAttr;
  FailureOr<Value> converted = convertValue(value);
  bool success = succeeded(converted) &&
                 matchPattern(*converted, m_Constant(&integerAttr));
  assert(success && "expected a constant integer value");
  (void)success;
  return integerAttr;
}

FloatAttr ModuleImport::matchFloatAttr(llvm::Value *value) {
  FloatAttr floatAttr;
  FailureOr<Value> converted = convertValue(value);
  bool success =
      succeeded(converted) && matchPattern(*converted, m_Constant(&floatAttr));
  assert(success && "expected a constant float value");
  (void)success;
  return floatAttr;
}

DILocalVariableAttr ModuleImport::matchLocalVariableAttr(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::DILocalVariable>(nodeAsVal->getMetadata());
  return debugImporter->translate(node);
}

DILabelAttr ModuleImport::matchLabelAttr(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::DILabel>(nodeAsVal->getMetadata());
  return debugImporter->translate(node);
}

FailureOr<SmallVector<AliasScopeAttr>>
ModuleImport::matchAliasScopeAttrs(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::MDNode>(nodeAsVal->getMetadata());
  return lookupAliasScopeAttrs(node);
}

Location ModuleImport::translateLoc(llvm::DILocation *loc) {
  return debugImporter->translateLoc(loc);
}

LogicalResult
ModuleImport::convertBranchArgs(llvm::Instruction *branch,
                                llvm::BasicBlock *target,
                                SmallVectorImpl<Value> &blockArguments) {
  for (auto inst = target->begin(); isa<llvm::PHINode>(inst); ++inst) {
    auto *phiInst = cast<llvm::PHINode>(&*inst);
    llvm::Value *value = phiInst->getIncomingValueForBlock(branch->getParent());
    FailureOr<Value> converted = convertValue(value);
    if (failed(converted))
      return failure();
    blockArguments.push_back(*converted);
  }
  return success();
}

LogicalResult
ModuleImport::convertCallTypeAndOperands(llvm::CallBase *callInst,
                                         SmallVectorImpl<Type> &types,
                                         SmallVectorImpl<Value> &operands) {
  if (!callInst->getType()->isVoidTy())
    types.push_back(convertType(callInst->getType()));

  if (!callInst->getCalledFunction()) {
    FailureOr<Value> called = convertValue(callInst->getCalledOperand());
    if (failed(called))
      return failure();
    operands.push_back(*called);
  }
  SmallVector<llvm::Value *> args(callInst->args());
  FailureOr<SmallVector<Value>> arguments = convertValues(args);
  if (failed(arguments))
    return failure();
  llvm::append_range(operands, *arguments);
  return success();
}

LogicalResult ModuleImport::convertIntrinsic(llvm::CallInst *inst) {
  if (succeeded(iface.convertIntrinsic(builder, inst, *this)))
    return success();

  Location loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled intrinsic: " << diag(*inst);
}

LogicalResult ModuleImport::convertInstruction(llvm::Instruction *inst) {
  // Convert all instructions that do not provide an MLIR builder.
  Location loc = translateLoc(inst->getDebugLoc());
  if (inst->getOpcode() == llvm::Instruction::Br) {
    auto *brInst = cast<llvm::BranchInst>(inst);

    SmallVector<Block *> succBlocks;
    SmallVector<SmallVector<Value>> succBlockArgs;
    for (auto i : llvm::seq<unsigned>(0, brInst->getNumSuccessors())) {
      llvm::BasicBlock *succ = brInst->getSuccessor(i);
      SmallVector<Value> blockArgs;
      if (failed(convertBranchArgs(brInst, succ, blockArgs)))
        return failure();
      succBlocks.push_back(lookupBlock(succ));
      succBlockArgs.push_back(blockArgs);
    }

    if (!brInst->isConditional()) {
      auto brOp = builder.create<LLVM::BrOp>(loc, succBlockArgs.front(),
                                             succBlocks.front());
      mapNoResultOp(inst, brOp);
      return success();
    }
    FailureOr<Value> condition = convertValue(brInst->getCondition());
    if (failed(condition))
      return failure();
    auto condBrOp = builder.create<LLVM::CondBrOp>(
        loc, *condition, succBlocks.front(), succBlockArgs.front(),
        succBlocks.back(), succBlockArgs.back());
    mapNoResultOp(inst, condBrOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Switch) {
    auto *swInst = cast<llvm::SwitchInst>(inst);
    // Process the condition value.
    FailureOr<Value> condition = convertValue(swInst->getCondition());
    if (failed(condition))
      return failure();
    SmallVector<Value> defaultBlockArgs;
    // Process the default case.
    llvm::BasicBlock *defaultBB = swInst->getDefaultDest();
    if (failed(convertBranchArgs(swInst, defaultBB, defaultBlockArgs)))
      return failure();

    // Process the cases.
    unsigned numCases = swInst->getNumCases();
    SmallVector<SmallVector<Value>> caseOperands(numCases);
    SmallVector<ValueRange> caseOperandRefs(numCases);
    SmallVector<APInt> caseValues(numCases);
    SmallVector<Block *> caseBlocks(numCases);
    for (const auto &it : llvm::enumerate(swInst->cases())) {
      const llvm::SwitchInst::CaseHandle &caseHandle = it.value();
      llvm::BasicBlock *succBB = caseHandle.getCaseSuccessor();
      if (failed(convertBranchArgs(swInst, succBB, caseOperands[it.index()])))
        return failure();
      caseOperandRefs[it.index()] = caseOperands[it.index()];
      caseValues[it.index()] = caseHandle.getCaseValue()->getValue();
      caseBlocks[it.index()] = lookupBlock(succBB);
    }

    auto switchOp = builder.create<SwitchOp>(
        loc, *condition, lookupBlock(defaultBB), defaultBlockArgs, caseValues,
        caseBlocks, caseOperandRefs);
    mapNoResultOp(inst, switchOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::PHI) {
    Type type = convertType(inst->getType());
    mapValue(inst, builder.getInsertionBlock()->addArgument(
                       type, translateLoc(inst->getDebugLoc())));
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Call) {
    auto *callInst = cast<llvm::CallInst>(inst);

    SmallVector<Type> types;
    SmallVector<Value> operands;
    if (failed(convertCallTypeAndOperands(callInst, types, operands)))
      return failure();

    CallOp callOp;
    if (llvm::Function *callee = callInst->getCalledFunction()) {
      callOp = builder.create<CallOp>(
          loc, types, SymbolRefAttr::get(context, callee->getName()), operands);
    } else {
      callOp = builder.create<CallOp>(loc, types, operands);
    }
    setFastmathFlagsAttr(inst, callOp);
    if (!callInst->getType()->isVoidTy())
      mapValue(inst, callOp.getResult());
    else
      mapNoResultOp(inst, callOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::LandingPad) {
    auto *lpInst = cast<llvm::LandingPadInst>(inst);

    SmallVector<Value> operands;
    operands.reserve(lpInst->getNumClauses());
    for (auto i : llvm::seq<unsigned>(0, lpInst->getNumClauses())) {
      FailureOr<Value> operand = convertValue(lpInst->getClause(i));
      if (failed(operand))
        return failure();
      operands.push_back(*operand);
    }

    Type type = convertType(lpInst->getType());
    auto lpOp =
        builder.create<LandingpadOp>(loc, type, lpInst->isCleanup(), operands);
    mapValue(inst, lpOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Invoke) {
    auto *invokeInst = cast<llvm::InvokeInst>(inst);

    SmallVector<Type> types;
    SmallVector<Value> operands;
    if (failed(convertCallTypeAndOperands(invokeInst, types, operands)))
      return failure();

    // Check whether the invoke result is an argument to the normal destination
    // block.
    bool invokeResultUsedInPhi = llvm::any_of(
        invokeInst->getNormalDest()->phis(), [&](const llvm::PHINode &phi) {
          return phi.getIncomingValueForBlock(invokeInst->getParent()) ==
                 invokeInst;
        });

    Block *normalDest = lookupBlock(invokeInst->getNormalDest());
    Block *directNormalDest = normalDest;
    if (invokeResultUsedInPhi) {
      // The invoke result cannot be an argument to the normal destination
      // block, as that would imply using the invoke operation result in its
      // definition, so we need to create a dummy block to serve as an
      // intermediate destination.
      OpBuilder::InsertionGuard g(builder);
      directNormalDest = builder.createBlock(normalDest);
    }

    SmallVector<Value> unwindArgs;
    if (failed(convertBranchArgs(invokeInst, invokeInst->getUnwindDest(),
                                 unwindArgs)))
      return failure();

    // Create the invoke operation. Normal destination block arguments will be
    // added later on to handle the case in which the operation result is
    // included in this list.
    InvokeOp invokeOp;
    if (llvm::Function *callee = invokeInst->getCalledFunction()) {
      invokeOp = builder.create<InvokeOp>(
          loc, types,
          SymbolRefAttr::get(builder.getContext(), callee->getName()), operands,
          directNormalDest, ValueRange(),
          lookupBlock(invokeInst->getUnwindDest()), unwindArgs);
    } else {
      invokeOp = builder.create<InvokeOp>(
          loc, types, operands, directNormalDest, ValueRange(),
          lookupBlock(invokeInst->getUnwindDest()), unwindArgs);
    }
    if (!invokeInst->getType()->isVoidTy())
      mapValue(inst, invokeOp.getResults().front());
    else
      mapNoResultOp(inst, invokeOp);

    SmallVector<Value> normalArgs;
    if (failed(convertBranchArgs(invokeInst, invokeInst->getNormalDest(),
                                 normalArgs)))
      return failure();

    if (invokeResultUsedInPhi) {
      // The dummy normal dest block will just host an unconditional branch
      // instruction to the normal destination block passing the required block
      // arguments (including the invoke operation's result).
      OpBuilder::InsertionGuard g(builder);
      builder.setInsertionPointToStart(directNormalDest);
      builder.create<LLVM::BrOp>(loc, normalArgs, normalDest);
    } else {
      // If the invoke operation's result is not a block argument to the normal
      // destination block, just add the block arguments as usual.
      assert(llvm::none_of(
                 normalArgs,
                 [&](Value val) { return val.getDefiningOp() == invokeOp; }) &&
             "An llvm.invoke operation cannot pass its result as a block "
             "argument.");
      invokeOp.getNormalDestOperandsMutable().append(normalArgs);
    }

    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::GetElementPtr) {
    auto *gepInst = cast<llvm::GetElementPtrInst>(inst);
    Type sourceElementType = convertType(gepInst->getSourceElementType());
    FailureOr<Value> basePtr = convertValue(gepInst->getOperand(0));
    if (failed(basePtr))
      return failure();

    // Treat every indices as dynamic since GEPOp::build will refine those
    // indices into static attributes later. One small downside of this
    // approach is that many unused `llvm.mlir.constant` would be emitted
    // at first place.
    SmallVector<GEPArg> indices;
    for (llvm::Value *operand : llvm::drop_begin(gepInst->operand_values())) {
      FailureOr<Value> index = convertValue(operand);
      if (failed(index))
        return failure();
      indices.push_back(*index);
    }

    Type type = convertType(inst->getType());
    auto gepOp = builder.create<GEPOp>(loc, type, sourceElementType, *basePtr,
                                       indices, gepInst->isInBounds());
    mapValue(inst, gepOp);
    return success();
  }

  // Convert all instructions that have an mlirBuilder.
  if (succeeded(convertInstructionImpl(builder, inst, *this)))
    return success();

  return emitError(loc) << "unhandled instruction: " << diag(*inst);
}

LogicalResult ModuleImport::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData.
  // FIXME: Add support for call / operand attributes.
  // FIXME: Add support for the indirectbr, cleanupret, catchret, catchswitch,
  // callbr, vaarg, landingpad, catchpad, cleanuppad instructions.

  // Convert LLVM intrinsics calls to MLIR intrinsics.
  if (auto *callInst = dyn_cast<llvm::CallInst>(inst)) {
    llvm::Function *callee = callInst->getCalledFunction();
    if (callee && callee->isIntrinsic())
      return convertIntrinsic(callInst);
  }

  // Convert all remaining LLVM instructions to MLIR operations.
  return convertInstruction(inst);
}

FlatSymbolRefAttr ModuleImport::getPersonalityAsAttr(llvm::Function *f) {
  if (!f->hasPersonalityFn())
    return nullptr;

  llvm::Constant *pf = f->getPersonalityFn();

  // If it directly has a name, we can use it.
  if (pf->hasName())
    return SymbolRefAttr::get(builder.getContext(), pf->getName());

  // If it doesn't have a name, currently, only function pointers that are
  // bitcast to i8* are parsed.
  if (auto *ce = dyn_cast<llvm::ConstantExpr>(pf)) {
    if (ce->getOpcode() == llvm::Instruction::BitCast &&
        ce->getType() == llvm::Type::getInt8PtrTy(f->getContext())) {
      if (auto *func = dyn_cast<llvm::Function>(ce->getOperand(0)))
        return SymbolRefAttr::get(builder.getContext(), func->getName());
    }
  }
  return FlatSymbolRefAttr();
}

static void processMemoryEffects(llvm::Function *func, LLVMFuncOp funcOp) {
  llvm::MemoryEffects memEffects = func->getMemoryEffects();

  auto othermem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::Other));
  auto argMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::ArgMem));
  auto inaccessibleMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::InaccessibleMem));
  auto memAttr = MemoryEffectsAttr::get(funcOp.getContext(), othermem, argMem,
                                        inaccessibleMem);
  // Only set the attr when it does not match the default value.
  if (memAttr.isReadWrite())
    return;
  funcOp.setMemoryAttr(memAttr);
}

static void processPassthroughAttrs(llvm::Function *func, LLVMFuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  SmallVector<Attribute> passthroughs;
  llvm::AttributeSet funcAttrs = func->getAttributes().getAttributes(
      llvm::AttributeList::AttrIndex::FunctionIndex);
  for (llvm::Attribute attr : funcAttrs) {
    // Skip the memory attribute since the LLVMFuncOp has an explicit memory
    // attribute.
    if (attr.hasAttribute(llvm::Attribute::Memory))
      continue;

    // Skip invalid type attributes.
    if (attr.isTypeAttribute()) {
      emitWarning(funcOp.getLoc(),
                  "type attributes on a function are invalid, skipping it");
      continue;
    }

    StringRef attrName;
    if (attr.isStringAttribute())
      attrName = attr.getKindAsString();
    else
      attrName = llvm::Attribute::getNameFromAttrKind(attr.getKindAsEnum());
    auto keyAttr = StringAttr::get(context, attrName);

    // Skip the aarch64_pstate_sm_<body|enabled> since the LLVMFuncOp has an
    // explicit attribute.
    if (attrName == "aarch64_pstate_sm_enabled" ||
        attrName == "aarch64_pstate_sm_body")
      continue;

    if (attr.isStringAttribute()) {
      StringRef val = attr.getValueAsString();
      if (val.empty()) {
        passthroughs.push_back(keyAttr);
        continue;
      }
      passthroughs.push_back(
          ArrayAttr::get(context, {keyAttr, StringAttr::get(context, val)}));
      continue;
    }
    if (attr.isIntAttribute()) {
      auto val = std::to_string(attr.getValueAsInt());
      passthroughs.push_back(
          ArrayAttr::get(context, {keyAttr, StringAttr::get(context, val)}));
      continue;
    }
    if (attr.isEnumAttribute()) {
      passthroughs.push_back(keyAttr);
      continue;
    }

    llvm_unreachable("unexpected attribute kind");
  }

  if (!passthroughs.empty())
    funcOp.setPassthroughAttr(ArrayAttr::get(context, passthroughs));
}

void ModuleImport::processFunctionAttributes(llvm::Function *func,
                                             LLVMFuncOp funcOp) {
  processMemoryEffects(func, funcOp);
  processPassthroughAttrs(func, funcOp);

  if (func->hasFnAttribute("aarch64_pstate_sm_enabled"))
    funcOp.setArmStreaming(true);
  else if (func->hasFnAttribute("aarch64_pstate_sm_body"))
    funcOp.setArmLocallyStreaming(true);
}

DictionaryAttr
ModuleImport::convertParameterAttribute(llvm::AttributeSet llvmParamAttrs,
                                        OpBuilder &builder) {
  SmallVector<NamedAttribute> paramAttrs;
  for (auto [llvmKind, mlirName] : getAttrKindToNameMapping()) {
    auto llvmAttr = llvmParamAttrs.getAttribute(llvmKind);
    // Skip attributes that are not attached.
    if (!llvmAttr.isValid())
      continue;
    Attribute mlirAttr;
    if (llvmAttr.isTypeAttribute())
      mlirAttr = TypeAttr::get(convertType(llvmAttr.getValueAsType()));
    else if (llvmAttr.isIntAttribute())
      mlirAttr = builder.getI64IntegerAttr(llvmAttr.getValueAsInt());
    else if (llvmAttr.isEnumAttribute())
      mlirAttr = builder.getUnitAttr();
    else
      llvm_unreachable("unexpected parameter attribute kind");
    paramAttrs.push_back(builder.getNamedAttr(mlirName, mlirAttr));
  }

  return builder.getDictionaryAttr(paramAttrs);
}

void ModuleImport::convertParameterAttributes(llvm::Function *func,
                                              LLVMFuncOp funcOp,
                                              OpBuilder &builder) {
  auto llvmAttrs = func->getAttributes();
  for (size_t i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    llvm::AttributeSet llvmArgAttrs = llvmAttrs.getParamAttrs(i);
    funcOp.setArgAttrs(i, convertParameterAttribute(llvmArgAttrs, builder));
  }
  // Convert the result attributes and attach them wrapped in an ArrayAttribute
  // to the funcOp.
  llvm::AttributeSet llvmResAttr = llvmAttrs.getRetAttrs();
  if (!llvmResAttr.hasAttributes())
    return;
  funcOp.setResAttrsAttr(
      builder.getArrayAttr(convertParameterAttribute(llvmResAttr, builder)));
}

LogicalResult ModuleImport::processFunction(llvm::Function *func) {
  clearBlockAndValueMapping();

  auto functionType =
      dyn_cast<LLVMFunctionType>(convertType(func->getFunctionType()));
  if (func->isIntrinsic() &&
      iface.isConvertibleIntrinsic(func->getIntrinsicID()))
    return success();

  bool dsoLocal = func->hasLocalLinkage();
  CConv cconv = convertCConvFromLLVM(func->getCallingConv());

  // Insert the function at the end of the module.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(mlirModule.getBody(), mlirModule.getBody()->end());

  Location loc = debugImporter->translateFuncLocation(func);
  LLVMFuncOp funcOp = builder.create<LLVMFuncOp>(
      loc, func->getName(), functionType,
      convertLinkageFromLLVM(func->getLinkage()), dsoLocal, cconv);

  convertParameterAttributes(func, funcOp, builder);

  if (FlatSymbolRefAttr personality = getPersonalityAsAttr(func))
    funcOp.setPersonalityAttr(personality);
  else if (func->hasPersonalityFn())
    emitWarning(funcOp.getLoc(), "could not deduce personality, skipping it");

  if (func->hasGC())
    funcOp.setGarbageCollector(StringRef(func->getGC()));

  if (func->hasAtLeastLocalUnnamedAddr())
    funcOp.setUnnamedAddr(convertUnnamedAddrFromLLVM(func->getUnnamedAddr()));

  if (func->hasSection())
    funcOp.setSection(StringRef(func->getSection()));

  funcOp.setVisibility_(convertVisibilityFromLLVM(func->getVisibility()));

  if (func->hasComdat())
    funcOp.setComdatAttr(comdatMapping.lookup(func->getComdat()));

  if (llvm::MaybeAlign maybeAlign = func->getAlign())
    funcOp.setAlignment(maybeAlign->value());

  // Handle Function attributes.
  processFunctionAttributes(func, funcOp);

  // Convert non-debug metadata by using the dialect interface.
  SmallVector<std::pair<unsigned, llvm::MDNode *>> allMetadata;
  func->getAllMetadata(allMetadata);
  for (auto &[kind, node] : allMetadata) {
    if (!iface.isConvertibleMetadata(kind))
      continue;
    if (failed(iface.setMetadataAttrs(builder, kind, node, funcOp, *this))) {
      emitWarning(funcOp.getLoc())
          << "unhandled function metadata: " << diagMD(node, llvmModule.get())
          << " on " << diag(*func);
    }
  }

  if (func->isDeclaration())
    return success();

  // Eagerly create all blocks.
  for (llvm::BasicBlock &bb : *func) {
    Block *block =
        builder.createBlock(&funcOp.getBody(), funcOp.getBody().end());
    mapBlock(&bb, block);
  }

  // Add function arguments to the entry block.
  for (const auto &it : llvm::enumerate(func->args())) {
    BlockArgument blockArg = funcOp.getFunctionBody().addArgument(
        functionType.getParamType(it.index()), funcOp.getLoc());
    mapValue(&it.value(), blockArg);
  }

  // Process the blocks in topological order. The ordered traversal ensures
  // operands defined in a dominating block have a valid mapping to an MLIR
  // value once a block is translated.
  SetVector<llvm::BasicBlock *> blocks = getTopologicallySortedBlocks(func);
  setConstantInsertionPointToStart(lookupBlock(blocks.front()));
  for (llvm::BasicBlock *bb : blocks) {
    if (failed(processBasicBlock(bb, lookupBlock(bb))))
      return failure();
  }

  return success();
}

LogicalResult ModuleImport::processBasicBlock(llvm::BasicBlock *bb,
                                              Block *block) {
  builder.setInsertionPointToStart(block);
  for (llvm::Instruction &inst : *bb) {
    if (failed(processInstruction(&inst)))
      return failure();

    // Set the non-debug metadata attributes on the imported operation and emit
    // a warning if an instruction other than a phi instruction is dropped
    // during the import.
    if (Operation *op = lookupOperation(&inst)) {
      setNonDebugMetadataAttrs(&inst, op);
    } else if (inst.getOpcode() != llvm::Instruction::PHI) {
      if (emitExpensiveWarnings) {
        Location loc = debugImporter->translateLoc(inst.getDebugLoc());
        emitWarning(loc) << "dropped instruction: " << diag(inst);
      }
    }
  }
  return success();
}

FailureOr<SmallVector<AccessGroupAttr>>
ModuleImport::lookupAccessGroupAttrs(const llvm::MDNode *node) const {
  return loopAnnotationImporter->lookupAccessGroupAttrs(node);
}

LoopAnnotationAttr
ModuleImport::translateLoopAnnotationAttr(const llvm::MDNode *node,
                                          Location loc) const {
  return loopAnnotationImporter->translateLoopAnnotation(node, loc);
}

OwningOpRef<ModuleOp>
mlir::translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                              MLIRContext *context,
                              bool emitExpensiveWarnings) {
  // Preload all registered dialects to allow the import to iterate the
  // registered LLVMImportDialectInterface implementations and query the
  // supported LLVM IR constructs before starting the translation. Assumes the
  // LLVM and DLTI dialects that convert the core LLVM IR constructs have been
  // registered before.
  assert(llvm::is_contained(context->getAvailableDialects(),
                            LLVMDialect::getDialectNamespace()));
  assert(llvm::is_contained(context->getAvailableDialects(),
                            DLTIDialect::getDialectNamespace()));
  context->loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module(ModuleOp::create(FileLineColLoc::get(
      StringAttr::get(context, llvmModule->getSourceFileName()), /*line=*/0,
      /*column=*/0)));

  ModuleImport moduleImport(module.get(), std::move(llvmModule),
                            emitExpensiveWarnings);
  if (failed(moduleImport.initializeImportInterface()))
    return {};
  if (failed(moduleImport.convertDataLayout()))
    return {};
  if (failed(moduleImport.convertComdats()))
    return {};
  if (failed(moduleImport.convertMetadata()))
    return {};
  if (failed(moduleImport.convertGlobals()))
    return {};
  if (failed(moduleImport.convertFunctions()))
    return {};

  return module;
}
