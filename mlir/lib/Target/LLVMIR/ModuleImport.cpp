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
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/LLVMIR/Import.h"

#include "AttrKindDetail.h"
#include "DebugImporter.h"
#include "LoopAnnotationImporter.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Target/LLVMIR/DataLayoutImporter.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/ModRef.h"
#include <optional>

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
  return str;
}

// Utility to print an LLVM metadata node as a string for passing
// to emitError(). The module argument is needed to print the nodes
// canonically numbered.
static std::string diagMD(const llvm::Metadata *node,
                          const llvm::Module *module) {
  std::string str;
  llvm::raw_string_ostream os(str);
  node->print(os, module, /*IsForDebug=*/true);
  return str;
}

/// Returns the name of the global_ctors global variables.
static constexpr StringRef getGlobalCtorsVarName() {
  return "llvm.global_ctors";
}

/// Prefix used for symbols of nameless llvm globals.
static constexpr StringRef getNamelessGlobalPrefix() {
  return "mlir.llvm.nameless_global";
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
                                            ModuleImport &moduleImport,
                                            LLVMImportInterface &iface) {
  // Copy the operands to an LLVM operands array reference for conversion.
  SmallVector<llvm::Value *> operands(inst->operands());
  ArrayRef<llvm::Value *> llvmOperands(operands);

  // Convert all instructions that provide an MLIR builder.
  if (iface.isConvertibleInstruction(inst->getOpcode()))
    return iface.convertInstruction(odsBuilder, inst, llvmOperands,
                                    moduleImport);
  // TODO: Implement the `convertInstruction` hooks in the
  // `LLVMDialectLLVMIRImportInterface` and move the following include there.
#include "mlir/Dialect/LLVMIR/LLVMOpFromLLVMIRConversions.inc"

  return failure();
}

/// Get a topologically sorted list of blocks for the given basic blocks.
static SetVector<llvm::BasicBlock *>
getTopologicallySortedBlocks(ArrayRef<llvm::BasicBlock *> basicBlocks) {
  SetVector<llvm::BasicBlock *> blocks;
  for (llvm::BasicBlock *basicBlock : basicBlocks) {
    if (!blocks.contains(basicBlock)) {
      llvm::ReversePostOrderTraversal<llvm::BasicBlock *> traversal(basicBlock);
      blocks.insert_range(traversal);
    }
  }
  assert(blocks.size() == basicBlocks.size() && "some blocks are not sorted");
  return blocks;
}

ModuleImport::ModuleImport(ModuleOp mlirModule,
                           std::unique_ptr<llvm::Module> llvmModule,
                           bool emitExpensiveWarnings,
                           bool importEmptyDICompositeTypes,
                           bool preferUnregisteredIntrinsics,
                           bool importStructsAsLiterals)
    : builder(mlirModule->getContext()), context(mlirModule->getContext()),
      mlirModule(mlirModule), llvmModule(std::move(llvmModule)),
      iface(mlirModule->getContext()),
      typeTranslator(*mlirModule->getContext(), importStructsAsLiterals),
      debugImporter(std::make_unique<DebugImporter>(
          mlirModule, importEmptyDICompositeTypes)),
      loopAnnotationImporter(
          std::make_unique<LoopAnnotationImporter>(*this, builder)),
      emitExpensiveWarnings(emitExpensiveWarnings),
      preferUnregisteredIntrinsics(preferUnregisteredIntrinsics) {
  builder.setInsertionPointToStart(mlirModule.getBody());
}

ComdatOp ModuleImport::getGlobalComdatOp() {
  if (globalComdatOp)
    return globalComdatOp;

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());
  globalComdatOp =
      ComdatOp::create(builder, mlirModule.getLoc(), getGlobalComdatOpName());
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
      // the TBAARootAttr on the first walk.
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
  auto verifySelfRefOrString = [](const llvm::MDNode *node) {
    return node->getNumOperands() != 0 &&
           (node == dyn_cast<llvm::MDNode>(node->getOperand(0)) ||
            isa<llvm::MDString>(node->getOperand(0)));
  };
  // Helper that verifies the given operand is a string or does not exist.
  auto verifyDescription = [](const llvm::MDNode *node, unsigned idx) {
    return idx >= node->getNumOperands() ||
           isa<llvm::MDString>(node->getOperand(idx));
  };

  auto getIdAttr = [&](const llvm::MDNode *node) -> Attribute {
    if (verifySelfRef(node))
      return DistinctAttr::create(builder.getUnitAttr());

    auto name = cast<llvm::MDString>(node->getOperand(0));
    return builder.getStringAttr(name->getString());
  };

  // Helper that creates an alias scope domain attribute.
  auto createAliasScopeDomainOp = [&](const llvm::MDNode *aliasDomain) {
    StringAttr description = nullptr;
    if (aliasDomain->getNumOperands() >= 2)
      if (auto *operand = dyn_cast<llvm::MDString>(aliasDomain->getOperand(1)))
        description = builder.getStringAttr(operand->getString());
    Attribute idAttr = getIdAttr(aliasDomain);
    return builder.getAttr<AliasScopeDomainAttr>(idAttr, description);
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
      if (!verifySelfRefOrString(scope) || !domain ||
          !verifyDescription(scope, 2))
        return emitError(loc) << "unsupported alias scope node: "
                              << diagMD(scope, llvmModule.get());
      if (!verifySelfRefOrString(domain) || !verifyDescription(domain, 1))
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
      Attribute idAttr = getIdAttr(scope);
      auto aliasScopeOp = builder.getAttr<AliasScopeAttr>(
          idAttr, cast<AliasScopeDomainAttr>(it->second), description);

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

void ModuleImport::addDebugIntrinsic(llvm::CallInst *intrinsic) {
  debugIntrinsics.insert(intrinsic);
}

static Attribute convertCGProfileModuleFlagValue(ModuleOp mlirModule,
                                                 llvm::MDTuple *mdTuple) {
  auto getLLVMFunction =
      [&](const llvm::MDOperand &funcMDO) -> llvm::Function * {
    auto *f = cast_or_null<llvm::ValueAsMetadata>(funcMDO);
    // nullptr is a valid value for the function pointer.
    if (!f)
      return nullptr;
    auto *llvmFn = cast<llvm::Function>(f->getValue()->stripPointerCasts());
    return llvmFn;
  };

  // Each tuple element becomes one ModuleFlagCGProfileEntryAttr.
  SmallVector<Attribute> cgProfile;
  for (unsigned i = 0; i < mdTuple->getNumOperands(); i++) {
    const llvm::MDOperand &mdo = mdTuple->getOperand(i);
    auto *cgEntry = cast<llvm::MDNode>(mdo);
    llvm::Constant *llvmConstant =
        cast<llvm::ConstantAsMetadata>(cgEntry->getOperand(2))->getValue();
    uint64_t count = cast<llvm::ConstantInt>(llvmConstant)->getZExtValue();
    auto *fromFn = getLLVMFunction(cgEntry->getOperand(0));
    auto *toFn = getLLVMFunction(cgEntry->getOperand(1));
    // FlatSymbolRefAttr::get(mlirModule->getContext(), llvmFn->getName());
    cgProfile.push_back(ModuleFlagCGProfileEntryAttr::get(
        mlirModule->getContext(),
        fromFn ? FlatSymbolRefAttr::get(mlirModule->getContext(),
                                        fromFn->getName())
               : nullptr,
        toFn ? FlatSymbolRefAttr::get(mlirModule->getContext(), toFn->getName())
             : nullptr,
        count));
  }
  return ArrayAttr::get(mlirModule->getContext(), cgProfile);
}

/// Extract a two element `MDTuple` from a `MDOperand`. Emit a warning in case
/// something else is found.
static llvm::MDTuple *getTwoElementMDTuple(ModuleOp mlirModule,
                                           const llvm::Module *llvmModule,
                                           const llvm::MDOperand &md) {
  auto *tupleEntry = dyn_cast_or_null<llvm::MDTuple>(md);
  if (!tupleEntry || tupleEntry->getNumOperands() != 2)
    emitWarning(mlirModule.getLoc())
        << "expected 2-element tuple metadata: " << diagMD(md, llvmModule);
  return tupleEntry;
}

/// Extract a constant metadata value from a two element tuple (<key, value>).
/// Return nullptr if requirements are not met. A warning is emitted if the
/// `matchKey` is different from the tuple's key.
static llvm::ConstantAsMetadata *getConstantMDFromKeyValueTuple(
    ModuleOp mlirModule, const llvm::Module *llvmModule,
    const llvm::MDOperand &md, StringRef matchKey, bool optional = false) {
  llvm::MDTuple *tupleEntry = getTwoElementMDTuple(mlirModule, llvmModule, md);
  if (!tupleEntry)
    return nullptr;
  auto *keyMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(0));
  if (!keyMD || keyMD->getString() != matchKey) {
    if (!optional)
      emitWarning(mlirModule.getLoc())
          << "expected '" << matchKey << "' key, but found: "
          << diagMD(tupleEntry->getOperand(0), llvmModule);
    return nullptr;
  }

  return dyn_cast<llvm::ConstantAsMetadata>(tupleEntry->getOperand(1));
}

/// Extract an integer value from a two element tuple (<key, value>).
/// Fail if requirements are not met. A warning is emitted if the
/// found value isn't a LLVM constant integer.
static FailureOr<uint64_t>
convertInt64FromKeyValueTuple(ModuleOp mlirModule,
                              const llvm::Module *llvmModule,
                              const llvm::MDOperand &md, StringRef matchKey) {
  llvm::ConstantAsMetadata *valMD =
      getConstantMDFromKeyValueTuple(mlirModule, llvmModule, md, matchKey);
  if (!valMD)
    return failure();

  if (auto *cstInt = dyn_cast<llvm::ConstantInt>(valMD->getValue()))
    return cstInt->getZExtValue();

  emitWarning(mlirModule.getLoc())
      << "expected integer metadata value for key '" << matchKey
      << "': " << diagMD(md, llvmModule);
  return failure();
}

static std::optional<ProfileSummaryFormatKind>
convertProfileSummaryFormat(ModuleOp mlirModule, const llvm::Module *llvmModule,
                            const llvm::MDOperand &formatMD) {
  auto *tupleEntry = getTwoElementMDTuple(mlirModule, llvmModule, formatMD);
  if (!tupleEntry)
    return std::nullopt;

  llvm::MDString *keyMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(0));
  if (!keyMD || keyMD->getString() != "ProfileFormat") {
    emitWarning(mlirModule.getLoc())
        << "expected 'ProfileFormat' key: "
        << diagMD(tupleEntry->getOperand(0), llvmModule);
    return std::nullopt;
  }

  llvm::MDString *valMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(1));
  std::optional<ProfileSummaryFormatKind> fmtKind =
      symbolizeProfileSummaryFormatKind(valMD->getString());
  if (!fmtKind) {
    emitWarning(mlirModule.getLoc())
        << "expected 'SampleProfile', 'InstrProf' or 'CSInstrProf' values, "
           "but found: "
        << diagMD(valMD, llvmModule);
    return std::nullopt;
  }

  return fmtKind;
}

static FailureOr<SmallVector<ModuleFlagProfileSummaryDetailedAttr>>
convertProfileSummaryDetailed(ModuleOp mlirModule,
                              const llvm::Module *llvmModule,
                              const llvm::MDOperand &summaryMD) {
  auto *tupleEntry = getTwoElementMDTuple(mlirModule, llvmModule, summaryMD);
  if (!tupleEntry)
    return failure();

  llvm::MDString *keyMD = dyn_cast<llvm::MDString>(tupleEntry->getOperand(0));
  if (!keyMD || keyMD->getString() != "DetailedSummary") {
    emitWarning(mlirModule.getLoc())
        << "expected 'DetailedSummary' key: "
        << diagMD(tupleEntry->getOperand(0), llvmModule);
    return failure();
  }

  llvm::MDTuple *entriesMD = dyn_cast<llvm::MDTuple>(tupleEntry->getOperand(1));
  if (!entriesMD) {
    emitWarning(mlirModule.getLoc())
        << "expected tuple value for 'DetailedSummary' key: "
        << diagMD(tupleEntry->getOperand(1), llvmModule);
    return failure();
  }

  SmallVector<ModuleFlagProfileSummaryDetailedAttr> detailedSummary;
  for (auto &&entry : entriesMD->operands()) {
    llvm::MDTuple *entryMD = dyn_cast<llvm::MDTuple>(entry);
    if (!entryMD || entryMD->getNumOperands() != 3) {
      emitWarning(mlirModule.getLoc())
          << "'DetailedSummary' entry expects 3 operands: "
          << diagMD(entry, llvmModule);
      return failure();
    }

    auto *op0 = dyn_cast<llvm::ConstantAsMetadata>(entryMD->getOperand(0));
    auto *op1 = dyn_cast<llvm::ConstantAsMetadata>(entryMD->getOperand(1));
    auto *op2 = dyn_cast<llvm::ConstantAsMetadata>(entryMD->getOperand(2));
    if (!op0 || !op1 || !op2) {
      emitWarning(mlirModule.getLoc())
          << "expected only integer entries in 'DetailedSummary': "
          << diagMD(entry, llvmModule);
      return failure();
    }

    auto detaildSummaryEntry = ModuleFlagProfileSummaryDetailedAttr::get(
        mlirModule->getContext(),
        cast<llvm::ConstantInt>(op0->getValue())->getZExtValue(),
        cast<llvm::ConstantInt>(op1->getValue())->getZExtValue(),
        cast<llvm::ConstantInt>(op2->getValue())->getZExtValue());
    detailedSummary.push_back(detaildSummaryEntry);
  }
  return detailedSummary;
}

static Attribute
convertProfileSummaryModuleFlagValue(ModuleOp mlirModule,
                                     const llvm::Module *llvmModule,
                                     llvm::MDTuple *mdTuple) {
  unsigned profileNumEntries = mdTuple->getNumOperands();
  if (profileNumEntries < 8) {
    emitWarning(mlirModule.getLoc())
        << "expected at 8 entries in 'ProfileSummary': "
        << diagMD(mdTuple, llvmModule);
    return nullptr;
  }

  unsigned summayIdx = 0;
  auto checkOptionalPosition = [&](const llvm::MDOperand &md,
                                   StringRef matchKey) -> LogicalResult {
    // Make sure we won't step over the bound of the array of summary entries.
    // Since (non-optional) DetailedSummary always comes last, the next entry in
    // the tuple operand array must exist.
    if (summayIdx + 1 >= profileNumEntries) {
      emitWarning(mlirModule.getLoc())
          << "the last summary entry is '" << matchKey
          << "', expected 'DetailedSummary': " << diagMD(md, llvmModule);
      return failure();
    }

    return success();
  };

  auto getOptIntValue =
      [&](const llvm::MDOperand &md,
          StringRef matchKey) -> FailureOr<std::optional<uint64_t>> {
    if (!getConstantMDFromKeyValueTuple(mlirModule, llvmModule, md, matchKey,
                                        /*optional=*/true))
      return FailureOr<std::optional<uint64_t>>(std::nullopt);
    if (checkOptionalPosition(md, matchKey).failed())
      return failure();
    FailureOr<uint64_t> val =
        convertInt64FromKeyValueTuple(mlirModule, llvmModule, md, matchKey);
    if (failed(val))
      return failure();
    return val;
  };

  auto getOptDoubleValue = [&](const llvm::MDOperand &md,
                               StringRef matchKey) -> FailureOr<FloatAttr> {
    auto *valMD = getConstantMDFromKeyValueTuple(mlirModule, llvmModule, md,
                                                 matchKey, /*optional=*/true);
    if (!valMD)
      return FloatAttr{};
    if (auto *cstFP = dyn_cast<llvm::ConstantFP>(valMD->getValue())) {
      if (checkOptionalPosition(md, matchKey).failed())
        return failure();
      return FloatAttr::get(Float64Type::get(mlirModule.getContext()),
                            cstFP->getValueAPF());
    }
    emitWarning(mlirModule.getLoc())
        << "expected double metadata value for key '" << matchKey
        << "': " << diagMD(md, llvmModule);
    return failure();
  };

  // Build ModuleFlagProfileSummaryAttr by sequentially fetching elements in
  // a fixed order: format, total count, etc.
  std::optional<ProfileSummaryFormatKind> format = convertProfileSummaryFormat(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++));
  if (!format.has_value())
    return nullptr;

  FailureOr<uint64_t> totalCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "TotalCount");
  if (failed(totalCount))
    return nullptr;

  FailureOr<uint64_t> maxCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "MaxCount");
  if (failed(maxCount))
    return nullptr;

  FailureOr<uint64_t> maxInternalCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++),
      "MaxInternalCount");
  if (failed(maxInternalCount))
    return nullptr;

  FailureOr<uint64_t> maxFunctionCount = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++),
      "MaxFunctionCount");
  if (failed(maxFunctionCount))
    return nullptr;

  FailureOr<uint64_t> numCounts = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "NumCounts");
  if (failed(numCounts))
    return nullptr;

  FailureOr<uint64_t> numFunctions = convertInt64FromKeyValueTuple(
      mlirModule, llvmModule, mdTuple->getOperand(summayIdx++), "NumFunctions");
  if (failed(numFunctions))
    return nullptr;

  // Handle optional keys.
  FailureOr<std::optional<uint64_t>> isPartialProfile =
      getOptIntValue(mdTuple->getOperand(summayIdx), "IsPartialProfile");
  if (failed(isPartialProfile))
    return nullptr;
  if (isPartialProfile->has_value())
    summayIdx++;

  FailureOr<FloatAttr> partialProfileRatio =
      getOptDoubleValue(mdTuple->getOperand(summayIdx), "PartialProfileRatio");
  if (failed(partialProfileRatio))
    return nullptr;
  if (*partialProfileRatio)
    summayIdx++;

  // Handle detailed summary.
  FailureOr<SmallVector<ModuleFlagProfileSummaryDetailedAttr>> detailed =
      convertProfileSummaryDetailed(mlirModule, llvmModule,
                                    mdTuple->getOperand(summayIdx));
  if (failed(detailed))
    return nullptr;

  // Build the final profile summary attribute.
  return ModuleFlagProfileSummaryAttr::get(
      mlirModule->getContext(), *format, *totalCount, *maxCount,
      *maxInternalCount, *maxFunctionCount, *numCounts, *numFunctions,
      *isPartialProfile, *partialProfileRatio, *detailed);
}

/// Invoke specific handlers for each known module flag value, returns nullptr
/// if the key is unknown or unimplemented.
static Attribute
convertModuleFlagValueFromMDTuple(ModuleOp mlirModule,
                                  const llvm::Module *llvmModule, StringRef key,
                                  llvm::MDTuple *mdTuple) {
  if (key == LLVMDialect::getModuleFlagKeyCGProfileName())
    return convertCGProfileModuleFlagValue(mlirModule, mdTuple);
  if (key == LLVMDialect::getModuleFlagKeyProfileSummaryName())
    return convertProfileSummaryModuleFlagValue(mlirModule, llvmModule,
                                                mdTuple);
  return nullptr;
}

LogicalResult ModuleImport::convertModuleFlagsMetadata() {
  SmallVector<llvm::Module::ModuleFlagEntry> llvmModuleFlags;
  llvmModule->getModuleFlagsMetadata(llvmModuleFlags);

  SmallVector<Attribute> moduleFlags;
  for (const auto [behavior, key, val] : llvmModuleFlags) {
    Attribute valAttr = nullptr;
    if (auto *constInt = llvm::mdconst::dyn_extract<llvm::ConstantInt>(val)) {
      valAttr = builder.getI32IntegerAttr(constInt->getZExtValue());
    } else if (auto *mdString = dyn_cast<llvm::MDString>(val)) {
      valAttr = builder.getStringAttr(mdString->getString());
    } else if (auto *mdTuple = dyn_cast<llvm::MDTuple>(val)) {
      valAttr = convertModuleFlagValueFromMDTuple(mlirModule, llvmModule.get(),
                                                  key->getString(), mdTuple);
    }

    if (!valAttr) {
      emitWarning(mlirModule.getLoc())
          << "unsupported module flag value for key '" << key->getString()
          << "' : " << diagMD(val, llvmModule.get());
      continue;
    }

    moduleFlags.push_back(builder.getAttr<ModuleFlagAttr>(
        convertModFlagBehaviorFromLLVM(behavior),
        builder.getStringAttr(key->getString()), valAttr));
  }

  if (!moduleFlags.empty())
    LLVM::ModuleFlagsOp::create(builder, mlirModule.getLoc(),
                                builder.getArrayAttr(moduleFlags));

  return success();
}

LogicalResult ModuleImport::convertLinkerOptionsMetadata() {
  for (const llvm::NamedMDNode &named : llvmModule->named_metadata()) {
    if (named.getName() != "llvm.linker.options")
      continue;
    // llvm.linker.options operands are lists of strings.
    for (const llvm::MDNode *node : named.operands()) {
      SmallVector<StringRef> options;
      options.reserve(node->getNumOperands());
      for (const llvm::MDOperand &option : node->operands())
        options.push_back(cast<llvm::MDString>(option)->getString());
      LLVM::LinkerOptionsOp::create(builder, mlirModule.getLoc(),
                                    builder.getStrArrayAttr(options));
    }
  }
  return success();
}

LogicalResult ModuleImport::convertDependentLibrariesMetadata() {
  for (const llvm::NamedMDNode &named : llvmModule->named_metadata()) {
    if (named.getName() != "llvm.dependent-libraries")
      continue;
    SmallVector<StringRef> libraries;
    for (const llvm::MDNode *node : named.operands()) {
      if (node->getNumOperands() == 1)
        if (auto *mdString = dyn_cast<llvm::MDString>(node->getOperand(0)))
          libraries.push_back(mdString->getString());
    }
    if (!libraries.empty())
      mlirModule->setAttr(LLVM::LLVMDialect::getDependentLibrariesAttrName(),
                          builder.getStrArrayAttr(libraries));
  }
  return success();
}

LogicalResult ModuleImport::convertIdentMetadata() {
  for (const llvm::NamedMDNode &named : llvmModule->named_metadata()) {
    // llvm.ident should have a single operand. That operand is itself an
    // MDNode with a single string operand.
    if (named.getName() != LLVMDialect::getIdentAttrName())
      continue;

    if (named.getNumOperands() == 1)
      if (auto *md = dyn_cast<llvm::MDNode>(named.getOperand(0)))
        if (md->getNumOperands() == 1)
          if (auto *mdStr = dyn_cast<llvm::MDString>(md->getOperand(0)))
            mlirModule->setAttr(LLVMDialect::getIdentAttrName(),
                                builder.getStringAttr(mdStr->getString()));
  }
  return success();
}

LogicalResult ModuleImport::convertCommandlineMetadata() {
  for (const llvm::NamedMDNode &nmd : llvmModule->named_metadata()) {
    // llvm.commandline should have a single operand. That operand is itself an
    // MDNode with a single string operand.
    if (nmd.getName() != LLVMDialect::getCommandlineAttrName())
      continue;

    if (nmd.getNumOperands() == 1)
      if (auto *md = dyn_cast<llvm::MDNode>(nmd.getOperand(0)))
        if (md->getNumOperands() == 1)
          if (auto *mdStr = dyn_cast<llvm::MDString>(md->getOperand(0)))
            mlirModule->setAttr(LLVMDialect::getCommandlineAttrName(),
                                builder.getStringAttr(mdStr->getString()));
  }
  return success();
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
  if (failed(convertLinkerOptionsMetadata()))
    return failure();
  if (failed(convertDependentLibrariesMetadata()))
    return failure();
  if (failed(convertModuleFlagsMetadata()))
    return failure();
  if (failed(convertIdentMetadata()))
    return failure();
  if (failed(convertCommandlineMetadata()))
    return failure();
  return success();
}

void ModuleImport::processComdat(const llvm::Comdat *comdat) {
  if (comdatMapping.contains(comdat))
    return;

  ComdatOp comdatOp = getGlobalComdatOp();
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&comdatOp.getBody().back());
  auto selectorOp = ComdatSelectorOp::create(
      builder, mlirModule.getLoc(), comdat->getName(),
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

LogicalResult ModuleImport::convertAliases() {
  for (llvm::GlobalAlias &alias : llvmModule->aliases()) {
    if (failed(convertAlias(&alias))) {
      return emitError(UnknownLoc::get(context))
             << "unhandled global alias: " << diag(alias);
    }
  }
  return success();
}

LogicalResult ModuleImport::convertIFuncs() {
  for (llvm::GlobalIFunc &ifunc : llvmModule->ifuncs()) {
    if (failed(convertIFunc(&ifunc))) {
      return emitError(UnknownLoc::get(context))
             << "unhandled global ifunc: " << diag(ifunc);
    }
  }
  return success();
}

LogicalResult ModuleImport::convertDataLayout() {
  Location loc = mlirModule.getLoc();
  DataLayoutImporter dataLayoutImporter(
      context, llvmModule->getDataLayout().getStringRepresentation());
  if (!dataLayoutImporter.getDataLayoutSpec())
    return emitError(loc, "cannot translate data layout: ")
           << dataLayoutImporter.getLastToken();

  for (StringRef token : dataLayoutImporter.getUnhandledTokens())
    emitWarning(loc, "unhandled data layout token: ") << token;

  mlirModule->setAttr(DLTIDialect::kDataLayoutAttrName,
                      dataLayoutImporter.getDataLayoutSpec());
  return success();
}

void ModuleImport::convertTargetTriple() {
  mlirModule->setAttr(
      LLVM::LLVMDialect::getTargetTripleAttrName(),
      builder.getStringAttr(llvmModule->getTargetTriple().str()));
}

void ModuleImport::convertModuleLevelAsm() {
  llvm::StringRef asmStr = llvmModule->getModuleInlineAsm();
  llvm::SmallVector<mlir::Attribute> asmArrayAttr;

  for (llvm::StringRef line : llvm::split(asmStr, '\n'))
    if (!line.empty())
      asmArrayAttr.push_back(builder.getStringAttr(line));

  mlirModule->setAttr(LLVM::LLVMDialect::getModuleLevelAsmAttrName(),
                      builder.getArrayAttr(asmArrayAttr));
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

void ModuleImport::setIntegerOverflowFlags(llvm::Instruction *inst,
                                           Operation *op) const {
  auto iface = cast<IntegerOverflowFlagsInterface>(op);

  IntegerOverflowFlags value = {};
  value = bitEnumSet(value, IntegerOverflowFlags::nsw, inst->hasNoSignedWrap());
  value =
      bitEnumSet(value, IntegerOverflowFlags::nuw, inst->hasNoUnsignedWrap());

  iface.setOverflowFlags(value);
}

void ModuleImport::setExactFlag(llvm::Instruction *inst, Operation *op) const {
  auto iface = cast<ExactFlagInterface>(op);

  iface.setIsExact(inst->isExact());
}

void ModuleImport::setDisjointFlag(llvm::Instruction *inst,
                                   Operation *op) const {
  auto iface = cast<DisjointFlagInterface>(op);
  auto instDisjoint = cast<llvm::PossiblyDisjointInst>(inst);

  iface.setIsDisjoint(instDisjoint->isDisjoint());
}

void ModuleImport::setNonNegFlag(llvm::Instruction *inst, Operation *op) const {
  auto iface = cast<NonNegFlagInterface>(op);

  iface.setNonNeg(inst->hasNonNeg());
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
  Type elementType = cast<VectorType>(type).getElementType();
  if (!elementType.isIntOrFloat())
    return {};

  SmallVector<int64_t> shape(arrayShape);
  shape.push_back(numElements.getKnownMinValue());
  return VectorType::get(shape, elementType);
}

Type ModuleImport::getBuiltinTypeForAttr(Type type) {
  if (!type)
    return {};

  // Return builtin integer and floating-point types as is.
  if (type.isIntOrFloat())
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
  if (type.isIntOrFloat())
    return RankedTensorType::get(arrayShape, type);
  return getVectorTypeForAttr(type, arrayShape);
}

/// Returns an integer or float attribute for the provided scalar constant
/// `constScalar` or nullptr if the conversion fails.
static TypedAttr getScalarConstantAsAttr(OpBuilder &builder,
                                         llvm::Constant *constScalar) {
  MLIRContext *context = builder.getContext();

  // Convert scalar intergers.
  if (auto *constInt = dyn_cast<llvm::ConstantInt>(constScalar)) {
    return builder.getIntegerAttr(
        IntegerType::get(context, constInt->getBitWidth()),
        constInt->getValue());
  }

  // Convert scalar floats.
  if (auto *constFloat = dyn_cast<llvm::ConstantFP>(constScalar)) {
    llvm::Type *type = constFloat->getType();
    FloatType floatType =
        type->isBFloatTy()
            ? BFloat16Type::get(context)
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

FlatSymbolRefAttr
ModuleImport::getOrCreateNamelessSymbolName(llvm::GlobalVariable *globalVar) {
  assert(globalVar->getName().empty() &&
         "expected to work with a nameless global");
  auto [it, success] = namelessGlobals.try_emplace(globalVar);
  if (!success)
    return it->second;

  // Make sure the symbol name does not clash with an existing symbol.
  SmallString<128> globalName = SymbolTable::generateSymbolName<128>(
      getNamelessGlobalPrefix(),
      [this](StringRef newName) { return llvmModule->getNamedValue(newName); },
      namelessGlobalId);
  auto symbolRef = FlatSymbolRefAttr::get(context, globalName);
  it->getSecond() = symbolRef;
  return symbolRef;
}

OpBuilder::InsertionGuard ModuleImport::setGlobalInsertionPoint() {
  OpBuilder::InsertionGuard guard(builder);
  if (globalInsertionOp)
    builder.setInsertionPointAfter(globalInsertionOp);
  else
    builder.setInsertionPointToStart(mlirModule.getBody());
  return guard;
}

LogicalResult ModuleImport::convertAlias(llvm::GlobalAlias *alias) {
  // Insert the alias after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  Type type = convertType(alias->getValueType());
  AliasOp aliasOp = AliasOp::create(builder, mlirModule.getLoc(), type,
                                    convertLinkageFromLLVM(alias->getLinkage()),
                                    alias->getName(),
                                    /*dso_local=*/alias->isDSOLocal(),
                                    /*thread_local=*/alias->isThreadLocal(),
                                    /*attrs=*/ArrayRef<NamedAttribute>());
  globalInsertionOp = aliasOp;

  clearRegionState();
  Block *block = builder.createBlock(&aliasOp.getInitializerRegion());
  setConstantInsertionPointToStart(block);
  FailureOr<Value> initializer = convertConstantExpr(alias->getAliasee());
  if (failed(initializer))
    return failure();
  ReturnOp::create(builder, aliasOp.getLoc(), *initializer);

  if (alias->hasAtLeastLocalUnnamedAddr())
    aliasOp.setUnnamedAddr(convertUnnamedAddrFromLLVM(alias->getUnnamedAddr()));
  aliasOp.setVisibility_(convertVisibilityFromLLVM(alias->getVisibility()));

  return success();
}

LogicalResult ModuleImport::convertIFunc(llvm::GlobalIFunc *ifunc) {
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  Type type = convertType(ifunc->getValueType());
  llvm::Constant *resolver = ifunc->getResolver();
  Type resolverType = convertType(resolver->getType());
  IFuncOp::create(builder, mlirModule.getLoc(), ifunc->getName(), type,
                  resolver->getName(), resolverType,
                  convertLinkageFromLLVM(ifunc->getLinkage()),
                  ifunc->isDSOLocal(), ifunc->getAddressSpace(),
                  convertUnnamedAddrFromLLVM(ifunc->getUnnamedAddr()),
                  convertVisibilityFromLLVM(ifunc->getVisibility()));
  return success();
}

LogicalResult ModuleImport::convertGlobal(llvm::GlobalVariable *globalVar) {
  // Insert the global after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

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

  // Get the global expression associated with this global variable and convert
  // it.
  SmallVector<Attribute> globalExpressionAttrs;
  SmallVector<llvm::DIGlobalVariableExpression *> globalExpressions;
  globalVar->getDebugInfo(globalExpressions);

  for (llvm::DIGlobalVariableExpression *expr : globalExpressions) {
    DIGlobalVariableExpressionAttr globalExpressionAttr =
        debugImporter->translateGlobalVariableExpression(expr);
    globalExpressionAttrs.push_back(globalExpressionAttr);
  }

  // Workaround to support LLVM's nameless globals. MLIR, in contrast to LLVM,
  // always requires a symbol name.
  StringRef globalName = globalVar->getName();
  if (globalName.empty())
    globalName = getOrCreateNamelessSymbolName(globalVar).getValue();

  GlobalOp globalOp = GlobalOp::create(
      builder, mlirModule.getLoc(), type, globalVar->isConstant(),
      convertLinkageFromLLVM(globalVar->getLinkage()), StringRef(globalName),
      valueAttr, alignment, /*addr_space=*/globalVar->getAddressSpace(),
      /*dso_local=*/globalVar->isDSOLocal(),
      /*thread_local=*/globalVar->isThreadLocal(), /*comdat=*/SymbolRefAttr(),
      /*attrs=*/ArrayRef<NamedAttribute>(), /*dbgExprs=*/globalExpressionAttrs);
  globalInsertionOp = globalOp;

  if (globalVar->hasInitializer() && !valueAttr) {
    clearRegionState();
    Block *block = builder.createBlock(&globalOp.getInitializerRegion());
    setConstantInsertionPointToStart(block);
    FailureOr<Value> initializer =
        convertConstantExpr(globalVar->getInitializer());
    if (failed(initializer))
      return failure();
    ReturnOp::create(builder, globalOp.getLoc(), *initializer);
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
  llvm::Constant *initializer = globalVar->getInitializer();

  bool knownInit = isa<llvm::ConstantArray>(initializer) ||
                   isa<llvm::ConstantAggregateZero>(initializer);
  if (!knownInit)
    return failure();

  // ConstantAggregateZero does not engage with the operand initialization
  // in the loop that follows - there should be no operands. This implies
  // empty ctor/dtor lists.
  if (auto *caz = dyn_cast<llvm::ConstantAggregateZero>(initializer)) {
    if (caz->getElementCount().getFixedValue() != 0)
      return failure();
  }

  SmallVector<Attribute> funcs;
  SmallVector<int32_t> priorities;
  SmallVector<Attribute> dataList;
  for (llvm::Value *operand : initializer->operands()) {
    auto *aggregate = dyn_cast<llvm::ConstantAggregate>(operand);
    if (!aggregate || aggregate->getNumOperands() != 3)
      return failure();

    auto *priority = dyn_cast<llvm::ConstantInt>(aggregate->getOperand(0));
    auto *func = dyn_cast<llvm::Function>(aggregate->getOperand(1));
    auto *data = dyn_cast<llvm::Constant>(aggregate->getOperand(2));
    if (!priority || !func || !data)
      return failure();

    auto *gv = dyn_cast_or_null<llvm::GlobalValue>(data);
    Attribute dataAttr;
    if (gv)
      dataAttr = FlatSymbolRefAttr::get(context, gv->getName());
    else if (data->isNullValue())
      dataAttr = ZeroAttr::get(context);
    else
      return failure();

    funcs.push_back(FlatSymbolRefAttr::get(context, func->getName()));
    priorities.push_back(priority->getValue().getZExtValue());
    dataList.push_back(dataAttr);
  }

  // Insert the global after the last one or at the start of the module.
  OpBuilder::InsertionGuard guard = setGlobalInsertionPoint();

  if (globalVar->getName() == getGlobalCtorsVarName()) {
    globalInsertionOp = LLVM::GlobalCtorsOp::create(
        builder, mlirModule.getLoc(), builder.getArrayAttr(funcs),
        builder.getI32ArrayAttr(priorities), builder.getArrayAttr(dataList));
    return success();
  }
  globalInsertionOp = LLVM::GlobalDtorsOp::create(
      builder, mlirModule.getLoc(), builder.getArrayAttr(funcs),
      builder.getI32ArrayAttr(priorities), builder.getArrayAttr(dataList));
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
    // References of global objects are just pointers to the object. Avoid
    // walking the elements of these here.
    if (isa<llvm::GlobalObject>(current) || isa<llvm::GlobalAlias>(current)) {
      orderedSet.insert(current);
      workList.pop_back();
      continue;
    }

    // Collect all dependencies of the current constant and add them to the
    // adjacency list if none has been computed before.
    auto [adjacencyIt, inserted] = adjacencyLists.try_emplace(current);
    if (inserted) {
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
      return AddressOfOp::create(builder, loc, type, symbolRef.getValue())
          .getResult();
    }
    return ConstantOp::create(builder, loc, type, attr).getResult();
  }

  // Convert null pointer constants.
  if (auto *nullPtr = dyn_cast<llvm::ConstantPointerNull>(constant)) {
    Type type = convertType(nullPtr->getType());
    return ZeroOp::create(builder, loc, type).getResult();
  }

  // Convert none token constants.
  if (isa<llvm::ConstantTokenNone>(constant)) {
    return NoneTokenOp::create(builder, loc).getResult();
  }

  // Convert poison.
  if (auto *poisonVal = dyn_cast<llvm::PoisonValue>(constant)) {
    Type type = convertType(poisonVal->getType());
    return PoisonOp::create(builder, loc, type).getResult();
  }

  // Convert undef.
  if (auto *undefVal = dyn_cast<llvm::UndefValue>(constant)) {
    Type type = convertType(undefVal->getType());
    return UndefOp::create(builder, loc, type).getResult();
  }

  // Convert dso_local_equivalent.
  if (auto *dsoLocalEquivalent = dyn_cast<llvm::DSOLocalEquivalent>(constant)) {
    Type type = convertType(dsoLocalEquivalent->getType());
    return DSOLocalEquivalentOp::create(
               builder, loc, type,
               FlatSymbolRefAttr::get(
                   builder.getContext(),
                   dsoLocalEquivalent->getGlobalValue()->getName()))
        .getResult();
  }

  // Convert global variable accesses.
  if (auto *globalObj = dyn_cast<llvm::GlobalObject>(constant)) {
    Type type = convertType(globalObj->getType());
    StringRef globalName = globalObj->getName();
    FlatSymbolRefAttr symbolRef;
    // Empty names are only allowed for global variables.
    if (globalName.empty())
      symbolRef =
          getOrCreateNamelessSymbolName(cast<llvm::GlobalVariable>(globalObj));
    else
      symbolRef = FlatSymbolRefAttr::get(context, globalName);
    return AddressOfOp::create(builder, loc, type, symbolRef).getResult();
  }

  // Convert global alias accesses.
  if (auto *globalAliasObj = dyn_cast<llvm::GlobalAlias>(constant)) {
    Type type = convertType(globalAliasObj->getType());
    StringRef aliaseeName = globalAliasObj->getName();
    FlatSymbolRefAttr symbolRef = FlatSymbolRefAttr::get(context, aliaseeName);
    return AddressOfOp::create(builder, loc, type, symbolRef).getResult();
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
    Value root = UndefOp::create(builder, loc, rootType);
    for (const auto &it : llvm::enumerate(elementValues)) {
      if (isArrayOrStruct) {
        root =
            InsertValueOp::create(builder, loc, root, it.value(), it.index());
      } else {
        Attribute indexAttr = builder.getI32IntegerAttr(it.index());
        Value indexValue =
            ConstantOp::create(builder, loc, builder.getI32Type(), indexAttr);
        root = InsertElementOp::create(builder, loc, rootType, root, it.value(),
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
    // Create llvm.mlir.zero operation to represent zero-initialization of
    // target extension type.
    return LLVM::ZeroOp::create(builder, loc, targetExtType).getRes();
  }

  if (auto *blockAddr = dyn_cast<llvm::BlockAddress>(constant)) {
    auto fnSym =
        FlatSymbolRefAttr::get(context, blockAddr->getFunction()->getName());
    auto blockTag =
        BlockTagAttr::get(context, blockAddr->getBasicBlock()->getNumber());
    return BlockAddressOp::create(
               builder, loc, convertType(blockAddr->getType()),
               BlockAddressAttr::get(context, fnSym, blockTag))
        .getRes();
  }

  StringRef error = "";

  if (isa<llvm::ConstantPtrAuth>(constant))
    error = " since ptrauth(...) is unsupported";

  if (isa<llvm::NoCFIValue>(constant))
    error = " since no_cfi is unsupported";

  if (isa<llvm::GlobalValue>(constant))
    error = " since global value is unsupported";

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

LogicalResult ModuleImport::convertIntrinsicArguments(
    ArrayRef<llvm::Value *> values, ArrayRef<llvm::OperandBundleUse> opBundles,
    bool requiresOpBundles, ArrayRef<unsigned> immArgPositions,
    ArrayRef<StringLiteral> immArgAttrNames, SmallVectorImpl<Value> &valuesOut,
    SmallVectorImpl<NamedAttribute> &attrsOut) {
  assert(immArgPositions.size() == immArgAttrNames.size() &&
         "LLVM `immArgPositions` and MLIR `immArgAttrNames` should have equal "
         "length");

  SmallVector<llvm::Value *> operands(values);
  for (auto [immArgPos, immArgName] :
       llvm::zip(immArgPositions, immArgAttrNames)) {
    auto &value = operands[immArgPos];
    auto *constant = llvm::cast<llvm::Constant>(value);
    auto attr = getScalarConstantAsAttr(builder, constant);
    assert(attr && attr.getType().isIntOrFloat() &&
           "expected immarg to be float or integer constant");
    auto nameAttr = StringAttr::get(attr.getContext(), immArgName);
    attrsOut.push_back({nameAttr, attr});
    // Mark matched attribute values as null (so they can be removed below).
    value = nullptr;
  }

  for (llvm::Value *value : operands) {
    if (!value)
      continue;
    auto mlirValue = convertValue(value);
    if (failed(mlirValue))
      return failure();
    valuesOut.push_back(*mlirValue);
  }

  SmallVector<int> opBundleSizes;
  SmallVector<Attribute> opBundleTagAttrs;
  if (requiresOpBundles) {
    opBundleSizes.reserve(opBundles.size());
    opBundleTagAttrs.reserve(opBundles.size());

    for (const llvm::OperandBundleUse &bundle : opBundles) {
      opBundleSizes.push_back(bundle.Inputs.size());
      opBundleTagAttrs.push_back(StringAttr::get(context, bundle.getTagName()));

      for (const llvm::Use &opBundleOperand : bundle.Inputs) {
        auto operandMlirValue = convertValue(opBundleOperand.get());
        if (failed(operandMlirValue))
          return failure();
        valuesOut.push_back(*operandMlirValue);
      }
    }

    auto opBundleSizesAttr = DenseI32ArrayAttr::get(context, opBundleSizes);
    auto opBundleSizesAttrNameAttr =
        StringAttr::get(context, LLVMDialect::getOpBundleSizesAttrName());
    attrsOut.push_back({opBundleSizesAttrNameAttr, opBundleSizesAttr});

    auto opBundleTagsAttr = ArrayAttr::get(context, opBundleTagAttrs);
    auto opBundleTagsAttrNameAttr =
        StringAttr::get(context, LLVMDialect::getOpBundleTagsAttrName());
    attrsOut.push_back({opBundleTagsAttrNameAttr, opBundleTagsAttr});
  }

  return success();
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

FPExceptionBehaviorAttr
ModuleImport::matchFPExceptionBehaviorAttr(llvm::Value *value) {
  auto *metadata = cast<llvm::MetadataAsValue>(value);
  auto *mdstr = cast<llvm::MDString>(metadata->getMetadata());
  std::optional<llvm::fp::ExceptionBehavior> optLLVM =
      llvm::convertStrToExceptionBehavior(mdstr->getString());
  assert(optLLVM && "Expecting FP exception behavior");
  return builder.getAttr<FPExceptionBehaviorAttr>(
      convertFPExceptionBehaviorFromLLVM(*optLLVM));
}

RoundingModeAttr ModuleImport::matchRoundingModeAttr(llvm::Value *value) {
  auto *metadata = cast<llvm::MetadataAsValue>(value);
  auto *mdstr = cast<llvm::MDString>(metadata->getMetadata());
  std::optional<llvm::RoundingMode> optLLVM =
      llvm::convertStrToRoundingMode(mdstr->getString());
  assert(optLLVM && "Expecting rounding mode");
  return builder.getAttr<RoundingModeAttr>(
      convertRoundingModeFromLLVM(*optLLVM));
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

FailureOr<SmallVector<Value>>
ModuleImport::convertCallOperands(llvm::CallBase *callInst,
                                  bool allowInlineAsm) {
  bool isInlineAsm = callInst->isInlineAsm();
  if (isInlineAsm && !allowInlineAsm)
    return failure();

  SmallVector<Value> operands;

  // Cannot use isIndirectCall() here because we need to handle Constant callees
  // that are not considered indirect calls by LLVM. However, in MLIR, they are
  // treated as indirect calls to constant operands that need to be converted.
  // Skip the callee operand if it's inline assembly, as it's handled separately
  // in InlineAsmOp.
  llvm::Value *calleeOperand = callInst->getCalledOperand();
  if (!isa<llvm::Function, llvm::GlobalIFunc>(calleeOperand) && !isInlineAsm) {
    FailureOr<Value> called = convertValue(calleeOperand);
    if (failed(called))
      return failure();
    operands.push_back(*called);
  }

  SmallVector<llvm::Value *> args(callInst->args());
  FailureOr<SmallVector<Value>> arguments = convertValues(args);
  if (failed(arguments))
    return failure();

  llvm::append_range(operands, *arguments);
  return operands;
}

/// Checks if `callType` and `calleeType` are compatible and can be represented
/// in MLIR.
static LogicalResult
checkFunctionTypeCompatibility(LLVMFunctionType callType,
                               LLVMFunctionType calleeType) {
  if (callType.getReturnType() != calleeType.getReturnType())
    return failure();

  if (calleeType.isVarArg()) {
    // For variadic functions, the call can have more types than the callee
    // specifies.
    if (callType.getNumParams() < calleeType.getNumParams())
      return failure();
  } else {
    // For non-variadic functions, the number of parameters needs to be the
    // same.
    if (callType.getNumParams() != calleeType.getNumParams())
      return failure();
  }

  // Check that all operands match.
  for (auto [operandType, argumentType] :
       llvm::zip(callType.getParams(), calleeType.getParams()))
    if (operandType != argumentType)
      return failure();

  return success();
}

FailureOr<LLVMFunctionType>
ModuleImport::convertFunctionType(llvm::CallBase *callInst,
                                  bool &isIncompatibleCall) {
  isIncompatibleCall = false;
  auto castOrFailure = [](Type convertedType) -> FailureOr<LLVMFunctionType> {
    auto funcTy = dyn_cast_or_null<LLVMFunctionType>(convertedType);
    if (!funcTy)
      return failure();
    return funcTy;
  };

  llvm::Value *calledOperand = callInst->getCalledOperand();
  FailureOr<LLVMFunctionType> callType =
      castOrFailure(convertType(callInst->getFunctionType()));
  if (failed(callType))
    return failure();
  auto *callee = dyn_cast<llvm::Function>(calledOperand);

  llvm::FunctionType *origCalleeType = nullptr;
  if (callee) {
    origCalleeType = callee->getFunctionType();
  } else if (auto *ifunc = dyn_cast<llvm::GlobalIFunc>(calledOperand)) {
    origCalleeType = cast<llvm::FunctionType>(ifunc->getValueType());
  }

  // For indirect calls, return the type of the call itself.
  if (!origCalleeType)
    return callType;

  FailureOr<LLVMFunctionType> calleeType =
      castOrFailure(convertType(origCalleeType));
  if (failed(calleeType))
    return failure();

  // Compare the types and notify users via `isIncompatibleCall` if they are not
  // compatible.
  if (failed(checkFunctionTypeCompatibility(*callType, *calleeType))) {
    isIncompatibleCall = true;
    Location loc = translateLoc(callInst->getDebugLoc());
    emitWarning(loc) << "incompatible call and callee types: " << *callType
                     << " and " << *calleeType;
    return callType;
  }

  return calleeType;
}

FlatSymbolRefAttr ModuleImport::convertCalleeName(llvm::CallBase *callInst) {
  llvm::Value *calledOperand = callInst->getCalledOperand();
  if (isa<llvm::Function, llvm::GlobalIFunc>(calledOperand))
    return SymbolRefAttr::get(context, calledOperand->getName());
  return {};
}

LogicalResult ModuleImport::convertIntrinsic(llvm::CallInst *inst) {
  if (succeeded(iface.convertIntrinsic(builder, inst, *this)))
    return success();

  Location loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled intrinsic: " << diag(*inst);
}

ArrayAttr
ModuleImport::convertAsmInlineOperandAttrs(const llvm::CallBase &llvmCall) {
  const auto *ia = cast<llvm::InlineAsm>(llvmCall.getCalledOperand());
  unsigned argIdx = 0;
  SmallVector<mlir::Attribute> opAttrs;
  bool hasIndirect = false;

  for (const llvm::InlineAsm::ConstraintInfo &ci : ia->ParseConstraints()) {
    // Only deal with constraints that correspond to call arguments.
    if (ci.Type == llvm::InlineAsm::isLabel || !ci.hasArg())
      continue;

    // Only increment `argIdx` in terms of constraints containing arguments,
    // which are guaranteed to happen in the same order of the call arguments.
    if (ci.isIndirect) {
      if (llvm::Type *paramEltType = llvmCall.getParamElementType(argIdx)) {
        SmallVector<mlir::NamedAttribute> attrs;
        attrs.push_back(builder.getNamedAttr(
            mlir::LLVM::InlineAsmOp::getElementTypeAttrName(),
            mlir::TypeAttr::get(convertType(paramEltType))));
        opAttrs.push_back(builder.getDictionaryAttr(attrs));
        hasIndirect = true;
      }
    } else {
      opAttrs.push_back(builder.getDictionaryAttr({}));
    }
    argIdx++;
  }

  // Avoid emitting an array where all entries are empty dictionaries.
  return hasIndirect ? ArrayAttr::get(mlirModule->getContext(), opAttrs)
                     : nullptr;
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
      auto brOp = LLVM::BrOp::create(builder, loc, succBlockArgs.front(),
                                     succBlocks.front());
      mapNoResultOp(inst, brOp);
      return success();
    }
    FailureOr<Value> condition = convertValue(brInst->getCondition());
    if (failed(condition))
      return failure();
    auto condBrOp = LLVM::CondBrOp::create(
        builder, loc, *condition, succBlocks.front(), succBlockArgs.front(),
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

    auto switchOp = SwitchOp::create(builder, loc, *condition,
                                     lookupBlock(defaultBB), defaultBlockArgs,
                                     caseValues, caseBlocks, caseOperandRefs);
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
    llvm::Value *calledOperand = callInst->getCalledOperand();

    FailureOr<SmallVector<Value>> operands =
        convertCallOperands(callInst, /*allowInlineAsm=*/true);
    if (failed(operands))
      return failure();

    auto callOp = [&]() -> FailureOr<Operation *> {
      if (auto *asmI = dyn_cast<llvm::InlineAsm>(calledOperand)) {
        Type resultTy = convertType(callInst->getType());
        if (!resultTy)
          return failure();
        ArrayAttr operandAttrs = convertAsmInlineOperandAttrs(*callInst);
        return InlineAsmOp::create(
                   builder, loc, resultTy, *operands,
                   builder.getStringAttr(asmI->getAsmString()),
                   builder.getStringAttr(asmI->getConstraintString()),
                   asmI->hasSideEffects(), asmI->isAlignStack(),
                   convertTailCallKindFromLLVM(callInst->getTailCallKind()),
                   AsmDialectAttr::get(
                       mlirModule.getContext(),
                       convertAsmDialectFromLLVM(asmI->getDialect())),
                   operandAttrs)
            .getOperation();
      }
      bool isIncompatibleCall;
      FailureOr<LLVMFunctionType> funcTy =
          convertFunctionType(callInst, isIncompatibleCall);
      if (failed(funcTy))
        return failure();

      FlatSymbolRefAttr callee = nullptr;
      if (isIncompatibleCall) {
        // Use an indirect call (in order to represent valid and verifiable LLVM
        // IR). Build the indirect call by passing an empty `callee` operand and
        // insert into `operands` to include the indirect call target.
        FlatSymbolRefAttr calleeSym = convertCalleeName(callInst);
        Value indirectCallVal = LLVM::AddressOfOp::create(
            builder, loc, LLVM::LLVMPointerType::get(context), calleeSym);
        operands->insert(operands->begin(), indirectCallVal);
      } else {
        // Regular direct call using callee name.
        callee = convertCalleeName(callInst);
      }
      CallOp callOp = CallOp::create(builder, loc, *funcTy, callee, *operands);

      if (failed(convertCallAttributes(callInst, callOp)))
        return failure();

      // Handle parameter and result attributes unless it's an incompatible
      // call.
      if (!isIncompatibleCall)
        convertArgAndResultAttrs(callInst, callOp);
      return callOp.getOperation();
    }();

    if (failed(callOp))
      return failure();

    if (!callInst->getType()->isVoidTy())
      mapValue(inst, (*callOp)->getResult(0));
    else
      mapNoResultOp(inst, *callOp);
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
        LandingpadOp::create(builder, loc, type, lpInst->isCleanup(), operands);
    mapValue(inst, lpOp);
    return success();
  }
  if (inst->getOpcode() == llvm::Instruction::Invoke) {
    auto *invokeInst = cast<llvm::InvokeInst>(inst);

    if (invokeInst->isInlineAsm())
      return emitError(loc) << "invoke of inline assembly is not supported";

    FailureOr<SmallVector<Value>> operands = convertCallOperands(invokeInst);
    if (failed(operands))
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

    bool isIncompatibleInvoke;
    FailureOr<LLVMFunctionType> funcTy =
        convertFunctionType(invokeInst, isIncompatibleInvoke);
    if (failed(funcTy))
      return failure();

    FlatSymbolRefAttr calleeName = nullptr;
    if (isIncompatibleInvoke) {
      // Use an indirect invoke (in order to represent valid and verifiable LLVM
      // IR). Build the indirect invoke by passing an empty `callee` operand and
      // insert into `operands` to include the indirect invoke target.
      FlatSymbolRefAttr calleeSym = convertCalleeName(invokeInst);
      Value indirectInvokeVal = LLVM::AddressOfOp::create(
          builder, loc, LLVM::LLVMPointerType::get(context), calleeSym);
      operands->insert(operands->begin(), indirectInvokeVal);
    } else {
      // Regular direct invoke using callee name.
      calleeName = convertCalleeName(invokeInst);
    }
    // Create the invoke operation. Normal destination block arguments will be
    // added later on to handle the case in which the operation result is
    // included in this list.
    auto invokeOp = InvokeOp::create(
        builder, loc, *funcTy, calleeName, *operands, directNormalDest,
        ValueRange(), lookupBlock(invokeInst->getUnwindDest()), unwindArgs);

    if (failed(convertInvokeAttributes(invokeInst, invokeOp)))
      return failure();

    // Handle parameter and result attributes unless it's an incompatible
    // invoke.
    if (!isIncompatibleInvoke)
      convertArgAndResultAttrs(invokeInst, invokeOp);

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
      LLVM::BrOp::create(builder, loc, normalArgs, normalDest);
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
    auto gepOp = GEPOp::create(
        builder, loc, type, sourceElementType, *basePtr, indices,
        static_cast<GEPNoWrapFlags>(gepInst->getNoWrapFlags().getRaw()));
    mapValue(inst, gepOp);
    return success();
  }

  if (inst->getOpcode() == llvm::Instruction::IndirectBr) {
    auto *indBrInst = cast<llvm::IndirectBrInst>(inst);

    FailureOr<Value> basePtr = convertValue(indBrInst->getAddress());
    if (failed(basePtr))
      return failure();

    SmallVector<Block *> succBlocks;
    SmallVector<SmallVector<Value>> succBlockArgs;
    for (auto i : llvm::seq<unsigned>(0, indBrInst->getNumSuccessors())) {
      llvm::BasicBlock *succ = indBrInst->getSuccessor(i);
      SmallVector<Value> blockArgs;
      if (failed(convertBranchArgs(indBrInst, succ, blockArgs)))
        return failure();
      succBlocks.push_back(lookupBlock(succ));
      succBlockArgs.push_back(blockArgs);
    }
    SmallVector<ValueRange> succBlockArgsRange =
        llvm::to_vector_of<ValueRange>(succBlockArgs);
    Location loc = translateLoc(inst->getDebugLoc());
    auto indBrOp = LLVM::IndirectBrOp::create(builder, loc, *basePtr,
                                              succBlockArgsRange, succBlocks);

    mapNoResultOp(inst, indBrOp);
    return success();
  }

  // Convert all instructions that have an mlirBuilder.
  if (succeeded(convertInstructionImpl(builder, inst, *this, iface)))
    return success();

  return emitError(loc) << "unhandled instruction: " << diag(*inst);
}

LogicalResult ModuleImport::processInstruction(llvm::Instruction *inst) {
  // FIXME: Support uses of SubtargetData.
  // FIXME: Add support for call / operand attributes.
  // FIXME: Add support for the cleanupret, catchret, catchswitch, callbr,
  // vaarg, catchpad, cleanuppad instructions.

  // Convert LLVM intrinsics calls to MLIR intrinsics.
  if (auto *intrinsic = dyn_cast<llvm::IntrinsicInst>(inst))
    return convertIntrinsic(intrinsic);

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
        ce->getType() == llvm::PointerType::getUnqual(f->getContext())) {
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
  funcOp.setMemoryEffectsAttr(memAttr);
}

// List of LLVM IR attributes that map to an explicit attribute on the MLIR
// LLVMFuncOp.
static constexpr std::array kExplicitAttributes{
    StringLiteral("aarch64_in_za"),
    StringLiteral("aarch64_inout_za"),
    StringLiteral("aarch64_new_za"),
    StringLiteral("aarch64_out_za"),
    StringLiteral("aarch64_preserves_za"),
    StringLiteral("aarch64_pstate_sm_body"),
    StringLiteral("aarch64_pstate_sm_compatible"),
    StringLiteral("aarch64_pstate_sm_enabled"),
    StringLiteral("alwaysinline"),
    StringLiteral("approx-func-fp-math"),
    StringLiteral("convergent"),
    StringLiteral("denormal-fp-math"),
    StringLiteral("denormal-fp-math-f32"),
    StringLiteral("fp-contract"),
    StringLiteral("frame-pointer"),
    StringLiteral("instrument-function-entry"),
    StringLiteral("instrument-function-exit"),
    StringLiteral("no-infs-fp-math"),
    StringLiteral("no-nans-fp-math"),
    StringLiteral("no-signed-zeros-fp-math"),
    StringLiteral("noinline"),
    StringLiteral("nounwind"),
    StringLiteral("optnone"),
    StringLiteral("target-features"),
    StringLiteral("tune-cpu"),
    StringLiteral("unsafe-fp-math"),
    StringLiteral("uwtable"),
    StringLiteral("vscale_range"),
    StringLiteral("willreturn"),
};

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

    // Skip attributes that map to an explicit attribute on the LLVMFuncOp.
    if (llvm::is_contained(kExplicitAttributes, attrName))
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

  if (func->hasFnAttribute(llvm::Attribute::NoInline))
    funcOp.setNoInline(true);
  if (func->hasFnAttribute(llvm::Attribute::AlwaysInline))
    funcOp.setAlwaysInline(true);
  if (func->hasFnAttribute(llvm::Attribute::OptimizeNone))
    funcOp.setOptimizeNone(true);
  if (func->hasFnAttribute(llvm::Attribute::Convergent))
    funcOp.setConvergent(true);
  if (func->hasFnAttribute(llvm::Attribute::NoUnwind))
    funcOp.setNoUnwind(true);
  if (func->hasFnAttribute(llvm::Attribute::WillReturn))
    funcOp.setWillReturn(true);

  if (func->hasFnAttribute("aarch64_pstate_sm_enabled"))
    funcOp.setArmStreaming(true);
  else if (func->hasFnAttribute("aarch64_pstate_sm_body"))
    funcOp.setArmLocallyStreaming(true);
  else if (func->hasFnAttribute("aarch64_pstate_sm_compatible"))
    funcOp.setArmStreamingCompatible(true);

  if (func->hasFnAttribute("aarch64_new_za"))
    funcOp.setArmNewZa(true);
  else if (func->hasFnAttribute("aarch64_in_za"))
    funcOp.setArmInZa(true);
  else if (func->hasFnAttribute("aarch64_out_za"))
    funcOp.setArmOutZa(true);
  else if (func->hasFnAttribute("aarch64_inout_za"))
    funcOp.setArmInoutZa(true);
  else if (func->hasFnAttribute("aarch64_preserves_za"))
    funcOp.setArmPreservesZa(true);

  llvm::Attribute attr = func->getFnAttribute(llvm::Attribute::VScaleRange);
  if (attr.isValid()) {
    MLIRContext *context = funcOp.getContext();
    auto intTy = IntegerType::get(context, 32);
    funcOp.setVscaleRangeAttr(LLVM::VScaleRangeAttr::get(
        context, IntegerAttr::get(intTy, attr.getVScaleRangeMin()),
        IntegerAttr::get(intTy, attr.getVScaleRangeMax().value_or(0))));
  }

  // Process frame-pointer attribute.
  if (func->hasFnAttribute("frame-pointer")) {
    StringRef stringRefFramePointerKind =
        func->getFnAttribute("frame-pointer").getValueAsString();
    funcOp.setFramePointerAttr(LLVM::FramePointerKindAttr::get(
        funcOp.getContext(), LLVM::framePointerKind::symbolizeFramePointerKind(
                                 stringRefFramePointerKind)
                                 .value()));
  }

  if (llvm::Attribute attr = func->getFnAttribute("target-cpu");
      attr.isStringAttribute())
    funcOp.setTargetCpuAttr(StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("tune-cpu");
      attr.isStringAttribute())
    funcOp.setTuneCpuAttr(StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("target-features");
      attr.isStringAttribute())
    funcOp.setTargetFeaturesAttr(
        LLVM::TargetFeaturesAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("reciprocal-estimates");
      attr.isStringAttribute())
    funcOp.setReciprocalEstimatesAttr(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("prefer-vector-width");
      attr.isStringAttribute())
    funcOp.setPreferVectorWidth(attr.getValueAsString());

  if (llvm::Attribute attr = func->getFnAttribute("unsafe-fp-math");
      attr.isStringAttribute())
    funcOp.setUnsafeFpMath(attr.getValueAsBool());

  if (llvm::Attribute attr = func->getFnAttribute("no-infs-fp-math");
      attr.isStringAttribute())
    funcOp.setNoInfsFpMath(attr.getValueAsBool());

  if (llvm::Attribute attr = func->getFnAttribute("no-nans-fp-math");
      attr.isStringAttribute())
    funcOp.setNoNansFpMath(attr.getValueAsBool());

  if (llvm::Attribute attr = func->getFnAttribute("approx-func-fp-math");
      attr.isStringAttribute())
    funcOp.setApproxFuncFpMath(attr.getValueAsBool());

  if (llvm::Attribute attr = func->getFnAttribute("instrument-function-entry");
      attr.isStringAttribute())
    funcOp.setInstrumentFunctionEntry(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("instrument-function-exit");
      attr.isStringAttribute())
    funcOp.setInstrumentFunctionExit(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("no-signed-zeros-fp-math");
      attr.isStringAttribute())
    funcOp.setNoSignedZerosFpMath(attr.getValueAsBool());

  if (llvm::Attribute attr = func->getFnAttribute("denormal-fp-math");
      attr.isStringAttribute())
    funcOp.setDenormalFpMathAttr(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("denormal-fp-math-f32");
      attr.isStringAttribute())
    funcOp.setDenormalFpMathF32Attr(
        StringAttr::get(context, attr.getValueAsString()));

  if (llvm::Attribute attr = func->getFnAttribute("fp-contract");
      attr.isStringAttribute())
    funcOp.setFpContractAttr(StringAttr::get(context, attr.getValueAsString()));

  if (func->hasUWTable()) {
    ::llvm::UWTableKind uwtableKind = func->getUWTableKind();
    funcOp.setUwtableKindAttr(LLVM::UWTableKindAttr::get(
        funcOp.getContext(), convertUWTableKindFromLLVM(uwtableKind)));
  }
}

DictionaryAttr
ModuleImport::convertArgOrResultAttrSet(llvm::AttributeSet llvmAttrSet) {
  SmallVector<NamedAttribute> paramAttrs;
  for (auto [llvmKind, mlirName] : getAttrKindToNameMapping()) {
    auto llvmAttr = llvmAttrSet.getAttribute(llvmKind);
    // Skip attributes that are not attached.
    if (!llvmAttr.isValid())
      continue;

    // TODO: Import captures(none) as a nocapture unit attribute until the
    // LLVM dialect switches to the captures representation.
    if (llvmAttr.hasKindAsEnum() &&
        llvmAttr.getKindAsEnum() == llvm::Attribute::Captures) {
      if (llvm::capturesNothing(llvmAttr.getCaptureInfo()))
        paramAttrs.push_back(
            builder.getNamedAttr(mlirName, builder.getUnitAttr()));
      continue;
    }

    Attribute mlirAttr;
    if (llvmAttr.isTypeAttribute())
      mlirAttr = TypeAttr::get(convertType(llvmAttr.getValueAsType()));
    else if (llvmAttr.isIntAttribute())
      mlirAttr = builder.getI64IntegerAttr(llvmAttr.getValueAsInt());
    else if (llvmAttr.isEnumAttribute())
      mlirAttr = builder.getUnitAttr();
    else if (llvmAttr.isConstantRangeAttribute()) {
      const llvm::ConstantRange &value = llvmAttr.getValueAsConstantRange();
      mlirAttr = builder.getAttr<LLVM::ConstantRangeAttr>(value.getLower(),
                                                          value.getUpper());
    } else {
      llvm_unreachable("unexpected parameter attribute kind");
    }
    paramAttrs.push_back(builder.getNamedAttr(mlirName, mlirAttr));
  }

  return builder.getDictionaryAttr(paramAttrs);
}

void ModuleImport::convertArgAndResultAttrs(llvm::Function *func,
                                            LLVMFuncOp funcOp) {
  auto llvmAttrs = func->getAttributes();
  for (size_t i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    llvm::AttributeSet llvmArgAttrs = llvmAttrs.getParamAttrs(i);
    funcOp.setArgAttrs(i, convertArgOrResultAttrSet(llvmArgAttrs));
  }
  // Convert the result attributes and attach them wrapped in an ArrayAttribute
  // to the funcOp.
  llvm::AttributeSet llvmResAttr = llvmAttrs.getRetAttrs();
  if (!llvmResAttr.hasAttributes())
    return;
  funcOp.setResAttrsAttr(
      builder.getArrayAttr({convertArgOrResultAttrSet(llvmResAttr)}));
}

void ModuleImport::convertArgAndResultAttrs(
    llvm::CallBase *call, ArgAndResultAttrsOpInterface attrsOp,
    ArrayRef<unsigned> immArgPositions) {
  // Compute the set of immediate argument positions.
  llvm::SmallDenseSet<unsigned> immArgPositionsSet(immArgPositions.begin(),
                                                   immArgPositions.end());
  // Convert the argument attributes and filter out immediate arguments.
  llvm::AttributeList llvmAttrs = call->getAttributes();
  SmallVector<llvm::AttributeSet> llvmArgAttrsSet;
  bool anyArgAttrs = false;
  for (size_t i = 0, e = call->arg_size(); i < e; ++i) {
    // Skip immediate arguments.
    if (immArgPositionsSet.contains(i))
      continue;
    llvmArgAttrsSet.emplace_back(llvmAttrs.getParamAttrs(i));
    if (llvmArgAttrsSet.back().hasAttributes())
      anyArgAttrs = true;
  }
  auto getArrayAttr = [&](ArrayRef<DictionaryAttr> dictAttrs) {
    SmallVector<Attribute> attrs;
    for (auto &dict : dictAttrs)
      attrs.push_back(dict ? dict : builder.getDictionaryAttr({}));
    return builder.getArrayAttr(attrs);
  };
  if (anyArgAttrs) {
    SmallVector<DictionaryAttr> argAttrs;
    for (auto &llvmArgAttrs : llvmArgAttrsSet)
      argAttrs.emplace_back(convertArgOrResultAttrSet(llvmArgAttrs));
    attrsOp.setArgAttrsAttr(getArrayAttr(argAttrs));
  }

  // Convert the result attributes.
  llvm::AttributeSet llvmResAttr = llvmAttrs.getRetAttrs();
  if (!llvmResAttr.hasAttributes())
    return;
  DictionaryAttr resAttrs = convertArgOrResultAttrSet(llvmResAttr);
  attrsOp.setResAttrsAttr(getArrayAttr({resAttrs}));
}

template <typename Op>
static LogicalResult convertCallBaseAttributes(llvm::CallBase *inst, Op op) {
  op.setCConv(convertCConvFromLLVM(inst->getCallingConv()));
  return success();
}

LogicalResult ModuleImport::convertInvokeAttributes(llvm::InvokeInst *inst,
                                                    InvokeOp op) {
  return convertCallBaseAttributes(inst, op);
}

LogicalResult ModuleImport::convertCallAttributes(llvm::CallInst *inst,
                                                  CallOp op) {
  setFastmathFlagsAttr(inst, op.getOperation());
  // Query the attributes directly instead of using `inst->getFnAttr(Kind)`, the
  // latter does additional lookup to the parent and inherits, changing the
  // semantics too early.
  llvm::AttributeList callAttrs = inst->getAttributes();

  op.setTailCallKind(convertTailCallKindFromLLVM(inst->getTailCallKind()));
  op.setConvergent(callAttrs.getFnAttr(llvm::Attribute::Convergent).isValid());
  op.setNoUnwind(callAttrs.getFnAttr(llvm::Attribute::NoUnwind).isValid());
  op.setWillReturn(callAttrs.getFnAttr(llvm::Attribute::WillReturn).isValid());
  op.setNoInline(callAttrs.getFnAttr(llvm::Attribute::NoInline).isValid());
  op.setAlwaysInline(
      callAttrs.getFnAttr(llvm::Attribute::AlwaysInline).isValid());
  op.setInlineHint(callAttrs.getFnAttr(llvm::Attribute::InlineHint).isValid());

  llvm::MemoryEffects memEffects = inst->getMemoryEffects();
  ModRefInfo othermem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::Other));
  ModRefInfo argMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::ArgMem));
  ModRefInfo inaccessibleMem = convertModRefInfoFromLLVM(
      memEffects.getModRef(llvm::MemoryEffects::Location::InaccessibleMem));
  auto memAttr = MemoryEffectsAttr::get(op.getContext(), othermem, argMem,
                                        inaccessibleMem);
  // Only set the attribute when it does not match the default value.
  if (!memAttr.isReadWrite())
    op.setMemoryEffectsAttr(memAttr);

  return convertCallBaseAttributes(inst, op);
}

LogicalResult ModuleImport::processFunction(llvm::Function *func) {
  clearRegionState();

  auto functionType =
      dyn_cast<LLVMFunctionType>(convertType(func->getFunctionType()));
  if (func->isIntrinsic() &&
      iface.isConvertibleIntrinsic(func->getIntrinsicID()))
    return success();

  bool dsoLocal = func->isDSOLocal();
  CConv cconv = convertCConvFromLLVM(func->getCallingConv());

  // Insert the function at the end of the module.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());

  Location loc = debugImporter->translateFuncLocation(func);
  LLVMFuncOp funcOp = LLVMFuncOp::create(
      builder, loc, func->getName(), functionType,
      convertLinkageFromLLVM(func->getLinkage()), dsoLocal, cconv);

  convertArgAndResultAttrs(func, funcOp);

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

  // Collect the set of basic blocks reachable from the function's entry block.
  // This step is crucial as LLVM IR can contain unreachable blocks that
  // self-dominate. As a result, an operation might utilize a variable it
  // defines, which the import does not support. Given that MLIR lacks block
  // label support, we can safely remove unreachable blocks, as there are no
  // indirect branch instructions that could potentially target these blocks.
  llvm::df_iterator_default_set<llvm::BasicBlock *> reachable;
  for (llvm::BasicBlock *basicBlock : llvm::depth_first_ext(func, reachable))
    (void)basicBlock;

  // Eagerly create all reachable blocks.
  SmallVector<llvm::BasicBlock *> reachableBasicBlocks;
  for (llvm::BasicBlock &basicBlock : *func) {
    // Skip unreachable blocks.
    if (!reachable.contains(&basicBlock)) {
      if (basicBlock.hasAddressTaken())
        return emitError(funcOp.getLoc())
               << "unreachable block '" << basicBlock.getName()
               << "' with address taken";
      continue;
    }
    Region &body = funcOp.getBody();
    Block *block = builder.createBlock(&body, body.end());
    mapBlock(&basicBlock, block);
    reachableBasicBlocks.push_back(&basicBlock);
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
  SetVector<llvm::BasicBlock *> blocks =
      getTopologicallySortedBlocks(reachableBasicBlocks);
  setConstantInsertionPointToStart(lookupBlock(blocks.front()));
  for (llvm::BasicBlock *basicBlock : blocks)
    if (failed(processBasicBlock(basicBlock, lookupBlock(basicBlock))))
      return failure();

  // Process the debug intrinsics that require a delayed conversion after
  // everything else was converted.
  if (failed(processDebugIntrinsics()))
    return failure();

  return success();
}

/// Checks if `dbgIntr` is a kill location that holds metadata instead of an SSA
/// value.
static bool isMetadataKillLocation(llvm::DbgVariableIntrinsic *dbgIntr) {
  if (!dbgIntr->isKillLocation())
    return false;
  llvm::Value *value = dbgIntr->getArgOperand(0);
  auto *nodeAsVal = dyn_cast<llvm::MetadataAsValue>(value);
  if (!nodeAsVal)
    return false;
  return !isa<llvm::ValueAsMetadata>(nodeAsVal->getMetadata());
}

LogicalResult
ModuleImport::processDebugIntrinsic(llvm::DbgVariableIntrinsic *dbgIntr,
                                    DominanceInfo &domInfo) {
  Location loc = translateLoc(dbgIntr->getDebugLoc());
  auto emitUnsupportedWarning = [&]() {
    if (emitExpensiveWarnings)
      emitWarning(loc) << "dropped intrinsic: " << diag(*dbgIntr);
    return success();
  };
  // Drop debug intrinsics with arg lists.
  // TODO: Support debug intrinsics that have arg lists.
  if (dbgIntr->hasArgList())
    return emitUnsupportedWarning();
  // Kill locations can have metadata nodes as location operand. This
  // cannot be converted to poison as the type cannot be reconstructed.
  // TODO: find a way to support this case.
  if (isMetadataKillLocation(dbgIntr))
    return emitUnsupportedWarning();
  // Drop debug intrinsics if the associated variable information cannot be
  // translated due to cyclic debug metadata.
  // TODO: Support cyclic debug metadata.
  DILocalVariableAttr localVariableAttr =
      matchLocalVariableAttr(dbgIntr->getArgOperand(1));
  if (!localVariableAttr)
    return emitUnsupportedWarning();
  FailureOr<Value> argOperand = convertMetadataValue(dbgIntr->getArgOperand(0));
  if (failed(argOperand))
    return emitError(loc) << "failed to convert a debug intrinsic operand: "
                          << diag(*dbgIntr);

  // Ensure that the debug intrinsic is inserted right after its operand is
  // defined. Otherwise, the operand might not necessarily dominate the
  // intrinsic. If the defining operation is a terminator, insert the intrinsic
  // into a dominated block.
  OpBuilder::InsertionGuard guard(builder);
  if (Operation *op = argOperand->getDefiningOp();
      op && op->hasTrait<OpTrait::IsTerminator>()) {
    // Find a dominated block that can hold the debug intrinsic.
    auto dominatedBlocks = domInfo.getNode(op->getBlock())->children();
    // If no block is dominated by the terminator, this intrinisc cannot be
    // converted.
    if (dominatedBlocks.empty())
      return emitUnsupportedWarning();
    // Set insertion point before the terminator, to avoid inserting something
    // before landingpads.
    Block *dominatedBlock = (*dominatedBlocks.begin())->getBlock();
    builder.setInsertionPoint(dominatedBlock->getTerminator());
  } else {
    Value insertPt = *argOperand;
    if (auto blockArg = dyn_cast<BlockArgument>(*argOperand)) {
      // The value might be coming from a phi node and is now a block argument,
      // which means the insertion point is set to the start of the block. If
      // this block is a target destination of an invoke, the insertion point
      // must happen after the landing pad operation.
      Block *insertionBlock = argOperand->getParentBlock();
      if (!insertionBlock->empty() &&
          isa<LandingpadOp>(insertionBlock->front()))
        insertPt = cast<LandingpadOp>(insertionBlock->front()).getRes();
    }

    builder.setInsertionPointAfterValue(insertPt);
  }
  auto locationExprAttr =
      debugImporter->translateExpression(dbgIntr->getExpression());
  Operation *op =
      llvm::TypeSwitch<llvm::DbgVariableIntrinsic *, Operation *>(dbgIntr)
          .Case([&](llvm::DbgDeclareInst *) {
            return LLVM::DbgDeclareOp::create(
                builder, loc, *argOperand, localVariableAttr, locationExprAttr);
          })
          .Case([&](llvm::DbgValueInst *) {
            return LLVM::DbgValueOp::create(
                builder, loc, *argOperand, localVariableAttr, locationExprAttr);
          });
  mapNoResultOp(dbgIntr, op);
  setNonDebugMetadataAttrs(dbgIntr, op);
  return success();
}

LogicalResult ModuleImport::processDebugIntrinsics() {
  DominanceInfo domInfo;
  for (llvm::Instruction *inst : debugIntrinsics) {
    auto *intrCall = cast<llvm::DbgVariableIntrinsic>(inst);
    if (failed(processDebugIntrinsic(intrCall, domInfo)))
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

    // Skip additional processing when the instructions is a debug intrinsics
    // that was not yet converted.
    if (debugIntrinsics.contains(&inst))
      continue;

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

  if (bb->hasAddressTaken()) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(block);
    BlockTagOp::create(builder, block->getParentOp()->getLoc(),
                       BlockTagAttr::get(context, bb->getNumber()));
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

FailureOr<DereferenceableAttr>
ModuleImport::translateDereferenceableAttr(const llvm::MDNode *node,
                                           unsigned kindID) {
  Location loc = mlirModule.getLoc();

  // The only operand should be a constant integer representing the number of
  // dereferenceable bytes.
  if (node->getNumOperands() != 1)
    return emitError(loc) << "dereferenceable metadata must have one operand: "
                          << diagMD(node, llvmModule.get());

  auto *numBytesMD = dyn_cast<llvm::ConstantAsMetadata>(node->getOperand(0));
  auto *numBytesCst = dyn_cast<llvm::ConstantInt>(numBytesMD->getValue());
  if (!numBytesCst || !numBytesCst->getValue().isNonNegative())
    return emitError(loc) << "dereferenceable metadata operand must be a "
                             "non-negative constant integer: "
                          << diagMD(node, llvmModule.get());

  bool mayBeNull = kindID == llvm::LLVMContext::MD_dereferenceable_or_null;
  auto derefAttr = builder.getAttr<DereferenceableAttr>(
      numBytesCst->getZExtValue(), mayBeNull);

  return derefAttr;
}

OwningOpRef<ModuleOp> mlir::translateLLVMIRToModule(
    std::unique_ptr<llvm::Module> llvmModule, MLIRContext *context,
    bool emitExpensiveWarnings, bool dropDICompositeTypeElements,
    bool loadAllDialects, bool preferUnregisteredIntrinsics,
    bool importStructsAsLiterals) {
  // Preload all registered dialects to allow the import to iterate the
  // registered LLVMImportDialectInterface implementations and query the
  // supported LLVM IR constructs before starting the translation. Assumes the
  // LLVM and DLTI dialects that convert the core LLVM IR constructs have been
  // registered before.
  assert(llvm::is_contained(context->getAvailableDialects(),
                            LLVMDialect::getDialectNamespace()));
  assert(llvm::is_contained(context->getAvailableDialects(),
                            DLTIDialect::getDialectNamespace()));
  if (loadAllDialects)
    context->loadAllAvailableDialects();
  OwningOpRef<ModuleOp> module(ModuleOp::create(FileLineColLoc::get(
      StringAttr::get(context, llvmModule->getSourceFileName()), /*line=*/0,
      /*column=*/0)));

  ModuleImport moduleImport(module.get(), std::move(llvmModule),
                            emitExpensiveWarnings, dropDICompositeTypeElements,
                            preferUnregisteredIntrinsics,
                            importStructsAsLiterals);
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
  if (failed(moduleImport.convertAliases()))
    return {};
  if (failed(moduleImport.convertIFuncs()))
    return {};
  moduleImport.convertTargetTriple();
  moduleImport.convertModuleLevelAsm();
  return module;
}
