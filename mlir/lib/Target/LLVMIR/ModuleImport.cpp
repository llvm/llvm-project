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

#include "DebugImporter.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
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

/// Returns a supported MLIR floating point type of the given bit width or null
/// if the bit width is not supported.
static FloatType getDLFloatType(MLIRContext &ctx, int32_t bitwidth) {
  switch (bitwidth) {
  case 16:
    return FloatType::getF16(&ctx);
  case 32:
    return FloatType::getF32(&ctx);
  case 64:
    return FloatType::getF64(&ctx);
  case 80:
    return FloatType::getF80(&ctx);
  case 128:
    return FloatType::getF128(&ctx);
  default:
    return nullptr;
  }
}

static ICmpPredicate getICmpPredicate(llvm::CmpInst::Predicate pred) {
  switch (pred) {
  default:
    llvm_unreachable("incorrect comparison predicate");
  case llvm::CmpInst::Predicate::ICMP_EQ:
    return LLVM::ICmpPredicate::eq;
  case llvm::CmpInst::Predicate::ICMP_NE:
    return LLVM::ICmpPredicate::ne;
  case llvm::CmpInst::Predicate::ICMP_SLT:
    return LLVM::ICmpPredicate::slt;
  case llvm::CmpInst::Predicate::ICMP_SLE:
    return LLVM::ICmpPredicate::sle;
  case llvm::CmpInst::Predicate::ICMP_SGT:
    return LLVM::ICmpPredicate::sgt;
  case llvm::CmpInst::Predicate::ICMP_SGE:
    return LLVM::ICmpPredicate::sge;
  case llvm::CmpInst::Predicate::ICMP_ULT:
    return LLVM::ICmpPredicate::ult;
  case llvm::CmpInst::Predicate::ICMP_ULE:
    return LLVM::ICmpPredicate::ule;
  case llvm::CmpInst::Predicate::ICMP_UGT:
    return LLVM::ICmpPredicate::ugt;
  case llvm::CmpInst::Predicate::ICMP_UGE:
    return LLVM::ICmpPredicate::uge;
  }
  llvm_unreachable("incorrect integer comparison predicate");
}

static FCmpPredicate getFCmpPredicate(llvm::CmpInst::Predicate pred) {
  switch (pred) {
  default:
    llvm_unreachable("incorrect comparison predicate");
  case llvm::CmpInst::Predicate::FCMP_FALSE:
    return LLVM::FCmpPredicate::_false;
  case llvm::CmpInst::Predicate::FCMP_TRUE:
    return LLVM::FCmpPredicate::_true;
  case llvm::CmpInst::Predicate::FCMP_OEQ:
    return LLVM::FCmpPredicate::oeq;
  case llvm::CmpInst::Predicate::FCMP_ONE:
    return LLVM::FCmpPredicate::one;
  case llvm::CmpInst::Predicate::FCMP_OLT:
    return LLVM::FCmpPredicate::olt;
  case llvm::CmpInst::Predicate::FCMP_OLE:
    return LLVM::FCmpPredicate::ole;
  case llvm::CmpInst::Predicate::FCMP_OGT:
    return LLVM::FCmpPredicate::ogt;
  case llvm::CmpInst::Predicate::FCMP_OGE:
    return LLVM::FCmpPredicate::oge;
  case llvm::CmpInst::Predicate::FCMP_ORD:
    return LLVM::FCmpPredicate::ord;
  case llvm::CmpInst::Predicate::FCMP_ULT:
    return LLVM::FCmpPredicate::ult;
  case llvm::CmpInst::Predicate::FCMP_ULE:
    return LLVM::FCmpPredicate::ule;
  case llvm::CmpInst::Predicate::FCMP_UGT:
    return LLVM::FCmpPredicate::ugt;
  case llvm::CmpInst::Predicate::FCMP_UGE:
    return LLVM::FCmpPredicate::uge;
  case llvm::CmpInst::Predicate::FCMP_UNO:
    return LLVM::FCmpPredicate::uno;
  case llvm::CmpInst::Predicate::FCMP_UEQ:
    return LLVM::FCmpPredicate::ueq;
  case llvm::CmpInst::Predicate::FCMP_UNE:
    return LLVM::FCmpPredicate::une;
  }
  llvm_unreachable("incorrect floating point comparison predicate");
}

static AtomicOrdering getLLVMAtomicOrdering(llvm::AtomicOrdering ordering) {
  switch (ordering) {
  case llvm::AtomicOrdering::NotAtomic:
    return LLVM::AtomicOrdering::not_atomic;
  case llvm::AtomicOrdering::Unordered:
    return LLVM::AtomicOrdering::unordered;
  case llvm::AtomicOrdering::Monotonic:
    return LLVM::AtomicOrdering::monotonic;
  case llvm::AtomicOrdering::Acquire:
    return LLVM::AtomicOrdering::acquire;
  case llvm::AtomicOrdering::Release:
    return LLVM::AtomicOrdering::release;
  case llvm::AtomicOrdering::AcquireRelease:
    return LLVM::AtomicOrdering::acq_rel;
  case llvm::AtomicOrdering::SequentiallyConsistent:
    return LLVM::AtomicOrdering::seq_cst;
  }
  llvm_unreachable("incorrect atomic ordering");
}

static AtomicBinOp getLLVMAtomicBinOp(llvm::AtomicRMWInst::BinOp binOp) {
  switch (binOp) {
  case llvm::AtomicRMWInst::Xchg:
    return LLVM::AtomicBinOp::xchg;
  case llvm::AtomicRMWInst::Add:
    return LLVM::AtomicBinOp::add;
  case llvm::AtomicRMWInst::Sub:
    return LLVM::AtomicBinOp::sub;
  case llvm::AtomicRMWInst::And:
    return LLVM::AtomicBinOp::_and;
  case llvm::AtomicRMWInst::Nand:
    return LLVM::AtomicBinOp::nand;
  case llvm::AtomicRMWInst::Or:
    return LLVM::AtomicBinOp::_or;
  case llvm::AtomicRMWInst::Xor:
    return LLVM::AtomicBinOp::_xor;
  case llvm::AtomicRMWInst::Max:
    return LLVM::AtomicBinOp::max;
  case llvm::AtomicRMWInst::Min:
    return LLVM::AtomicBinOp::min;
  case llvm::AtomicRMWInst::UMax:
    return LLVM::AtomicBinOp::umax;
  case llvm::AtomicRMWInst::UMin:
    return LLVM::AtomicBinOp::umin;
  case llvm::AtomicRMWInst::FAdd:
    return LLVM::AtomicBinOp::fadd;
  case llvm::AtomicRMWInst::FSub:
    return LLVM::AtomicBinOp::fsub;
  default:
    llvm_unreachable("unsupported atomic binary operation");
  }
}

/// Converts the sync scope identifier of `fenceInst` to the string
/// representation necessary to build the LLVM dialect fence operation.
static StringRef getLLVMSyncScope(llvm::FenceInst *fenceInst) {
  llvm::LLVMContext &llvmContext = fenceInst->getContext();
  SmallVector<StringRef> syncScopeNames;
  llvmContext.getSyncScopeNames(syncScopeNames);
  for (StringRef name : syncScopeNames)
    if (fenceInst->getSyncScopeID() == llvmContext.getOrInsertSyncScopeID(name))
      return name;
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

/// Creates an attribute containing ABI and preferred alignment numbers parsed
/// a string. The string may be either "abi:preferred" or just "abi". In the
/// latter case, the preferred alignment is considered equal to ABI alignment.
static DenseIntElementsAttr parseDataLayoutAlignment(MLIRContext &ctx,
                                                     StringRef spec) {
  auto i32 = IntegerType::get(&ctx, 32);

  StringRef abiString, preferredString;
  std::tie(abiString, preferredString) = spec.split(':');
  int abi, preferred;
  if (abiString.getAsInteger(/*Radix=*/10, abi))
    return nullptr;

  if (preferredString.empty())
    preferred = abi;
  else if (preferredString.getAsInteger(/*Radix=*/10, preferred))
    return nullptr;

  return DenseIntElementsAttr::get(VectorType::get({2}, i32), {abi, preferred});
}

/// Translate the given LLVM data layout into an MLIR equivalent using the DLTI
/// dialect.
DataLayoutSpecInterface
mlir::translateDataLayout(const llvm::DataLayout &dataLayout,
                          MLIRContext *context) {
  assert(context && "expected MLIR context");
  std::string layoutstr = dataLayout.getStringRepresentation();

  // Remaining unhandled default layout defaults
  // e (little endian if not set)
  // p[n]:64:64:64 (non zero address spaces have 64-bit properties)
  std::string append =
      "p:64:64:64-S0-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f16:16:16-f64:"
      "64:64-f128:128:128-v64:64:64-v128:128:128-a:0:64";
  if (layoutstr.empty())
    layoutstr = append;
  else
    layoutstr = layoutstr + "-" + append;

  StringRef layout(layoutstr);

  SmallVector<DataLayoutEntryInterface> entries;
  StringSet<> seen;
  while (!layout.empty()) {
    // Split at '-'.
    std::pair<StringRef, StringRef> split = layout.split('-');
    StringRef current;
    std::tie(current, layout) = split;

    // Split at ':'.
    StringRef kind, spec;
    std::tie(kind, spec) = current.split(':');
    if (seen.contains(kind))
      continue;
    seen.insert(kind);

    char symbol = kind.front();
    StringRef parameter = kind.substr(1);

    if (symbol == 'i' || symbol == 'f') {
      unsigned bitwidth;
      if (parameter.getAsInteger(/*Radix=*/10, bitwidth))
        return nullptr;
      DenseIntElementsAttr params = parseDataLayoutAlignment(*context, spec);
      if (!params)
        return nullptr;
      auto entry = DataLayoutEntryAttr::get(
          symbol == 'i' ? static_cast<Type>(IntegerType::get(context, bitwidth))
                        : getDLFloatType(*context, bitwidth),
          params);
      entries.emplace_back(entry);
    } else if (symbol == 'e' || symbol == 'E') {
      auto value = StringAttr::get(
          context, symbol == 'e' ? DLTIDialect::kDataLayoutEndiannessLittle
                                 : DLTIDialect::kDataLayoutEndiannessBig);
      auto entry = DataLayoutEntryAttr::get(
          StringAttr::get(context, DLTIDialect::kDataLayoutEndiannessKey),
          value);
      entries.emplace_back(entry);
    }
  }

  return DataLayoutSpecAttr::get(context, entries);
}

/// Get a topologically sorted list of blocks for the given function.
static SetVector<llvm::BasicBlock *>
getTopologicallySortedBlocks(llvm::Function *func) {
  SetVector<llvm::BasicBlock *> blocks;
  for (llvm::BasicBlock &bb : *func) {
    if (blocks.count(&bb) == 0) {
      llvm::ReversePostOrderTraversal<llvm::BasicBlock *> traversal(&bb);
      blocks.insert(traversal.begin(), traversal.end());
    }
  }
  assert(blocks.size() == func->size() && "some blocks are not sorted");

  return blocks;
}

ModuleImport::ModuleImport(ModuleOp mlirModule,
                           std::unique_ptr<llvm::Module> llvmModule)
    : builder(mlirModule->getContext()), context(mlirModule->getContext()),
      mlirModule(mlirModule), llvmModule(std::move(llvmModule)),
      iface(mlirModule->getContext()),
      typeTranslator(*mlirModule->getContext()),
      debugImporter(std::make_unique<DebugImporter>(mlirModule)) {
  builder.setInsertionPointToStart(mlirModule.getBody());
}

MetadataOp ModuleImport::getTBAAMetadataOp() {
  if (tbaaMetadataOp)
    return tbaaMetadataOp;

  OpBuilder::InsertionGuard guard(builder);
  Location loc = mlirModule.getLoc();

  builder.setInsertionPointToEnd(mlirModule.getBody());
  tbaaMetadataOp = builder.create<MetadataOp>(loc, getTBAAMetadataOpName());

  return tbaaMetadataOp;
}

std::string ModuleImport::getNewTBAANodeName(StringRef basename) {
  return (Twine("tbaa_") + Twine(basename) + Twine('_') +
          Twine(tbaaNodeCounter++))
      .str();
}

LogicalResult ModuleImport::processTBAAMetadata(const llvm::MDNode *node) {
  Location loc = mlirModule.getLoc();
  SmallVector<const llvm::MDNode *> workList;
  SetVector<const llvm::MDNode *> nodesToConvert;
  workList.push_back(node);
  while (!workList.empty()) {
    const llvm::MDNode *current = workList.pop_back_val();
    if (tbaaMapping.count(current))
      continue;
    // Allow cycles in TBAA metadata. Just import it as-is,
    // and diagnose the problem during LLVMIR dialect verification.
    if (!nodesToConvert.insert(current))
      continue;
    for (const llvm::MDOperand &operand : current->operands())
      if (auto *opNode = dyn_cast_or_null<const llvm::MDNode>(operand.get()))
        workList.push_back(opNode);
  }

  // If `node` is a valid TBAA root node, then return its identity
  // string, otherwise return std::nullopt.
  auto getIdentityIfRootNode =
      [&](const llvm::MDNode *node) -> std::optional<StringRef> {
    // Root node, e.g.:
    //   !0 = !{!"Simple C/C++ TBAA"}
    if (node->getNumOperands() != 1)
      return std::nullopt;
    // If the operand is MDString, then assume that this is a root node.
    if (const auto *op0 = dyn_cast<const llvm::MDString>(node->getOperand(0)))
      return op0->getString();
    return std::nullopt;
  };

  // If `node` looks like a TBAA type descriptor metadata,
  // then return true, if it is a valid node, and false otherwise.
  // If it does not look like a TBAA type descriptor metadata, then
  // return std::nullopt.
  // If `identity` and `memberTypes/Offsets` are non-null, then they will
  // contain the converted metadata operands for a valid TBAA node (i.e. when
  // true is returned).
  auto isTypeDescriptorNode =
      [&](const llvm::MDNode *node, StringRef *identity = nullptr,
          SmallVectorImpl<Attribute> *memberTypes = nullptr,
          SmallVectorImpl<int64_t> *memberOffsets =
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

      if (memberTypes)
        memberTypes->push_back(tbaaMapping.lookup(memberNode));
      if (memberOffsets)
        memberOffsets->push_back(offset);
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
  auto isTagNode =
      [&](const llvm::MDNode *node, SymbolRefAttr *baseSymRef = nullptr,
          SymbolRefAttr *accessSymRef = nullptr, int64_t *offset = nullptr,
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
    if (baseSymRef)
      *baseSymRef = tbaaMapping.lookup(baseMD);
    if (accessSymRef)
      *accessSymRef = tbaaMapping.lookup(accessMD);
    if (offset)
      *offset = offsetCI->getZExtValue();
    if (isConstant)
      *isConstant = isConst;
    return true;
  };

  // Insert new operations at the end of the MetadataOp.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(&getTBAAMetadataOp().getBody().back());
  StringAttr metadataOpName = SymbolTable::getSymbolName(getTBAAMetadataOp());

  // On the first walk, create SymbolRefAttr's and map them
  // to nodes in `nodesToConvert`.
  for (const auto *current : nodesToConvert) {
    if (std::optional<StringRef> identity = getIdentityIfRootNode(current)) {
      if (identity.value().empty())
        return emitError(loc) << "TBAA root node must have non-empty identity: "
                              << diagMD(current, llvmModule.get());

      // The root nodes do not have operands, so we can create
      // the TBAARootMetadataOp on the first walk.
      auto rootNode = builder.create<TBAARootMetadataOp>(
          loc, getNewTBAANodeName("root"), identity.value());
      tbaaMapping.try_emplace(current, FlatSymbolRefAttr::get(rootNode));
      continue;
    }
    if (std::optional<bool> isValid = isTypeDescriptorNode(current)) {
      if (!isValid.value())
        return failure();
      tbaaMapping.try_emplace(
          current, FlatSymbolRefAttr::get(builder.getContext(),
                                          getNewTBAANodeName("type_desc")));
      continue;
    }
    if (std::optional<bool> isValid = isTagNode(current)) {
      if (!isValid.value())
        return failure();
      // TBAATagOp symbols must be referred by their fully qualified
      // names, so create a path to TBAATagOp symbol.
      tbaaMapping.try_emplace(
          current, SymbolRefAttr::get(
                       builder.getContext(), metadataOpName,
                       FlatSymbolRefAttr::get(builder.getContext(),
                                              getNewTBAANodeName("tag"))));
      continue;
    }
    return emitError(loc) << "unsupported TBAA node format: "
                          << diagMD(current, llvmModule.get());
  }

  // On the second walk, create TBAA operations using the symbol names from the
  // map.
  for (const auto *current : nodesToConvert) {
    StringRef identity;
    SmallVector<Attribute> memberTypes;
    SmallVector<int64_t> memberOffsets;
    if (std::optional<bool> isValid = isTypeDescriptorNode(
            current, &identity, &memberTypes, &memberOffsets)) {
      assert(isValid.value() && "type descriptor node must be valid");

      builder.create<TBAATypeDescriptorOp>(
          loc, tbaaMapping.lookup(current).getLeafReference(),
          builder.getStringAttr(identity), builder.getArrayAttr(memberTypes),
          memberOffsets);
      continue;
    }
    SymbolRefAttr baseSymRef, accessSymRef;
    int64_t offset;
    bool isConstant;
    if (std::optional<bool> isValid = isTagNode(
            current, &baseSymRef, &accessSymRef, &offset, &isConstant)) {
      assert(isValid.value() && "access tag node must be valid");
      builder.create<TBAATagOp>(
          loc, tbaaMapping.lookup(current).getLeafReference(),
          baseSymRef.getLeafReference(), accessSymRef.getLeafReference(),
          offset, isConstant);
      continue;
    }
  }

  return success();
}

LogicalResult ModuleImport::convertMetadata() {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(mlirModule.getBody());
  for (const llvm::Function &func : llvmModule->functions())
    for (const llvm::Instruction &inst : llvm::instructions(func)) {
      llvm::AAMDNodes nodes = inst.getAAMetadata();
      if (!nodes)
        continue;

      if (const llvm::MDNode *tbaaMD = nodes.TBAA)
        if (failed(processTBAAMetadata(tbaaMD)))
          return failure();
      // TODO: only TBAA metadata is currently supported.
    }

  return success();
}

LogicalResult ModuleImport::convertGlobals() {
  for (llvm::GlobalVariable &globalVar : llvmModule->globals()) {
    if (globalVar.getName() == getGlobalCtorsVarName() ||
        globalVar.getName() == getGlobalDtorsVarName()) {
      if (failed(convertGlobalCtorsAndDtors(&globalVar))) {
        return emitError(mlirModule.getLoc())
               << "unhandled global variable: " << diag(globalVar);
      }
      continue;
    }
    if (failed(convertGlobal(&globalVar))) {
      return emitError(mlirModule.getLoc())
             << "unhandled global variable: " << diag(globalVar);
    }
  }
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
      Location loc = debugImporter->translateLoc(inst->getDebugLoc());
      emitWarning(loc) << "unhandled metadata: "
                       << diagMD(node, llvmModule.get()) << " on "
                       << diag(*inst);
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

// We only need integers, floats, doubles, and vectors and tensors thereof for
// attributes. Scalar and vector types are converted to the standard
// equivalents. Array types are converted to ranked tensors; nested array types
// are converted to multi-dimensional tensors or vectors, depending on the
// innermost type being a scalar or a vector.
Type ModuleImport::getStdTypeForAttr(Type type) {
  if (!type)
    return nullptr;

  if (type.isa<IntegerType, FloatType>())
    return type;

  // LLVM vectors can only contain scalars.
  if (LLVM::isCompatibleVectorType(type)) {
    llvm::ElementCount numElements = LLVM::getVectorNumElements(type);
    if (numElements.isScalable()) {
      emitError(UnknownLoc::get(context)) << "scalable vectors not supported";
      return nullptr;
    }
    Type elementType = getStdTypeForAttr(LLVM::getVectorElementType(type));
    if (!elementType)
      return nullptr;
    return VectorType::get(numElements.getKnownMinValue(), elementType);
  }

  // LLVM arrays can contain other arrays or vectors.
  if (auto arrayType = type.dyn_cast<LLVMArrayType>()) {
    // Recover the nested array shape.
    SmallVector<int64_t, 4> shape;
    shape.push_back(arrayType.getNumElements());
    while (arrayType.getElementType().isa<LLVMArrayType>()) {
      arrayType = arrayType.getElementType().cast<LLVMArrayType>();
      shape.push_back(arrayType.getNumElements());
    }

    // If the innermost type is a vector, use the multi-dimensional vector as
    // attribute type.
    if (LLVM::isCompatibleVectorType(arrayType.getElementType())) {
      llvm::ElementCount numElements =
          LLVM::getVectorNumElements(arrayType.getElementType());
      if (numElements.isScalable()) {
        emitError(UnknownLoc::get(context)) << "scalable vectors not supported";
        return nullptr;
      }
      shape.push_back(numElements.getKnownMinValue());

      Type elementType = getStdTypeForAttr(
          LLVM::getVectorElementType(arrayType.getElementType()));
      if (!elementType)
        return nullptr;
      return VectorType::get(shape, elementType);
    }

    // Otherwise use a tensor.
    Type elementType = getStdTypeForAttr(arrayType.getElementType());
    if (!elementType)
      return nullptr;
    return RankedTensorType::get(shape, elementType);
  }

  return nullptr;
}

// Get the given constant as an attribute. Not all constants can be represented
// as attributes.
Attribute ModuleImport::getConstantAsAttr(llvm::Constant *value) {
  if (auto *ci = dyn_cast<llvm::ConstantInt>(value))
    return builder.getIntegerAttr(
        IntegerType::get(context, ci->getType()->getBitWidth()),
        ci->getValue());
  if (auto *c = dyn_cast<llvm::ConstantDataArray>(value))
    if (c->isString())
      return builder.getStringAttr(c->getAsString());
  if (auto *c = dyn_cast<llvm::ConstantFP>(value)) {
    llvm::Type *type = c->getType();
    FloatType floatTy;
    if (type->isBFloatTy())
      floatTy = FloatType::getBF16(context);
    else
      floatTy = getDLFloatType(*context, type->getScalarSizeInBits());
    assert(floatTy && "unsupported floating point type");
    return builder.getFloatAttr(floatTy, c->getValueAPF());
  }
  if (auto *f = dyn_cast<llvm::Function>(value))
    return SymbolRefAttr::get(builder.getContext(), f->getName());

  // Convert constant data to a dense elements attribute.
  if (auto *cd = dyn_cast<llvm::ConstantDataSequential>(value)) {
    Type type = convertType(cd->getElementType());
    auto attrType = getStdTypeForAttr(convertType(cd->getType()))
                        .dyn_cast_or_null<ShapedType>();
    if (!attrType)
      return nullptr;

    if (type.isa<IntegerType>()) {
      SmallVector<APInt, 8> values;
      values.reserve(cd->getNumElements());
      for (unsigned i = 0, e = cd->getNumElements(); i < e; ++i)
        values.push_back(cd->getElementAsAPInt(i));
      return DenseElementsAttr::get(attrType, values);
    }

    if (type.isa<Float32Type, Float64Type>()) {
      SmallVector<APFloat, 8> values;
      values.reserve(cd->getNumElements());
      for (unsigned i = 0, e = cd->getNumElements(); i < e; ++i)
        values.push_back(cd->getElementAsAPFloat(i));
      return DenseElementsAttr::get(attrType, values);
    }

    return nullptr;
  }

  // Unpack constant aggregates to create dense elements attribute whenever
  // possible. Return nullptr (failure) otherwise.
  if (isa<llvm::ConstantAggregate>(value)) {
    auto outerType = getStdTypeForAttr(convertType(value->getType()))
                         .dyn_cast_or_null<ShapedType>();
    if (!outerType)
      return nullptr;

    SmallVector<Attribute, 8> values;
    SmallVector<int64_t, 8> shape;

    for (unsigned i = 0, e = value->getNumOperands(); i < e; ++i) {
      auto nested = getConstantAsAttr(value->getAggregateElement(i))
                        .dyn_cast_or_null<DenseElementsAttr>();
      if (!nested)
        return nullptr;

      values.append(nested.value_begin<Attribute>(),
                    nested.value_end<Attribute>());
    }

    return DenseElementsAttr::get(outerType, values);
  }

  return nullptr;
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
  // Traverse the constant dependencies in post order.
  SmallVector<llvm::Constant *> workList;
  SmallVector<llvm::Constant *> orderedList;
  workList.push_back(constant);
  while (!workList.empty()) {
    llvm::Constant *current = workList.pop_back_val();
    // Skip constants that have been converted before and store all other ones.
    if (valueMapping.count(current))
      continue;
    orderedList.push_back(current);
    // Add the current constant's dependencies to the work list. Only add
    // constant dependencies and skip any other values such as basic block
    // addresses.
    for (llvm::Value *operand : current->operands())
      if (auto *constDependency = dyn_cast<llvm::Constant>(operand))
        workList.push_back(constDependency);
    // Use the `getElementValue` method to add the dependencies of zero
    // initialized aggregate constants since they do not take any operands.
    if (auto *constAgg = dyn_cast<llvm::ConstantAggregateZero>(current)) {
      unsigned numElements = constAgg->getElementCount().getFixedValue();
      for (unsigned i = 0, e = numElements; i != e; ++i)
        workList.push_back(constAgg->getElementValue(i));
    }
  }

  // Add the constants in reverse post order to the result set to ensure all
  // dependencies are satisfied. Avoid storing duplicates since LLVM constants
  // are uniqued and only one `valueMapping` entry per constant is possible.
  SetVector<llvm::Constant *> orderedSet;
  for (llvm::Constant *orderedConst : llvm::reverse(orderedList))
    orderedSet.insert(orderedConst);
  return orderedSet;
}

FailureOr<Value> ModuleImport::convertConstant(llvm::Constant *constant) {
  Location loc = mlirModule.getLoc();

  // Convert constants that can be represented as attributes.
  if (Attribute attr = getConstantAsAttr(constant)) {
    Type type = convertType(constant->getType());
    if (auto symbolRef = attr.dyn_cast<FlatSymbolRefAttr>()) {
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
      assert(noResultOpMapping.find(inst) == noResultOpMapping.end() &&
             "expected constant expression to return a result");
      valueMapping.erase(inst);
      inst->deleteValue();
    });
    // Note: `processInstruction` does not call `convertConstant` recursively
    // since all constant dependencies have been converted before.
    assert(llvm::all_of(inst->operands(), [&](llvm::Value *value) {
      return valueMapping.count(value);
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
    bool isArrayOrStruct = rootType.isa<LLVMArrayType, LLVMStructType>();
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

  return emitError(loc) << "unhandled constant: " << diag(*constant);
}

FailureOr<Value> ModuleImport::convertConstantExpr(llvm::Constant *constant) {
  assert(constantInsertionBlock &&
         "expected the constant insertion block to be non-null");

  // Insert the constant after the last one or at the start or the entry block.
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
  // A value may be wrapped as metadata, for example, when passed to a debug
  // intrinsic. Unwrap these values before the conversion.
  if (auto *nodeAsVal = dyn_cast<llvm::MetadataAsValue>(value))
    if (auto *node = dyn_cast<llvm::ValueAsMetadata>(nodeAsVal->getMetadata()))
      value = node->getValue();

  // Return the mapped value if it has been converted before.
  if (valueMapping.count(value))
    return lookupValue(value);

  // Convert constants such as immediate values that have no mapping yet.
  if (auto *constant = dyn_cast<llvm::Constant>(value))
    return convertConstantExpr(constant);

  Location loc = mlirModule.getLoc();
  if (auto *inst = dyn_cast<llvm::Instruction>(value))
    loc = translateLoc(inst->getDebugLoc());
  return emitError(loc) << "unhandled value: " << diag(*value);
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
  assert(success && "expected a constant value");
  (void)success;
  return integerAttr;
}

DILocalVariableAttr ModuleImport::matchLocalVariableAttr(llvm::Value *value) {
  auto *nodeAsVal = cast<llvm::MetadataAsValue>(value);
  auto *node = cast<llvm::DILocalVariable>(nodeAsVal->getMetadata());
  return debugImporter->translate(node);
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
    SmallVector<int32_t> caseValues(numCases);
    SmallVector<Block *> caseBlocks(numCases);
    for (const auto &it : llvm::enumerate(swInst->cases())) {
      const llvm::SwitchInst::CaseHandle &caseHandle = it.value();
      llvm::BasicBlock *succBB = caseHandle.getCaseSuccessor();
      if (failed(convertBranchArgs(swInst, succBB, caseOperands[it.index()])))
        return failure();
      caseOperandRefs[it.index()] = caseOperands[it.index()];
      caseValues[it.index()] = caseHandle.getCaseValue()->getSExtValue();
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
      FailureOr<Value> operand = convertConstantExpr(lpInst->getClause(i));
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

    SmallVector<Value> normalArgs, unwindArgs;
    (void)convertBranchArgs(invokeInst, invokeInst->getNormalDest(),
                            normalArgs);
    (void)convertBranchArgs(invokeInst, invokeInst->getUnwindDest(),
                            unwindArgs);

    InvokeOp invokeOp;
    if (llvm::Function *callee = invokeInst->getCalledFunction()) {
      invokeOp = builder.create<InvokeOp>(
          loc, types,
          SymbolRefAttr::get(builder.getContext(), callee->getName()), operands,
          lookupBlock(invokeInst->getNormalDest()), normalArgs,
          lookupBlock(invokeInst->getUnwindDest()), unwindArgs);
    } else {
      invokeOp = builder.create<InvokeOp>(
          loc, types, operands, lookupBlock(invokeInst->getNormalDest()),
          normalArgs, lookupBlock(invokeInst->getUnwindDest()), unwindArgs);
    }
    if (!invokeInst->getType()->isVoidTy())
      mapValue(inst, invokeOp.getResults().front());
    else
      mapNoResultOp(inst, invokeOp);
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

void ModuleImport::processFunctionAttributes(llvm::Function *func,
                                             LLVMFuncOp funcOp) {
  processMemoryEffects(func, funcOp);
}

LogicalResult ModuleImport::processFunction(llvm::Function *func) {
  clearBlockAndValueMapping();

  auto functionType =
      convertType(func->getFunctionType()).dyn_cast<LLVMFunctionType>();
  if (func->isIntrinsic() &&
      iface.isConvertibleIntrinsic(func->getIntrinsicID()))
    return success();

  bool dsoLocal = func->hasLocalLinkage();
  CConv cconv = convertCConvFromLLVM(func->getCallingConv());

  // Insert the function at the end of the module.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(mlirModule.getBody(), mlirModule.getBody()->end());

  LLVMFuncOp funcOp = builder.create<LLVMFuncOp>(
      mlirModule.getLoc(), func->getName(), functionType,
      convertLinkageFromLLVM(func->getLinkage()), dsoLocal, cconv);

  // Set the function debug information if available.
  debugImporter->translate(func, funcOp);

  for (const auto &it : llvm::enumerate(functionType.getParams())) {
    llvm::SmallVector<NamedAttribute, 1> argAttrs;
    if (auto *type = func->getParamByValType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(builder.getStringAttr(LLVMDialect::getByValAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = func->getParamByRefType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(
          NamedAttribute(builder.getStringAttr(LLVMDialect::getByRefAttrName()),
                         TypeAttr::get(mlirType)));
    }
    if (auto *type = func->getParamStructRetType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(NamedAttribute(
          builder.getStringAttr(LLVMDialect::getStructRetAttrName()),
          TypeAttr::get(mlirType)));
    }
    if (auto *type = func->getParamInAllocaType(it.index())) {
      Type mlirType = convertType(type);
      argAttrs.push_back(NamedAttribute(
          builder.getStringAttr(LLVMDialect::getInAllocaAttrName()),
          TypeAttr::get(mlirType)));
    }

    funcOp.setArgAttrs(it.index(), argAttrs);
  }

  if (FlatSymbolRefAttr personality = getPersonalityAsAttr(func))
    funcOp.setPersonalityAttr(personality);
  else if (func->hasPersonalityFn())
    emitWarning(funcOp.getLoc(), "could not deduce personality, skipping it");

  if (func->hasGC())
    funcOp.setGarbageCollector(StringRef(func->getGC()));

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
      Location loc = debugImporter->translateLoc(inst.getDebugLoc());
      emitWarning(loc) << "dropped instruction: " << diag(inst);
    }
  }
  return success();
}

OwningOpRef<ModuleOp>
mlir::translateLLVMIRToModule(std::unique_ptr<llvm::Module> llvmModule,
                              MLIRContext *context) {
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

  DataLayoutSpecInterface dlSpec =
      translateDataLayout(llvmModule->getDataLayout(), context);
  if (!dlSpec) {
    emitError(UnknownLoc::get(context), "can't translate data layout");
    return {};
  }
  module.get()->setAttr(DLTIDialect::kDataLayoutAttrName, dlSpec);

  ModuleImport moduleImport(module.get(), std::move(llvmModule));
  if (failed(moduleImport.initializeImportInterface()))
    return {};
  if (failed(moduleImport.convertMetadata()))
    return {};
  if (failed(moduleImport.convertGlobals()))
    return {};
  if (failed(moduleImport.convertFunctions()))
    return {};

  return module;
}
