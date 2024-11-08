//===- TestDialectInterfaces.cpp - Test dialect interface definitions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "TestOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Reducer/ReductionPatternInterface.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace test;

//===----------------------------------------------------------------------===//
// TestDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

/// Testing the correctness of some traits.
static_assert(
    llvm::is_detected<OpTrait::has_implicit_terminator_t,
                      SingleBlockImplicitTerminatorOp>::value,
    "has_implicit_terminator_t does not match SingleBlockImplicitTerminatorOp");
static_assert(OpTrait::hasSingleBlockImplicitTerminator<
                  SingleBlockImplicitTerminatorOp>::value,
              "hasSingleBlockImplicitTerminator does not match "
              "SingleBlockImplicitTerminatorOp");

struct TestResourceBlobManagerInterface
    : public ResourceBlobManagerDialectInterfaceBase<
          TestDialectResourceBlobHandle> {
  using ResourceBlobManagerDialectInterfaceBase<
      TestDialectResourceBlobHandle>::ResourceBlobManagerDialectInterfaceBase;
};

namespace {
enum test_encoding { k_attr_params = 0, k_test_i32 = 99 };
} // namespace

// Test support for interacting with the Bytecode reader/writer.
struct TestBytecodeDialectInterface : public BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;
  TestBytecodeDialectInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const final {
    if (auto concreteType = llvm::dyn_cast<TestI32Type>(type)) {
      writer.writeVarInt(test_encoding::k_test_i32);
      return success();
    }
    return failure();
  }

  Type readType(DialectBytecodeReader &reader) const final {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding)))
      return Type();
    if (encoding == test_encoding::k_test_i32)
      return TestI32Type::get(getContext());
    return Type();
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const final {
    if (auto concreteAttr = llvm::dyn_cast<TestAttrParamsAttr>(attr)) {
      writer.writeVarInt(test_encoding::k_attr_params);
      writer.writeVarInt(concreteAttr.getV0());
      writer.writeVarInt(concreteAttr.getV1());
      return success();
    }
    return failure();
  }

  Attribute readAttribute(DialectBytecodeReader &reader) const final {
    auto versionOr = reader.getDialectVersion<test::TestDialect>();
    // Assume current version if not available through the reader.
    const auto version =
        (succeeded(versionOr))
            ? *reinterpret_cast<const TestDialectVersion *>(*versionOr)
            : TestDialectVersion();
    if (version.major_ < 2)
      return readAttrOldEncoding(reader);
    if (version.major_ == 2 && version.minor_ == 0)
      return readAttrNewEncoding(reader);
    // Forbid reading future versions by returning nullptr.
    return Attribute();
  }

  // Emit a specific version of the dialect.
  void writeVersion(DialectBytecodeWriter &writer) const final {
    // Construct the current dialect version.
    test::TestDialectVersion versionToEmit;

    // Check if a target version to emit was specified on the writer configs.
    auto versionOr = writer.getDialectVersion<test::TestDialect>();
    if (succeeded(versionOr))
      versionToEmit =
          *reinterpret_cast<const test::TestDialectVersion *>(*versionOr);
    writer.writeVarInt(versionToEmit.major_); // major
    writer.writeVarInt(versionToEmit.minor_); // minor
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const final {
    uint64_t major_, minor_;
    if (failed(reader.readVarInt(major_)) || failed(reader.readVarInt(minor_)))
      return nullptr;
    auto version = std::make_unique<TestDialectVersion>();
    version->major_ = major_;
    version->minor_ = minor_;
    return version;
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &version_) const final {
    const auto &version = static_cast<const TestDialectVersion &>(version_);
    if ((version.major_ == 2) && (version.minor_ == 0))
      return success();
    if (version.major_ > 2 || (version.major_ == 2 && version.minor_ > 0)) {
      return topLevelOp->emitError()
             << "current test dialect version is 2.0, can't parse version: "
             << version.major_ << "." << version.minor_;
    }
    // Prior version 2.0, the old op supported only a single attribute called
    // "dimensions". We can perform the upgrade.
    topLevelOp->walk([](TestVersionedOpA op) {
      // Prior version 2.0, `readProperties` did not process the modifier
      // attribute. Handle that according to the version here.
      auto &prop = op.getProperties();
      prop.modifier = BoolAttr::get(op->getContext(), false);
    });
    return success();
  }

private:
  Attribute readAttrNewEncoding(DialectBytecodeReader &reader) const {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding)) ||
        encoding != test_encoding::k_attr_params)
      return Attribute();
    // The new encoding has v0 first, v1 second.
    uint64_t v0, v1;
    if (failed(reader.readVarInt(v0)) || failed(reader.readVarInt(v1)))
      return Attribute();
    return TestAttrParamsAttr::get(getContext(), static_cast<int>(v0),
                                   static_cast<int>(v1));
  }

  Attribute readAttrOldEncoding(DialectBytecodeReader &reader) const {
    uint64_t encoding;
    if (failed(reader.readVarInt(encoding)) ||
        encoding != test_encoding::k_attr_params)
      return Attribute();
    // The old encoding has v1 first, v0 second.
    uint64_t v0, v1;
    if (failed(reader.readVarInt(v1)) || failed(reader.readVarInt(v0)))
      return Attribute();
    return TestAttrParamsAttr::get(getContext(), static_cast<int>(v0),
                                   static_cast<int>(v1));
  }
};

// Test support for interacting with the AsmPrinter.
struct TestOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  TestOpAsmInterface(Dialect *dialect, TestResourceBlobManagerInterface &mgr)
      : OpAsmDialectInterface(dialect), blobManager(mgr) {}

  //===------------------------------------------------------------------===//
  // Aliases
  //===------------------------------------------------------------------===//

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    StringAttr strAttr = dyn_cast<StringAttr>(attr);
    if (!strAttr)
      return AliasResult::NoAlias;

    // Check the contents of the string attribute to see what the test alias
    // should be named.
    std::optional<StringRef> aliasName =
        StringSwitch<std::optional<StringRef>>(strAttr.getValue())
            .Case("alias_test:dot_in_name", StringRef("test.alias"))
            .Case("alias_test:trailing_digit", StringRef("test_alias0"))
            .Case("alias_test:prefixed_digit", StringRef("0_test_alias"))
            .Case("alias_test:prefixed_symbol", StringRef("%test"))
            .Case("alias_test:sanitize_conflict_a",
                  StringRef("test_alias_conflict0"))
            .Case("alias_test:sanitize_conflict_b",
                  StringRef("test_alias_conflict0_"))
            .Case("alias_test:tensor_encoding", StringRef("test_encoding"))
            .Default(std::nullopt);
    if (!aliasName)
      return AliasResult::NoAlias;

    os << *aliasName;
    return AliasResult::FinalAlias;
  }

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto tupleType = dyn_cast<TupleType>(type)) {
      if (tupleType.size() > 0 &&
          llvm::all_of(tupleType.getTypes(), [](Type elemType) {
            return isa<SimpleAType>(elemType);
          })) {
        os << "test_tuple";
        return AliasResult::FinalAlias;
      }
    }
    if (auto intType = dyn_cast<TestIntegerType>(type)) {
      if (intType.getSignedness() ==
              TestIntegerType::SignednessSemantics::Unsigned &&
          intType.getWidth() == 8) {
        os << "test_ui8";
        return AliasResult::FinalAlias;
      }
    }
    if (auto recType = dyn_cast<TestRecursiveType>(type)) {
      if (recType.getName() == "type_to_alias") {
        // We only make alias for a specific recursive type.
        os << "testrec";
        return AliasResult::FinalAlias;
      }
    }
    if (auto recAliasType = dyn_cast<TestRecursiveAliasType>(type)) {
      os << recAliasType.getName();
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }

  //===------------------------------------------------------------------===//
  // Resources
  //===------------------------------------------------------------------===//

  std::string
  getResourceKey(const AsmDialectResourceHandle &handle) const override {
    return cast<TestDialectResourceBlobHandle>(handle).getKey().str();
  }

  FailureOr<AsmDialectResourceHandle>
  declareResource(StringRef key) const final {
    return blobManager.insert(key);
  }

  LogicalResult parseResource(AsmParsedResourceEntry &entry) const final {
    FailureOr<AsmResourceBlob> blob = entry.parseAsBlob();
    if (failed(blob))
      return failure();

    // Update the blob for this entry.
    blobManager.update(entry.getKey(), std::move(*blob));
    return success();
  }

  void
  buildResources(Operation *op,
                 const SetVector<AsmDialectResourceHandle> &referencedResources,
                 AsmResourceBuilder &provider) const final {
    blobManager.buildResources(provider, referencedResources.getArrayRef());
  }

private:
  /// The blob manager for the dialect.
  TestResourceBlobManagerInterface &blobManager;
};

struct TestDialectFoldInterface : public DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // If this is a one region operation, then insert into it.
    return isa<OneRegionOp>(region->getParentOp());
  }
};

/// This class defines the interface for handling inlining with standard
/// operations.
struct TestInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Don't allow inlining calls that are marked `noinline`.
    return !call->hasAttr("noinline");
  }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    // Inlining into test dialect regions is legal.
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool shouldAnalyzeRecursively(Operation *op) const final {
    // Analyze recursively if this is not a functional region operation, it
    // froms a separate functional scope.
    return !isa<FunctionalRegionOp>(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only handle "test.return" here.
    auto returnOp = dyn_cast<TestReturnOp>(op);
    if (!returnOp)
      return;

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  /// Attempt to materialize a conversion for a type mismatch between a call
  /// from this dialect, and a callable region. This method should generate an
  /// operation that takes 'input' as the only operand, and produces a single
  /// result of 'resultType'. If a conversion can not be generated, nullptr
  /// should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type resultType,
                                       Location conversionLoc) const final {
    // Only allow conversion for i16/i32 types.
    if (!(resultType.isSignlessInteger(16) ||
          resultType.isSignlessInteger(32)) ||
        !(input.getType().isSignlessInteger(16) ||
          input.getType().isSignlessInteger(32)))
      return nullptr;
    return builder.create<TestCastOp>(conversionLoc, resultType, input);
  }

  Value handleArgument(OpBuilder &builder, Operation *call, Operation *callable,
                       Value argument,
                       DictionaryAttr argumentAttrs) const final {
    if (!argumentAttrs.contains("test.handle_argument"))
      return argument;
    return builder.create<TestTypeChangerOp>(call->getLoc(), argument.getType(),
                                             argument);
  }

  Value handleResult(OpBuilder &builder, Operation *call, Operation *callable,
                     Value result, DictionaryAttr resultAttrs) const final {
    if (!resultAttrs.contains("test.handle_result"))
      return result;
    return builder.create<TestTypeChangerOp>(call->getLoc(), result.getType(),
                                             result);
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
    if (!isa<ConversionCallOp>(call))
      return;

    // Set attributed on all ops in the inlined blocks.
    for (Block &block : inlinedBlocks) {
      block.walk([&](Operation *op) {
        op->setAttr("inlined_conversion", UnitAttr::get(call->getContext()));
      });
    }
  }
};

struct TestReductionPatternInterface : public DialectReductionPatternInterface {
public:
  TestReductionPatternInterface(Dialect *dialect)
      : DialectReductionPatternInterface(dialect) {}

  void populateReductionPatterns(RewritePatternSet &patterns) const final {
    populateTestReductionPatterns(patterns);
  }
};

} // namespace

void TestDialect::registerInterfaces() {
  auto &blobInterface = addInterface<TestResourceBlobManagerInterface>();
  addInterface<TestOpAsmInterface>(blobInterface);

  addInterfaces<TestDialectFoldInterface, TestInlinerInterface,
                TestReductionPatternInterface, TestBytecodeDialectInterface>();
}
