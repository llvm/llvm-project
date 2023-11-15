//===- TestBytecodeCallbacks.cpp - Pass to test bytecode callback hooks  --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include <list>

using namespace mlir;
using namespace llvm;

namespace {
class TestDialectVersionParser : public cl::parser<test::TestDialectVersion> {
public:
  TestDialectVersionParser(cl::Option &O)
      : cl::parser<test::TestDialectVersion>(O) {}

  bool parse(cl::Option &O, StringRef /*argName*/, StringRef arg,
             test::TestDialectVersion &v) {
    long long major_, minor_;
    if (getAsSignedInteger(arg.split(".").first, 10, major_))
      return O.error("Invalid argument '" + arg);
    if (getAsSignedInteger(arg.split(".").second, 10, minor_))
      return O.error("Invalid argument '" + arg);
    v = test::TestDialectVersion(major_, minor_);
    // Returns true on error.
    return false;
  }
  static void print(raw_ostream &os, const test::TestDialectVersion &v) {
    os << v.major_ << "." << v.minor_;
  };
};

/// This is a test pass which uses callbacks to encode attributes and types in a
/// custom fashion.
struct TestBytecodeRoundtripPass
    : public PassWrapper<TestBytecodeRoundtripPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBytecodeRoundtripPass)

  StringRef getArgument() const final { return "test-bytecode-roundtrip"; }
  StringRef getDescription() const final {
    return "Test pass to implement bytecode roundtrip tests.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
  }
  TestBytecodeRoundtripPass() = default;
  TestBytecodeRoundtripPass(const TestBytecodeRoundtripPass &) {}

  LogicalResult initialize(MLIRContext *context) override {
    testDialect = context->getOrLoadDialect<test::TestDialect>();
    return success();
  }

  void runOnOperation() override {
    switch (testKind) {
      // Tests 0-5 implement a custom roundtrip with callbacks.
    case (0):
      return runTest0(getOperation());
    case (1):
      return runTest1(getOperation());
    case (2):
      return runTest2(getOperation());
    case (3):
      return runTest3(getOperation());
    case (4):
      return runTest4(getOperation());
    case (5):
      return runTest5(getOperation());
    case (6):
      // test-kind 6 is a plain roundtrip with downgrade/upgrade to/from
      // `targetVersion`.
      return runTest6(getOperation());
    default:
      llvm_unreachable("unhandled test kind for TestBytecodeCallbacks pass");
    }
  }

  mlir::Pass::Option<test::TestDialectVersion, TestDialectVersionParser>
      targetVersion{*this, "test-dialect-version",
                    llvm::cl::desc(
                        "Specifies the test dialect version to emit and parse"),
                    cl::init(test::TestDialectVersion())};

  mlir::Pass::Option<int> testKind{
      *this, "test-kind", llvm::cl::desc("Specifies the test kind to execute"),
      cl::init(0)};

private:
  void doRoundtripWithConfigs(Operation *op,
                              const BytecodeWriterConfig &writeConfig,
                              const ParserConfig &parseConfig) {
    std::string bytecode;
    llvm::raw_string_ostream os(bytecode);
    if (failed(writeBytecodeToFile(op, os, writeConfig))) {
      op->emitError() << "failed to write bytecode\n";
      signalPassFailure();
      return;
    }
    auto newModuleOp = parseSourceString(StringRef(bytecode), parseConfig);
    if (!newModuleOp.get()) {
      op->emitError() << "failed to read bytecode\n";
      signalPassFailure();
      return;
    }
    // Print the module to the output stream, so that we can filecheck the
    // result.
    newModuleOp->print(llvm::outs());
  }

  // Test0: let's assume that versions older than 2.0 were relying on a special
  // integer attribute of a deprecated dialect called "funky". Assume that its
  // encoding was made by two varInts, the first was the ID (999) and the second
  // contained width and signedness info. We can emit it using a callback
  // writing a custom encoding for the "funky" dialect group, and parse it back
  // with a custom parser reading the same encoding in the same dialect group.
  // Note that the ID 999 does not correspond to a valid integer type in the
  // current encodings of builtin types.
  void runTest0(Operation *op) {
    auto newCtx = std::make_shared<MLIRContext>();
    test::TestDialectVersion targetEmissionVersion = targetVersion;
    BytecodeWriterConfig writeConfig;
    // Set the emission version for the test dialect.
    writeConfig.setDialectVersion<test::TestDialect>(
        std::make_unique<test::TestDialectVersion>(targetEmissionVersion));
    writeConfig.attachTypeCallback(
        [&](Type entryValue, std::optional<StringRef> &dialectGroupName,
            DialectBytecodeWriter &writer) -> LogicalResult {
          // Do not override anything if version greater than 2.0.
          auto versionOr = writer.getDialectVersion<test::TestDialect>();
          assert(succeeded(versionOr) && "expected reader to be able to access "
                                         "the version for test dialect");
          const auto *version =
              reinterpret_cast<const test::TestDialectVersion *>(*versionOr);
          if (version->major_ >= 2)
            return failure();

          // For version less than 2.0, override the encoding of IntegerType.
          if (auto type = llvm::dyn_cast<IntegerType>(entryValue)) {
            llvm::outs() << "Overriding IntegerType encoding...\n";
            dialectGroupName = StringLiteral("funky");
            writer.writeVarInt(/* IntegerType */ 999);
            writer.writeVarInt(type.getWidth() << 2 | type.getSignedness());
            return success();
          }
          return failure();
        });
    newCtx->appendDialectRegistry(op->getContext()->getDialectRegistry());
    newCtx->allowUnregisteredDialects();
    ParserConfig parseConfig(newCtx.get(), /*verifyAfterParse=*/true);
    parseConfig.getBytecodeReaderConfig().attachTypeCallback(
        [&](DialectBytecodeReader &reader, StringRef dialectName,
            Type &entry) -> LogicalResult {
          // Get test dialect version from the version map.
          auto versionOr = reader.getDialectVersion<test::TestDialect>();
          assert(succeeded(versionOr) && "expected reader to be able to access "
                                         "the version for test dialect");
          const auto *version =
              reinterpret_cast<const test::TestDialectVersion *>(*versionOr);
          if (version->major_ >= 2)
            return success();

          // `dialectName` is the name of the group we have the opportunity to
          // override. In this case, override only the dialect group "funky",
          // for which does not exist in memory.
          if (dialectName != StringLiteral("funky"))
            return success();

          uint64_t encoding;
          if (failed(reader.readVarInt(encoding)) || encoding != 999)
            return success();
          llvm::outs() << "Overriding parsing of IntegerType encoding...\n";
          uint64_t _widthAndSignedness, width;
          IntegerType::SignednessSemantics signedness;
          if (succeeded(reader.readVarInt(_widthAndSignedness)) &&
              ((width = _widthAndSignedness >> 2), true) &&
              ((signedness = static_cast<IntegerType::SignednessSemantics>(
                    _widthAndSignedness & 0x3)),
               true))
            entry = IntegerType::get(reader.getContext(), width, signedness);
          // Return nullopt to fall through the rest of the parsing code path.
          return success();
        });
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  // Test1: When writing bytecode, we override the encoding of TestI32Type with
  // the encoding of builtin IntegerType. We can natively parse this without
  // the use of a callback, relying on the existing builtin reader mechanism.
  void runTest1(Operation *op) {
    auto builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
    BytecodeDialectInterface *iface =
        builtin->getRegisteredInterface<BytecodeDialectInterface>();
    BytecodeWriterConfig writeConfig;
    writeConfig.attachTypeCallback(
        [&](Type entryValue, std::optional<StringRef> &dialectGroupName,
            DialectBytecodeWriter &writer) -> LogicalResult {
          // Emit TestIntegerType using the builtin dialect encoding.
          if (llvm::isa<test::TestI32Type>(entryValue)) {
            llvm::outs() << "Overriding TestI32Type encoding...\n";
            auto builtinI32Type =
                IntegerType::get(op->getContext(), 32,
                                 IntegerType::SignednessSemantics::Signless);
            // Specify that this type will need to be written as part of the
            // builtin group. This will override the default dialect group of
            // the attribute (test).
            dialectGroupName = StringLiteral("builtin");
            if (succeeded(iface->writeType(builtinI32Type, writer)))
              return success();
          }
          return failure();
        });
    // We natively parse the attribute as a builtin, so no callback needed.
    ParserConfig parseConfig(op->getContext(), /*verifyAfterParse=*/true);
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  // Test2: When writing bytecode, we write standard builtin IntegerTypes. At
  // parsing, we use the encoding of IntegerType to intercept all i32. Then,
  // instead of creating i32s, we assemble TestI32Type and return it.
  void runTest2(Operation *op) {
    auto builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
    BytecodeDialectInterface *iface =
        builtin->getRegisteredInterface<BytecodeDialectInterface>();
    BytecodeWriterConfig writeConfig;
    ParserConfig parseConfig(op->getContext(), /*verifyAfterParse=*/true);
    parseConfig.getBytecodeReaderConfig().attachTypeCallback(
        [&](DialectBytecodeReader &reader, StringRef dialectName,
            Type &entry) -> LogicalResult {
          if (dialectName != StringLiteral("builtin"))
            return success();
          Type builtinAttr = iface->readType(reader);
          if (auto integerType =
                  llvm::dyn_cast_or_null<IntegerType>(builtinAttr)) {
            if (integerType.getWidth() == 32 && integerType.isSignless()) {
              llvm::outs() << "Overriding parsing of TestI32Type encoding...\n";
              entry = test::TestI32Type::get(reader.getContext());
            }
          }
          return success();
        });
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  // Test3: When writing bytecode, we override the encoding of
  // TestAttrParamsAttr with the encoding of builtin DenseIntElementsAttr. We
  // can natively parse this without the use of a callback, relying on the
  // existing builtin reader mechanism.
  void runTest3(Operation *op) {
    auto builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
    BytecodeDialectInterface *iface =
        builtin->getRegisteredInterface<BytecodeDialectInterface>();
    auto i32Type = IntegerType::get(op->getContext(), 32,
                                    IntegerType::SignednessSemantics::Signless);
    BytecodeWriterConfig writeConfig;
    writeConfig.attachAttributeCallback(
        [&](Attribute entryValue, std::optional<StringRef> &dialectGroupName,
            DialectBytecodeWriter &writer) -> LogicalResult {
          // Emit TestIntegerType using the builtin dialect encoding.
          if (auto testParamAttrs =
                  llvm::dyn_cast<test::TestAttrParamsAttr>(entryValue)) {
            llvm::outs() << "Overriding TestAttrParamsAttr encoding...\n";
            // Specify that this attribute will need to be written as part of
            // the builtin group. This will override the default dialect group
            // of the attribute (test).
            dialectGroupName = StringLiteral("builtin");
            auto denseAttr = DenseIntElementsAttr::get(
                RankedTensorType::get({2}, i32Type),
                {testParamAttrs.getV0(), testParamAttrs.getV1()});
            if (succeeded(iface->writeAttribute(denseAttr, writer)))
              return success();
          }
          return failure();
        });
    // We natively parse the attribute as a builtin, so no callback needed.
    ParserConfig parseConfig(op->getContext(), /*verifyAfterParse=*/false);
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  // Test4: When writing bytecode, we write standard builtin
  // DenseIntElementsAttr. At parsing, we use the encoding of
  // DenseIntElementsAttr to intercept all ElementsAttr that have shaped type of
  // <2xi32>. Instead of assembling a DenseIntElementsAttr, we assemble
  // TestAttrParamsAttr and return it.
  void runTest4(Operation *op) {
    auto builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
    BytecodeDialectInterface *iface =
        builtin->getRegisteredInterface<BytecodeDialectInterface>();
    auto i32Type = IntegerType::get(op->getContext(), 32,
                                    IntegerType::SignednessSemantics::Signless);
    BytecodeWriterConfig writeConfig;
    ParserConfig parseConfig(op->getContext(), /*verifyAfterParse=*/false);
    parseConfig.getBytecodeReaderConfig().attachAttributeCallback(
        [&](DialectBytecodeReader &reader, StringRef dialectName,
            Attribute &entry) -> LogicalResult {
          // Override only the case where the return type of the builtin reader
          // is an i32 and fall through on all the other cases, since we want to
          // still use TestDialect normal codepath to parse the other types.
          Attribute builtinAttr = iface->readAttribute(reader);
          if (auto denseAttr =
                  llvm::dyn_cast_or_null<DenseIntElementsAttr>(builtinAttr)) {
            if (denseAttr.getType().getShape() == ArrayRef<int64_t>(2) &&
                denseAttr.getElementType() == i32Type) {
              llvm::outs()
                  << "Overriding parsing of TestAttrParamsAttr encoding...\n";
              int v0 = denseAttr.getValues<IntegerAttr>()[0].getInt();
              int v1 = denseAttr.getValues<IntegerAttr>()[1].getInt();
              entry =
                  test::TestAttrParamsAttr::get(reader.getContext(), v0, v1);
            }
          }
          return success();
        });
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  // Test5: When writing bytecode, we want TestDialect to use nothing else than
  // the builtin types and attributes and take full control of the encoding,
  // returning failure if any type or attribute is not part of builtin.
  void runTest5(Operation *op) {
    auto builtin = op->getContext()->getLoadedDialect<mlir::BuiltinDialect>();
    BytecodeDialectInterface *iface =
        builtin->getRegisteredInterface<BytecodeDialectInterface>();
    BytecodeWriterConfig writeConfig;
    writeConfig.attachAttributeCallback(
        [&](Attribute attr, std::optional<StringRef> &dialectGroupName,
            DialectBytecodeWriter &writer) -> LogicalResult {
          return iface->writeAttribute(attr, writer);
        });
    writeConfig.attachTypeCallback(
        [&](Type type, std::optional<StringRef> &dialectGroupName,
            DialectBytecodeWriter &writer) -> LogicalResult {
          return iface->writeType(type, writer);
        });
    ParserConfig parseConfig(op->getContext(), /*verifyAfterParse=*/false);
    parseConfig.getBytecodeReaderConfig().attachAttributeCallback(
        [&](DialectBytecodeReader &reader, StringRef dialectName,
            Attribute &entry) -> LogicalResult {
          Attribute builtinAttr = iface->readAttribute(reader);
          if (!builtinAttr)
            return failure();
          entry = builtinAttr;
          return success();
        });
    parseConfig.getBytecodeReaderConfig().attachTypeCallback(
        [&](DialectBytecodeReader &reader, StringRef dialectName,
            Type &entry) -> LogicalResult {
          Type builtinType = iface->readType(reader);
          if (!builtinType) {
            return failure();
          }
          entry = builtinType;
          return success();
        });
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  LogicalResult downgradeToVersion(Operation *op,
                                   const test::TestDialectVersion &version) {
    if ((version.major_ == 2) && (version.minor_ == 0))
      return success();
    if (version.major_ > 2 || (version.major_ == 2 && version.minor_ > 0)) {
      return op->emitError() << "current test dialect version is 2.0, "
                                "can't downgrade to version: "
                             << version.major_ << "." << version.minor_;
    }
    // Prior version 2.0, the old op supported only a single attribute called
    // "dimensions". We need to check that the modifier is false, otherwise we
    // can't do the downgrade.
    auto status = op->walk([&](test::TestVersionedOpA op) {
      auto &prop = op.getProperties();
      if (prop.modifier.getValue()) {
        op->emitOpError() << "cannot downgrade to version " << version.major_
                          << "." << version.minor_
                          << " since the modifier is not compatible";
        return WalkResult::interrupt();
      }
      llvm::outs() << "downgrading op...\n";
      return WalkResult::advance();
    });
    return failure(status.wasInterrupted());
  }

  // Test6: Downgrade IR to `targetVersion`, write to bytecode. Then, read and
  // upgrade IR when back in memory. The module is expected to be unmodified at
  // the end of the function.
  void runTest6(Operation *op) {
    test::TestDialectVersion targetEmissionVersion = targetVersion;

    // Downgrade IR constructs before writing the IR to bytecode.
    auto status = downgradeToVersion(op, targetEmissionVersion);
    assert(succeeded(status) && "expected the downgrade to succeed");
    (void)status;

    BytecodeWriterConfig writeConfig;
    writeConfig.setDialectVersion<test::TestDialect>(
        std::make_unique<test::TestDialectVersion>(targetEmissionVersion));
    ParserConfig parseConfig(op->getContext(), /*verifyAfterParse=*/true);
    doRoundtripWithConfigs(op, writeConfig, parseConfig);
  }

  test::TestDialect *testDialect;
};
} // namespace

namespace mlir {
void registerTestBytecodeRoundtripPasses() {
  PassRegistration<TestBytecodeRoundtripPass>();
}
} // namespace mlir
