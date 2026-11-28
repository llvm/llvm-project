//===- AdaptorTest.cpp - Adaptor unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeImplementation.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeReaderConfig.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <string>
#include <utility>

using namespace llvm;
using namespace mlir;

StringLiteral irWithResources = R"(
module @TestDialectResources attributes {
  bytecode.test = dense_resource<resource> : tensor<4xi32>
} {}
{-#
  dialect_resources: {
    builtin: {
      resource: "0x2000000001000000020000000300000004000000",
      resource_2: "0x2000000001000000020000000300000004000000"
    }
  }
#-}
)";

struct MockOstream final : public raw_ostream {
  std::unique_ptr<std::byte[]> buffer;
  size_t size = 0;

  MOCK_METHOD(void, reserveExtraSpace, (uint64_t extraSpace), (override));

  MockOstream() : raw_ostream(true) {}
  uint64_t current_pos() const override { return pos; }

private:
  size_t pos = 0;

  void write_impl(const char *ptr, size_t length) override {
    if (pos + length <= size) {
      memcpy((void *)(buffer.get() + pos), ptr, length);
      pos += length;
    } else {
      report_fatal_error(
          "Attempted to write past the end of the fixed size buffer.");
    }
  }
};

TEST(Bytecode, MultiModuleWithResource) {
  MLIRContext context;
  Builder builder(&context);
  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> module =
      parseSourceString<Operation *>(irWithResources, parseConfig);
  ASSERT_TRUE(module);

  // Write the module to bytecode.
  // Ensure that reserveExtraSpace is called with the size needed to write the
  // bytecode buffer.
  MockOstream ostream;
  EXPECT_CALL(ostream, reserveExtraSpace).WillOnce([&](uint64_t space) {
    ostream.buffer = std::make_unique<std::byte[]>(space);
    ostream.size = space;
  });
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module.get(), ostream)));

  // Create copy of buffer which is aligned to requested resource alignment.
  std::string buffer((char *)ostream.buffer.get(),
                     (char *)ostream.buffer.get() + ostream.size);
  constexpr size_t kAlignment = 0x20;
  size_t bufferSize = buffer.size();
  buffer.reserve(bufferSize + kAlignment - 1);
  size_t pad = (~(uintptr_t)buffer.data() + 1) & (kAlignment - 1);
  buffer.insert(0, pad, ' ');
  StringRef alignedBuffer(buffer.data() + pad, bufferSize);

  // Parse it back
  OwningOpRef<Operation *> roundTripModule =
      parseSourceString<Operation *>(alignedBuffer, parseConfig);
  ASSERT_TRUE(roundTripModule);

  // FIXME: Parsing external resources does not work on big-endian
  // platforms currently.
  if (llvm::endianness::native == llvm::endianness::big)
    GTEST_SKIP();

  // Try to see if we have a valid resource in the parsed module.
  auto checkResourceAttribute = [](Operation *parsedModule) {
    Attribute attr = parsedModule->getDiscardableAttr("bytecode.test");
    ASSERT_TRUE(attr);
    auto denseResourceAttr = dyn_cast<DenseI32ResourceElementsAttr>(attr);
    ASSERT_TRUE(denseResourceAttr);
    std::optional<ArrayRef<int32_t>> attrData =
        denseResourceAttr.tryGetAsArrayRef();
    ASSERT_TRUE(attrData.has_value());
    ASSERT_EQ(attrData->size(), static_cast<size_t>(4));
    EXPECT_EQ((*attrData)[0], 1);
    EXPECT_EQ((*attrData)[1], 2);
    EXPECT_EQ((*attrData)[2], 3);
    EXPECT_EQ((*attrData)[3], 4);
  };

  checkResourceAttribute(*module);
  checkResourceAttribute(*roundTripModule);
}

TEST(Bytecode, AlignmentFailure) {
  MLIRContext context;
  Builder builder(&context);
  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> module =
      parseSourceString<Operation *>(irWithResources, parseConfig);
  ASSERT_TRUE(module);
  // Ensure that the resource section needs an explicit alignment marker. The
  // test below exercises the reader's handling of that marker.
  module.get()->setDiscardableAttr("bytecode.padding",
                                   builder.getStringAttr("padding"));

  // Write the module to bytecode.
  std::string serializedBytecode;
  llvm::raw_string_ostream ostream(serializedBytecode);
  // Hardcode the producer to make sure that the length of the producer string
  // does not affect the alignment.
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module.get(), ostream,
                                            BytecodeWriterConfig("MLIR"))));

  // Create copy of buffer which is not aligned to requested resource alignment.
  std::string buffer(serializedBytecode);
  size_t bufferSize = buffer.size();

  // Increment into the buffer until we get to an address that is 2 byte aligned
  // but not 32 byte aligned.
  size_t pad = 0;
  while (true) {
    if (llvm::isAddrAligned(Align(2), buffer.data() + pad) &&
        !llvm::isAddrAligned(Align(32), buffer.data() + pad))
      break;

    pad++;
    // Pad the beginning of the buffer to push the start point to an unaligned
    // value.
    buffer.insert(0, 1, ' ');
  }

  StringRef alignedBuffer(buffer.data() + pad, bufferSize);

  // Attach a diagnostic handler to get the error message.
  llvm::SmallVector<std::string> msg;
  ScopedDiagnosticHandler handler(
      &context, [&msg](Diagnostic &diag) { msg.push_back(diag.str()); });

  // Parse it back
  OwningOpRef<Operation *> roundTripModule =
      parseSourceString<Operation *>(alignedBuffer, parseConfig);
  ASSERT_FALSE(roundTripModule);
  ASSERT_THAT(msg[0].data(), ::testing::StartsWith(
                                 "expected section alignment 32 but bytecode "
                                 "buffer"));
  ASSERT_STREQ(msg[1].data(), "failed to align section ID: 5");
}

namespace {
/// A custom operation for the purpose of showcasing how discardable attributes
/// are handled in absence of properties.
class OpWithoutProperties : public Op<OpWithoutProperties> {
public:
  // Begin boilerplate.
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWithoutProperties)
  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() {
    static StringRef attributeNames[] = {StringRef("inherent_attr")};
    return ArrayRef(attributeNames);
  };
  static StringRef getOperationName() {
    return "test_op_properties.op_without_properties";
  }
  // End boilerplate.
};

// A trivial supporting dialect to register the above operation.
class TestOpPropertiesDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOpPropertiesDialect)
  static constexpr StringLiteral getDialectNamespace() {
    return StringLiteral("test_op_properties");
  }
  explicit TestOpPropertiesDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TestOpPropertiesDialect>()) {
    addOperations<OpWithoutProperties>();
  }
};
} // namespace

constexpr StringLiteral withoutPropertiesAttrsSrc = R"mlir(
    "test_op_properties.op_without_properties"()
      {inherent_attr = 42, other_attr = 56} : () -> ()
)mlir";

TEST(Bytecode, OpWithoutProperties) {
  MLIRContext context;
  context.getOrLoadDialect<TestOpPropertiesDialect>();
  ParserConfig config(&context);
  OwningOpRef<Operation *> op =
      parseSourceString(withoutPropertiesAttrsSrc, config);

  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(op.get(), os)));
  std::unique_ptr<Block> block = std::make_unique<Block>();
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "string-buffer"), block.get(), config)));
  Operation *roundtripped = &block->front();
  EXPECT_EQ(roundtripped->getRawDictionaryAttrs().size(), 2u);
  EXPECT_EQ(roundtripped->getInherentAttr("inherent_attr"), std::nullopt);
  EXPECT_NE(roundtripped->getDiscardableAttr("inherent_attr"), Attribute());
  EXPECT_NE(roundtripped->getDiscardableAttr("other_attr"), Attribute());

  EXPECT_TRUE(OperationEquivalence::computeHash(op.get()) ==
              OperationEquivalence::computeHash(roundtripped));
}

TEST(Bytecode, DeepCallSiteLoc) {
  MLIRContext context;
  ParserConfig config(&context);

  // Create a deep CallSiteLoc chain to test iterative parsing.
  Location baseLoc = FileLineColLoc::get(&context, "test.mlir", 1, 1);
  Location loc = baseLoc;
  constexpr int kDepth = 1000;
  for (int i = 0; i < kDepth; ++i) {
    loc = CallSiteLoc::get(loc, baseLoc);
  }

  // Create a simple module with the deep location.
  Builder builder(&context);
  OwningOpRef<ModuleOp> module =
      ModuleOp::create(loc, /*attributes=*/std::nullopt);
  ASSERT_TRUE(module);

  // Write to bytecode.
  std::string bytecode;
  llvm::raw_string_ostream os(bytecode);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module.get(), os)));

  // Parse it back using the bytecode reader.
  std::unique_ptr<Block> block = std::make_unique<Block>();
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "string-buffer"), block.get(), config)));

  // Verify we got the roundtripped module.
  ASSERT_FALSE(block->empty());
  Operation *roundTripped = &block->front();

  // Verify the location matches.
  EXPECT_EQ(module.get()->getLoc(), roundTripped->getLoc());
}

// Regression test for https://github.com/llvm/llvm-project/issues/99626.
// FusedLoc::get(context, locs) returns UnknownLoc when locs is empty, so the
// bytecode reader must not use cast<FusedLoc>() on that result.
TEST(Bytecode, EmptyFusedLocRoundtrip) {
  MLIRContext context;

  // FusedLoc with empty locations (no metadata). FusedLoc::get returns
  // UnknownLoc in this case, but the bytecode writer stores a FusedLoc.
  FusedLoc fusedLoc = FusedLoc::get(&context, /*locs=*/{}, /*metadata=*/{});
  auto module = mlir::ModuleOp::create(fusedLoc, "test");

  // Write the module to bytecode.
  std::string buffer;
  llvm::raw_string_ostream ostream(buffer);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module, ostream)));
  ostream.flush();

  // Parse it back - this used to crash with an assertion failure in cast<>.
  ParserConfig parseConfig(&context);
  OwningOpRef<Operation *> roundTripModule =
      parseSourceString<Operation *>(buffer, parseConfig);
  ASSERT_TRUE(roundTripModule);

  module.erase();
}

TEST(Bytecode, LocationElision) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  ParserConfig config(&context);

  // Module 1: Reuses the same location "foo" everywhere.
  StringRef ir1 = R"mlir(
    module @Test {
      "test.op"() : () -> () loc("foo")
    } loc("foo")
  )mlir";

  // Module 2: Uses unique locations everywhere.
  StringRef ir2 = R"mlir(
    module @Test {
      "test.op"() : () -> () loc("a")
    } loc("b")
  )mlir";

  OwningOpRef<Operation *> op1 = parseSourceString(ir1, config);
  OwningOpRef<Operation *> op2 = parseSourceString(ir2, config);
  ASSERT_TRUE(op1);
  ASSERT_TRUE(op2);

  // Serialize both with location elision enabled.
  BytecodeWriterConfig writerConfig;
  writerConfig.setElideLocations(true);

  std::string bytecode1;
  {
    llvm::raw_string_ostream os(bytecode1);
    ASSERT_TRUE(succeeded(writeBytecodeToFile(op1.get(), os, writerConfig)));
  }

  std::string bytecode2;
  {
    llvm::raw_string_ostream os(bytecode2);
    ASSERT_TRUE(succeeded(writeBytecodeToFile(op2.get(), os, writerConfig)));
  }

  // If location elision is working correctly, both modules must produce
  // the EXACT same bytecode representation, because all locations (shared or
  // unique) will have been collapsed into a single shared UnknownLoc.
  EXPECT_EQ(bytecode1, bytecode2);
}

TEST(Bytecode, LocationElisionPreservesAttributes) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  ParserConfig config(&context);

  // An operation with a debug location ("elide_me") AND a semantic attribute
  // that is a LocationAttr ("preserve_me").
  StringRef ir = R"mlir(
    module @Test {
      "test.op"() {some_loc_attr = loc("preserve_me")} : () -> () loc("elide_me")
    } loc("elide_me")
  )mlir";

  OwningOpRef<Operation *> op = parseSourceString(ir, config);
  ASSERT_TRUE(op);

  // Serialize with location elision enabled.
  BytecodeWriterConfig writerConfig;
  writerConfig.setElideLocations(true);

  std::string bytecode;
  {
    llvm::raw_string_ostream os(bytecode);
    ASSERT_TRUE(succeeded(writeBytecodeToFile(op.get(), os, writerConfig)));
  }

  // Parse it back using the bytecode reader.
  std::unique_ptr<Block> block = std::make_unique<Block>();
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "string-buffer"), block.get(), config)));

  // Verify we got the roundtripped module.
  ASSERT_FALSE(block->empty());
  Operation *roundTrippedModule = &block->front();
  ASSERT_TRUE(roundTrippedModule);

  // Find the inner "test.op" operation.
  Operation *innerOp = nullptr;
  roundTrippedModule->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "test.op") {
      innerOp = op;
    }
  });
  ASSERT_TRUE(innerOp);

  // 1. Verify that the debug location of "test.op" WAS elided (became
  // UnknownLoc).
  EXPECT_TRUE(isa<UnknownLoc>(innerOp->getLoc()));

  // 2. Verify that the semantic location attribute WAS PRESERVED.
  Attribute semanticLocAttr = innerOp->getDiscardableAttr("some_loc_attr");
  ASSERT_TRUE(semanticLocAttr);
  auto locAttr = dyn_cast<LocationAttr>(semanticLocAttr);
  ASSERT_TRUE(locAttr);

  // It should still be loc("preserve_me"), not UnknownLoc.
  EXPECT_FALSE(isa<UnknownLoc>(locAttr));

  auto nameLoc = dyn_cast<NameLoc>(locAttr);
  ASSERT_TRUE(nameLoc);
  EXPECT_EQ(nameLoc.getName().getValue(), "preserve_me");
}

namespace {
struct MutableBoxTypeStorage : public TypeStorage {
  using KeyTy = StringRef;

  explicit MutableBoxTypeStorage(StringRef name) : name(name) {}

  bool operator==(const KeyTy &key) const { return name == key; }

  static MutableBoxTypeStorage *construct(TypeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<MutableBoxTypeStorage>())
        MutableBoxTypeStorage(allocator.copyInto(key));
  }

  LogicalResult mutate(TypeStorageAllocator &, Type newBody) {
    if (body && body != newBody)
      return failure();
    body = newBody;
    return success();
  }

  StringRef name;
  Type body;
};

class MutableBoxType
    : public Type::TypeBase<MutableBoxType, Type, MutableBoxTypeStorage,
                            TypeTrait::IsMutable> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MutableBoxType)
  using Base::Base;
  static constexpr StringLiteral name = "test_mutable_bc.box";

  static MutableBoxType get(MLIRContext *ctx, StringRef name) {
    return Base::get(ctx, name);
  }

  StringRef getName() const { return getImpl()->name; }
  Type getBody() const { return getImpl()->body; }
  LogicalResult setBody(Type body) { return Base::mutate(body); }
};

struct VerifiedBoxAttrStorage : public AttributeStorage {
  using KeyTy = std::pair<MutableBoxType, Attribute>;

  VerifiedBoxAttrStorage(MutableBoxType box, Attribute extra)
      : box(box), extra(extra) {}

  bool operator==(const KeyTy &key) const {
    return box == key.first && extra == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static VerifiedBoxAttrStorage *construct(AttributeStorageAllocator &allocator,
                                           const KeyTy &key) {
    return new (allocator.allocate<VerifiedBoxAttrStorage>())
        VerifiedBoxAttrStorage(key.first, key.second);
  }

  MutableBoxType box;
  Attribute extra;
};

class VerifiedBoxAttr : public Attribute::AttrBase<VerifiedBoxAttr, Attribute,
                                                   VerifiedBoxAttrStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifiedBoxAttr)
  using Base::Base;
  static constexpr StringLiteral name = "test_mutable_bc.verified_box";

  static VerifiedBoxAttr get(MLIRContext *ctx, MutableBoxType box,
                             Attribute extra = {}) {
    return Base::get(ctx, box, extra);
  }

  MutableBoxType getBox() const { return getImpl()->box; }
  Attribute getExtra() const { return getImpl()->extra; }
};

struct TestMutableBytecodeInterface : public BytecodeDialectInterface {
  using BytecodeDialectInterface::BytecodeDialectInterface;

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    auto box = dyn_cast<MutableBoxType>(type);
    if (!box)
      return failure();
    writer.writeOwnedString(box.getName());
    Type body = box.getBody();
    writer.writeOwnedBool(static_cast<bool>(body));
    if (body)
      writer.writeType(body);
    return success();
  }

  Type readType(DialectBytecodeReader &reader) const override {
    StringRef name;
    bool hasBody = false;
    if (failed(reader.readString(name)) || failed(reader.readBool(hasBody)))
      return Type();

    auto box = MutableBoxType::get(getContext(), name);
    if (!hasBody)
      return box;

    // Names starting with "unguarded_" intentionally omit tryStartCyclicRead
    // to exercise cycle detection in the bytecode reader.
    if (!name.starts_with("unguarded_")) {
      if (failed(reader.tryStartCyclicRead(box)))
        return Type();
    }

    Type body;
    if (failed(reader.readType(body)))
      return Type();
    if (failed(box.setBody(body))) {
      reader.emitError("mutable box '")
          << name << "' redefined with a different body";
      return Type();
    }

    if (name == "mismatch_return")
      return MutableBoxType::get(getContext(), "other_instance");
    return box;
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    auto verified = dyn_cast<VerifiedBoxAttr>(attr);
    if (!verified)
      return failure();
    writer.writeVarInt(0);
    writer.writeType(verified.getBox());
    writer.writeOptionalAttribute(verified.getExtra());
    return success();
  }

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    uint64_t code = 0;
    if (failed(reader.readVarInt(code)) || code != 0)
      return Attribute();
    MutableBoxType box;
    Attribute extra;
    if (failed(reader.readType(box)) ||
        failed(reader.readOptionalAttribute(extra)))
      return Attribute();
    if (!box.getBody()) {
      reader.emitError("mutable box body is incomplete when verifying "
                       "enclosing attribute");
      return Attribute();
    }
    return VerifiedBoxAttr::get(getContext(), box, extra);
  }
};

class TestMutableBytecodeDialect : public Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestMutableBytecodeDialect)
  static constexpr StringLiteral getDialectNamespace() {
    return StringLiteral("test_mutable_bc");
  }
  explicit TestMutableBytecodeDialect(MLIRContext *context)
      : Dialect(getDialectNamespace(), context,
                TypeID::get<TestMutableBytecodeDialect>()) {
    addTypes<MutableBoxType>();
    addAttributes<VerifiedBoxAttr>();
    addInterfaces<TestMutableBytecodeInterface>();
  }

  void printType(Type type, DialectAsmPrinter &os) const override {
    auto box = cast<MutableBoxType>(type);
    if (box.getName().starts_with("print_needs_body_") && !box.getBody())
      llvm_unreachable("printed incomplete mutable type before body was set");
    os << "box<" << box.getName() << ">";
  }

  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override {
    os << "verified_box<" << cast<VerifiedBoxAttr>(attr).getBox().getName()
       << ">";
  }
};
} // namespace

static Type wrapInTuples(MLIRContext *ctx, Type inner, unsigned depth) {
  Type current = inner;
  for (unsigned i = 0; i < depth; ++i)
    current = TupleType::get(ctx, TypeRange{current});
  return current;
}

static Attribute wrapInArrays(MLIRContext *ctx, Attribute inner,
                              unsigned depth) {
  Attribute current = inner;
  for (unsigned i = 0; i < depth; ++i)
    current = ArrayAttr::get(ctx, {current});
  return current;
}

static void serializeModule(ModuleOp module, std::string &bytecode) {
  bytecode.clear();
  llvm::raw_string_ostream os(bytecode);
  ASSERT_TRUE(succeeded(writeBytecodeToFile(module, os)));
}

static void expectReadDiagnostic(
    StringRef bytecode, StringRef expectedSubstr,
    llvm::function_ref<void(MLIRContext &, ParserConfig &)> setup = {}) {
  MLIRContext dstCtx;
  dstCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
  SmallVector<std::string> diagnostics;
  ScopedDiagnosticHandler handler(&dstCtx, [&](Diagnostic &diag) {
    diagnostics.push_back(diag.str());
    return success();
  });
  ParserConfig config(&dstCtx);
  if (setup)
    setup(dstCtx, config);
  Block block;
  EXPECT_TRUE(failed(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "diagnostic-buffer"), &block, config)));
  ASSERT_FALSE(diagnostics.empty());
  EXPECT_THAT(diagnostics[0], ::testing::HasSubstr(expectedSubstr));
}

TEST(Bytecode, VerifiedAttrOverDeepMutableType) {
  MLIRContext srcCtx;
  srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
  Builder srcBuilder(&srcCtx);

  // Non-cyclic mutable types at depth 7 and depth 14 (> 2 * maxAttrTypeDepth),
  // plus a self-recursive mutable type through 14 tuples, paired with a deep
  // optional attribute to exercise readOptionalAttribute deferral.
  auto deep7 = MutableBoxType::get(&srcCtx, "deep7");
  ASSERT_TRUE(succeeded(
      deep7.setBody(wrapInTuples(&srcCtx, srcBuilder.getI32Type(), 7))));

  auto deep14 = MutableBoxType::get(&srcCtx, "deep14");
  ASSERT_TRUE(succeeded(
      deep14.setBody(wrapInTuples(&srcCtx, srcBuilder.getI64Type(), 14))));

  auto rec14 = MutableBoxType::get(&srcCtx, "rec14");
  ASSERT_TRUE(succeeded(rec14.setBody(wrapInTuples(&srcCtx, rec14, 14))));

  Attribute deepOptAttr =
      wrapInArrays(&srcCtx, srcBuilder.getI32IntegerAttr(42), 10);

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
  (*module)->setDiscardableAttr("test_mutable_bc.attr7",
                                VerifiedBoxAttr::get(&srcCtx, deep7));
  (*module)->setDiscardableAttr(
      "test_mutable_bc.attr14",
      VerifiedBoxAttr::get(&srcCtx, deep14, deepOptAttr));
  (*module)->setDiscardableAttr(
      "test_mutable_bc.attr_rec14",
      wrapInArrays(&srcCtx, VerifiedBoxAttr::get(&srcCtx, rec14), 6));

  std::string bytecode;
  serializeModule(module.get(), bytecode);

  MLIRContext dstCtx;
  dstCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
  Builder dstBuilder(&dstCtx);
  ParserConfig config(&dstCtx);
  Block block;
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "deep-mutable-buffer"), &block, config)));
  ASSERT_FALSE(block.empty());

  Operation *roundTripped = &block.front();
  auto attr7 = dyn_cast_or_null<VerifiedBoxAttr>(
      roundTripped->getDiscardableAttr("test_mutable_bc.attr7"));
  ASSERT_TRUE(attr7);
  EXPECT_EQ(attr7.getBox().getName(), "deep7");
  EXPECT_EQ(attr7.getBox().getBody(),
            wrapInTuples(&dstCtx, dstBuilder.getI32Type(), 7));

  auto attr14 = dyn_cast_or_null<VerifiedBoxAttr>(
      roundTripped->getDiscardableAttr("test_mutable_bc.attr14"));
  ASSERT_TRUE(attr14);
  EXPECT_EQ(attr14.getBox().getName(), "deep14");
  EXPECT_EQ(attr14.getBox().getBody(),
            wrapInTuples(&dstCtx, dstBuilder.getI64Type(), 14));
  EXPECT_EQ(attr14.getExtra(),
            wrapInArrays(&dstCtx, dstBuilder.getI32IntegerAttr(42), 10));

  Attribute curAttr =
      roundTripped->getDiscardableAttr("test_mutable_bc.attr_rec14");
  for (unsigned i = 0; i < 6; ++i) {
    auto arr = dyn_cast_or_null<ArrayAttr>(curAttr);
    ASSERT_TRUE(arr);
    ASSERT_EQ(arr.size(), 1u);
    curAttr = arr[0];
  }
  auto parsedRec14Attr = dyn_cast_or_null<VerifiedBoxAttr>(curAttr);
  ASSERT_TRUE(parsedRec14Attr);
  MutableBoxType parsedRec14Box = parsedRec14Attr.getBox();
  EXPECT_EQ(parsedRec14Box.getName(), "rec14");
  Type curType = parsedRec14Box.getBody();
  for (unsigned i = 0; i < 14; ++i) {
    auto tup = dyn_cast_or_null<TupleType>(curType);
    ASSERT_TRUE(tup);
    ASSERT_EQ(tup.size(), 1u);
    curType = tup.getType(0);
  }
  EXPECT_EQ(curType, parsedRec14Box);
}

TEST(Bytecode, MutableTypeReachedThroughImmutableWrapper) {
  MLIRContext srcCtx;
  srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();

  // Mutable type first reached through the immutable TupleType that also
  // appears in its body (the pointer-to-self-referential-record pattern).
  auto wrapped = MutableBoxType::get(&srcCtx, "wrapped");
  Type wrappedTuple = TupleType::get(&srcCtx, TypeRange{wrapped});
  ASSERT_TRUE(succeeded(wrapped.setBody(wrappedTuple)));

  // Same pattern wrapped past maxAttrTypeDepth.
  auto deepWrapped = MutableBoxType::get(&srcCtx, "deep_wrapped");
  Type deepWrappedTuple = wrapInTuples(&srcCtx, deepWrapped, 7);
  ASSERT_TRUE(succeeded(deepWrapped.setBody(deepWrappedTuple)));

  // Two mutable types sharing one immutable wrapper, exercising multiple
  // guarded re-entries of the same wrapper across distinct live partials.
  auto sharedA = MutableBoxType::get(&srcCtx, "shared_a");
  auto sharedB = MutableBoxType::get(&srcCtx, "shared_b");
  Type sharedTuple = TupleType::get(&srcCtx, TypeRange{sharedA, sharedB});
  ASSERT_TRUE(succeeded(sharedA.setBody(sharedTuple)));
  ASSERT_TRUE(succeeded(sharedB.setBody(sharedTuple)));

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
  (*module)->setDiscardableAttr("test_mutable_bc.wrapped",
                                TypeAttr::get(wrappedTuple));
  (*module)->setDiscardableAttr("test_mutable_bc.deep_wrapped",
                                TypeAttr::get(deepWrappedTuple));
  (*module)->setDiscardableAttr("test_mutable_bc.shared_wrapped",
                                TypeAttr::get(sharedTuple));

  std::string bytecode;
  serializeModule(module.get(), bytecode);

  MLIRContext dstCtx;
  dstCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
  ParserConfig config(&dstCtx);
  // Attach a fallthrough callback on builtin types to confirm guarded re-entry
  // does not re-enter callbacks on an active Entry<T>.
  config.getBytecodeReaderConfig().attachTypeCallback(
      [](DialectBytecodeReader &, StringRef, Type &entry) -> LogicalResult {
        entry = Type();
        return success();
      });
  Block block;
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "wrapped-buffer"), &block, config)));
  ASSERT_FALSE(block.empty());

  Operation *roundTripped = &block.front();

  auto wrappedAttr = dyn_cast_or_null<TypeAttr>(
      roundTripped->getDiscardableAttr("test_mutable_bc.wrapped"));
  ASSERT_TRUE(wrappedAttr);
  auto parsedWrappedTuple = dyn_cast<TupleType>(wrappedAttr.getValue());
  ASSERT_TRUE(parsedWrappedTuple);
  ASSERT_EQ(parsedWrappedTuple.size(), 1u);
  auto parsedWrapped = dyn_cast<MutableBoxType>(parsedWrappedTuple.getType(0));
  ASSERT_TRUE(parsedWrapped);
  EXPECT_EQ(parsedWrapped.getName(), "wrapped");
  EXPECT_EQ(parsedWrapped.getBody(), parsedWrappedTuple);

  auto deepAttr = dyn_cast_or_null<TypeAttr>(
      roundTripped->getDiscardableAttr("test_mutable_bc.deep_wrapped"));
  ASSERT_TRUE(deepAttr);
  Type curType = deepAttr.getValue();
  for (unsigned i = 0; i < 7; ++i) {
    auto tup = dyn_cast_or_null<TupleType>(curType);
    ASSERT_TRUE(tup);
    ASSERT_EQ(tup.size(), 1u);
    curType = tup.getType(0);
  }
  auto parsedDeep = dyn_cast_or_null<MutableBoxType>(curType);
  ASSERT_TRUE(parsedDeep);
  EXPECT_EQ(parsedDeep.getName(), "deep_wrapped");
  EXPECT_EQ(parsedDeep.getBody(), deepAttr.getValue());

  auto sharedAttr = dyn_cast_or_null<TypeAttr>(
      roundTripped->getDiscardableAttr("test_mutable_bc.shared_wrapped"));
  ASSERT_TRUE(sharedAttr);
  auto parsedSharedTuple = dyn_cast<TupleType>(sharedAttr.getValue());
  ASSERT_TRUE(parsedSharedTuple);
  ASSERT_EQ(parsedSharedTuple.size(), 2u);
  auto parsedA = dyn_cast<MutableBoxType>(parsedSharedTuple.getType(0));
  auto parsedB = dyn_cast<MutableBoxType>(parsedSharedTuple.getType(1));
  ASSERT_TRUE(parsedA);
  ASSERT_TRUE(parsedB);
  EXPECT_EQ(parsedA.getName(), "shared_a");
  EXPECT_EQ(parsedB.getName(), "shared_b");
  EXPECT_EQ(parsedA.getBody(), parsedSharedTuple);
  EXPECT_EQ(parsedB.getBody(), parsedSharedTuple);
}

TEST(Bytecode, MutableTypeCallbackFallthroughAndMismatch) {
  MLIRContext srcCtx;
  srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
  auto rec = MutableBoxType::get(&srcCtx, "cb_box");
  ASSERT_TRUE(succeeded(rec.setBody(rec)));

  auto deepRec = MutableBoxType::get(&srcCtx, "deep_cb_box");
  ASSERT_TRUE(succeeded(deepRec.setBody(wrapInTuples(&srcCtx, deepRec, 7))));

  auto nestedRec = MutableBoxType::get(&srcCtx, "nested_cb_box");
  ASSERT_TRUE(
      succeeded(nestedRec.setBody(wrapInTuples(&srcCtx, nestedRec, 1))));

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
  (*module)->setDiscardableAttr("test_mutable_bc.type_ref", TypeAttr::get(rec));
  (*module)->setDiscardableAttr("test_mutable_bc.deep_type_ref",
                                TypeAttr::get(deepRec));
  (*module)->setDiscardableAttr("test_mutable_bc.nested_type_ref",
                                TypeAttr::get(nestedRec));

  std::string bytecode;
  serializeModule(module.get(), bytecode);

  // A callback that publishes a speculative partial and falls through must have
  // its partial rolled back and must not re-run when a deep body defers.
  {
    MLIRContext dstCtx;
    dstCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
    ParserConfig config(&dstCtx);
    config.getBytecodeReaderConfig().attachTypeCallback(
        [](DialectBytecodeReader &reader, StringRef dialectName,
           Type &entry) -> LogicalResult {
          if (dialectName != "test_mutable_bc")
            return success();
          auto speculative =
              MutableBoxType::get(reader.getContext(), "speculative");
          if (failed(reader.tryStartCyclicRead(speculative)))
            return failure();
          entry = Type();
          return success();
        });
    Block block;
    ASSERT_TRUE(succeeded(readBytecodeFile(
        llvm::MemoryBufferRef(bytecode, "cb-fallthrough"), &block, config)));
    ASSERT_FALSE(block.empty());

    Operation *roundTripped = &block.front();
    auto parsedTypeAttr = dyn_cast_or_null<TypeAttr>(
        roundTripped->getDiscardableAttr("test_mutable_bc.type_ref"));
    ASSERT_TRUE(parsedTypeAttr);
    auto parsedBox = dyn_cast<MutableBoxType>(parsedTypeAttr.getValue());
    ASSERT_TRUE(parsedBox);
    EXPECT_EQ(parsedBox.getName(), "cb_box");
    EXPECT_EQ(parsedBox.getBody(), parsedBox);

    auto parsedDeepAttr = dyn_cast_or_null<TypeAttr>(
        roundTripped->getDiscardableAttr("test_mutable_bc.deep_type_ref"));
    ASSERT_TRUE(parsedDeepAttr);
    auto parsedDeepBox = dyn_cast<MutableBoxType>(parsedDeepAttr.getValue());
    ASSERT_TRUE(parsedDeepBox);
    EXPECT_EQ(parsedDeepBox.getName(), "deep_cb_box");
    EXPECT_EQ(parsedDeepBox.getBody(), wrapInTuples(&dstCtx, parsedDeepBox, 7));
  }

  // A callback that publishes a speculative partial, resolves a nested entry
  // that captures that partial, and then falls through must fail rather than
  // leaving a poisoned cached entry behind.
  expectReadDiagnostic(
      bytecode,
      "custom type reader callback published a partial type that may have "
      "escaped into a resolved nested entry before falling through",
      [](MLIRContext &, ParserConfig &config) {
        config.getBytecodeReaderConfig().attachTypeCallback(
            [](DialectBytecodeReader &reader, StringRef dialectName,
               Type &entry) -> LogicalResult {
              if (dialectName != "test_mutable_bc")
                return success();
              StringRef name;
              bool hasBody = false;
              if (failed(reader.readString(name)) ||
                  failed(reader.readBool(hasBody)) || name != "nested_cb_box") {
                entry = Type();
                return success();
              }
              auto speculative =
                  MutableBoxType::get(reader.getContext(), "escaped_spec");
              Type body;
              if (failed(reader.tryStartCyclicRead(speculative)) ||
                  failed(reader.readType(body)))
                return failure();
              entry = Type();
              return success();
            });
      });

  // A callback that publishes one instance and returns a different instance
  // must fail with a contract-violation diagnostic.
  expectReadDiagnostic(
      bytecode,
      "custom type reader for dialect 'test_mutable_bc' returned a different "
      "type than the one published with tryStartCyclicRead",
      [](MLIRContext &, ParserConfig &config) {
        config.getBytecodeReaderConfig().attachTypeCallback(
            [](DialectBytecodeReader &reader, StringRef dialectName,
               Type &entry) -> LogicalResult {
              if (dialectName != "test_mutable_bc")
                return success();
              StringRef name;
              bool hasBody = false;
              Type body;
              auto published =
                  MutableBoxType::get(reader.getContext(), "first");
              if (failed(reader.readString(name)) ||
                  failed(reader.readBool(hasBody)) ||
                  failed(reader.tryStartCyclicRead(published)) ||
                  failed(reader.readType(body)))
                return failure();
              entry = MutableBoxType::get(reader.getContext(), "second");
              return success();
            });
      });

  // A dialect interface that publishes one instance and returns a different
  // instance must also fail with the contract-violation diagnostic.
  {
    MLIRContext badSrcCtx;
    badSrcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
    auto mismatch = MutableBoxType::get(&badSrcCtx, "mismatch_return");
    ASSERT_TRUE(succeeded(mismatch.setBody(mismatch)));
    OwningOpRef<ModuleOp> badModule =
        ModuleOp::create(UnknownLoc::get(&badSrcCtx));
    (*badModule)
        ->setDiscardableAttr("test_mutable_bc.type_ref",
                             TypeAttr::get(mismatch));
    std::string badBytecode;
    serializeModule(badModule.get(), badBytecode);

    expectReadDiagnostic(badBytecode,
                         "custom type reader for dialect 'test_mutable_bc' "
                         "returned a different type than the one published "
                         "with tryStartCyclicRead");
  }

  // Reading a mutable type into a context that already holds a conflicting body
  // for the same key must emit an explicit redefinition diagnostic.
  expectReadDiagnostic(
      bytecode, "mutable box 'cb_box' redefined with a different body",
      [](MLIRContext &dstCtx, ParserConfig &) {
        auto existing = MutableBoxType::get(&dstCtx, "cb_box");
        LogicalResult status = existing.setBody(IntegerType::get(&dstCtx, 64));
        assert(succeeded(status));
        (void)status;
      });
}

TEST(Bytecode, UnguardedCyclicTypeFailsCleanly) {
  // Direct self-cycle without tryStartCyclicRead (caught via activeEntries).
  {
    MLIRContext srcCtx;
    srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
    auto self = MutableBoxType::get(&srcCtx, "unguarded_self");
    ASSERT_TRUE(succeeded(self.setBody(self)));

    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
    (*module)->setDiscardableAttr("test_mutable_bc.bad", TypeAttr::get(self));

    std::string bytecode;
    serializeModule(module.get(), bytecode);
    expectReadDiagnostic(bytecode, "cyclic dependency detected in type");
  }

  // An unguarded self-cycle that also references a guarded mutable peer whose
  // body points back to it must still fail cleanly rather than spinning on the
  // peer's live partial.
  {
    MLIRContext srcCtx;
    srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
    auto unguarded = MutableBoxType::get(&srcCtx, "unguarded_pair");
    auto guardedPeer = MutableBoxType::get(&srcCtx, "guarded_peer");
    ASSERT_TRUE(succeeded(guardedPeer.setBody(unguarded)));
    ASSERT_TRUE(succeeded(unguarded.setBody(
        TupleType::get(&srcCtx, TypeRange{guardedPeer, unguarded}))));

    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
    (*module)->setDiscardableAttr("test_mutable_bc.bad_pair",
                                  TypeAttr::get(unguarded));

    std::string bytecode;
    serializeModule(module.get(), bytecode);
    expectReadDiagnostic(bytecode, "cyclic dependency detected in type");
  }

  // Cycle spanning 8 entries (> maxAttrTypeDepth) without tryStartCyclicRead
  // (caught via worklist progress epoch check rather than hanging), both with
  // and without a speculative fallthrough callback.
  {
    MLIRContext srcCtx;
    srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
    SmallVector<MutableBoxType> ring;
    for (unsigned i = 0; i < 8; ++i)
      ring.push_back(
          MutableBoxType::get(&srcCtx, ("unguarded_" + Twine(i)).str()));
    for (unsigned i = 0; i < 8; ++i)
      ASSERT_TRUE(succeeded(ring[i].setBody(ring[(i + 1) % 8])));

    OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
    (*module)->setDiscardableAttr("test_mutable_bc.bad_ring",
                                  TypeAttr::get(ring[0]));

    std::string bytecode;
    serializeModule(module.get(), bytecode);
    expectReadDiagnostic(bytecode, "cyclic dependency detected in type");

    expectReadDiagnostic(
        bytecode, "cyclic dependency detected in type",
        [](MLIRContext &, ParserConfig &config) {
          config.getBytecodeReaderConfig().attachTypeCallback(
              [](DialectBytecodeReader &reader, StringRef dialectName,
                 Type &entry) -> LogicalResult {
                if (dialectName != "test_mutable_bc")
                  return success();
                auto speculative =
                    MutableBoxType::get(reader.getContext(), "speculative");
                if (failed(reader.tryStartCyclicRead(speculative)))
                  return failure();
                entry = Type();
                return success();
              });
        });
  }
}

TEST(Bytecode, TryStartCyclicReadPreconditionDiagnostics) {
  MLIRContext srcCtx;
  srcCtx.getOrLoadDialect<TestMutableBytecodeDialect>();
  auto plain = MutableBoxType::get(&srcCtx, "plain");
  ASSERT_TRUE(succeeded(plain.setBody(IntegerType::get(&srcCtx, 32))));

  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&srcCtx));
  (*module)->setDiscardableAttr("test_mutable_bc.attr",
                                VerifiedBoxAttr::get(&srcCtx, plain));

  std::string bytecode;
  serializeModule(module.get(), bytecode);

  // Called while reading an attribute entry instead of a custom type entry.
  expectReadDiagnostic(
      bytecode,
      "tryStartCyclicRead called while not reading a custom type entry",
      [](MLIRContext &, ParserConfig &config) {
        config.getBytecodeReaderConfig().attachAttributeCallback(
            [](DialectBytecodeReader &reader, StringRef dialectName,
               Attribute &) -> LogicalResult {
              if (dialectName != "test_mutable_bc")
                return success();
              auto box = MutableBoxType::get(reader.getContext(), "wrong_ctx");
              return reader.tryStartCyclicRead(box);
            });
      });

  // Called with a null Type.
  expectReadDiagnostic(bytecode, "cannot start cyclic read with null type",
                       [](MLIRContext &, ParserConfig &config) {
                         config.getBytecodeReaderConfig().attachTypeCallback(
                             [](DialectBytecodeReader &reader,
                                StringRef dialectName,
                                Type &) -> LogicalResult {
                               if (dialectName != "test_mutable_bc")
                                 return success();
                               return reader.tryStartCyclicRead(Type());
                             });
                       });

  // Called with an immutable Type.
  expectReadDiagnostic(bytecode, "cannot start cyclic read on immutable type",
                       [](MLIRContext &, ParserConfig &config) {
                         config.getBytecodeReaderConfig().attachTypeCallback(
                             [](DialectBytecodeReader &reader,
                                StringRef dialectName,
                                Type &) -> LogicalResult {
                               if (dialectName != "test_mutable_bc")
                                 return success();
                               return reader.tryStartCyclicRead(
                                   IntegerType::get(reader.getContext(), 32));
                             });
                       });

  // Called twice for the same entry with distinct mutable Type instances; uses
  // "print_needs_body_" so formatting the incomplete published instance would
  // trigger llvm_unreachable.
  expectReadDiagnostic(
      bytecode, "already published a different 'test_mutable_bc.box' instance",
      [](MLIRContext &, ParserConfig &config) {
        config.getBytecodeReaderConfig().attachTypeCallback(
            [](DialectBytecodeReader &reader, StringRef dialectName,
               Type &) -> LogicalResult {
              if (dialectName != "test_mutable_bc")
                return success();
              auto first = MutableBoxType::get(reader.getContext(),
                                               "print_needs_body_first");
              auto second = MutableBoxType::get(reader.getContext(), "second");
              if (failed(reader.tryStartCyclicRead(first)))
                return failure();
              return reader.tryStartCyclicRead(second);
            });
      });
}
