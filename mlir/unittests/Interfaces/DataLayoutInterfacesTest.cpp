//===- DataLayoutInterfacesTest.cpp - Unit Tests for Data Layouts ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {
constexpr static llvm::StringLiteral kAttrName = "dltest.layout";

/// Trivial array storage for the custom data layout spec attribute, just a list
/// of entries.
class DataLayoutSpecStorage : public AttributeStorage {
public:
  using KeyTy = ArrayRef<DataLayoutEntryInterface>;

  DataLayoutSpecStorage(ArrayRef<DataLayoutEntryInterface> entries)
      : entries(entries) {}

  bool operator==(const KeyTy &key) const { return key == entries; }

  static DataLayoutSpecStorage *construct(AttributeStorageAllocator &allocator,
                                          const KeyTy &key) {
    return new (allocator.allocate<DataLayoutSpecStorage>())
        DataLayoutSpecStorage(allocator.copyInto(key));
  }

  ArrayRef<DataLayoutEntryInterface> entries;
};

/// Simple data layout spec containing a list of entries that always verifies
/// as valid.
struct CustomDataLayoutSpec
    : public Attribute::AttrBase<CustomDataLayoutSpec, Attribute,
                                 DataLayoutSpecStorage,
                                 DataLayoutSpecInterface::Trait> {
  using Base::Base;
  static CustomDataLayoutSpec get(MLIRContext *ctx,
                                  ArrayRef<DataLayoutEntryInterface> entries) {
    return Base::get(ctx, entries);
  }
  CustomDataLayoutSpec
  combineWith(ArrayRef<DataLayoutSpecInterface> specs) const {
    return *this;
  }
  DataLayoutEntryListRef getEntries() const { return getImpl()->entries; }
  LogicalResult verifySpec(Location loc) { return success(); }
};

/// A type subject to data layout that exits the program if it is queried more
/// than once. Handy to check if the cache works.
struct SingleQueryType
    : public Type::TypeBase<SingleQueryType, Type, TypeStorage,
                            DataLayoutTypeInterface::Trait> {
  using Base::Base;

  static SingleQueryType get(MLIRContext *ctx) { return Base::get(ctx); }

  unsigned getTypeSizeInBits(const DataLayout &layout,
                             DataLayoutEntryListRef params) const {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return 1;
  }

  unsigned getABIAlignment(const DataLayout &layout,
                           DataLayoutEntryListRef params) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return 2;
  }

  unsigned getPreferredAlignment(const DataLayout &layout,
                                 DataLayoutEntryListRef params) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return 4;
  }
};

/// A types that is not subject to data layout.
struct TypeNoLayout : public Type::TypeBase<TypeNoLayout, Type, TypeStorage> {
  using Base::Base;

  static TypeNoLayout get(MLIRContext *ctx) { return Base::get(ctx); }
};

/// An op that serves as scope for data layout queries with the relevant
/// attribute attached. This can handle data layout requests for the built-in
/// types itself.
struct OpWithLayout : public Op<OpWithLayout, DataLayoutOpInterface::Trait> {
  using Op::Op;

  static StringRef getOperationName() { return "dltest.op_with_layout"; }

  DataLayoutSpecInterface getDataLayoutSpec() {
    return getOperation()->getAttrOfType<DataLayoutSpecInterface>(kAttrName);
  }

  static unsigned getTypeSizeInBits(Type type, const DataLayout &dataLayout,
                                    DataLayoutEntryListRef params) {
    // Make a recursive query.
    if (type.isa<FloatType>())
      return dataLayout.getTypeSizeInBits(
          IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth()));

    // Handle built-in types that are not handled by the default process.
    if (auto iType = type.dyn_cast<IntegerType>()) {
      for (DataLayoutEntryInterface entry : params)
        if (entry.getKey().dyn_cast<Type>() == type)
          return 8 *
                 entry.getValue().cast<IntegerAttr>().getValue().getZExtValue();
      return 8 * iType.getIntOrFloatBitWidth();
    }

    // Use the default process for everything else.
    return detail::getDefaultTypeSize(type, dataLayout, params);
  }

  static unsigned getTypeABIAlignment(Type type, const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) {
    return llvm::PowerOf2Ceil(getTypeSize(type, dataLayout, params));
  }

  static unsigned getTypePreferredAlignment(Type type,
                                            const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) {
    return 2 * getTypeABIAlignment(type, dataLayout, params);
  }
};

struct OpWith7BitByte
    : public Op<OpWith7BitByte, DataLayoutOpInterface::Trait> {
  using Op::Op;

  static StringRef getOperationName() { return "dltest.op_with_7bit_byte"; }

  DataLayoutSpecInterface getDataLayoutSpec() {
    return getOperation()->getAttrOfType<DataLayoutSpecInterface>(kAttrName);
  }

  // Bytes are assumed to be 7-bit here.
  static unsigned getTypeSize(Type type, const DataLayout &dataLayout,
                              DataLayoutEntryListRef params) {
    return llvm::divideCeil(dataLayout.getTypeSizeInBits(type), 7);
  }
};

/// A dialect putting all the above together.
struct DLTestDialect : Dialect {
  explicit DLTestDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<DLTestDialect>()) {
    ctx->getOrLoadDialect<DLTIDialect>();
    addAttributes<CustomDataLayoutSpec>();
    addOperations<OpWithLayout, OpWith7BitByte>();
    addTypes<SingleQueryType, TypeNoLayout>();
  }
  static StringRef getDialectNamespace() { return "dltest"; }

  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override {
    printer << "spec<";
    llvm::interleaveComma(attr.cast<CustomDataLayoutSpec>().getEntries(),
                          printer);
    printer << ">";
  }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override {
    bool ok =
        succeeded(parser.parseKeyword("spec")) && succeeded(parser.parseLess());
    (void)ok;
    assert(ok);
    if (succeeded(parser.parseOptionalGreater()))
      return CustomDataLayoutSpec::get(parser.getBuilder().getContext(), {});

    SmallVector<DataLayoutEntryInterface> entries;
    do {
      entries.emplace_back();
      ok = succeeded(parser.parseAttribute(entries.back()));
      assert(ok);
    } while (succeeded(parser.parseOptionalComma()));
    ok = succeeded(parser.parseGreater());
    assert(ok);
    return CustomDataLayoutSpec::get(parser.getBuilder().getContext(), entries);
  }

  void printType(Type type, DialectAsmPrinter &printer) const override {
    if (type.isa<SingleQueryType>())
      printer << "single_query";
    else
      printer << "no_layout";
  }

  Type parseType(DialectAsmParser &parser) const override {
    bool ok = succeeded(parser.parseKeyword("single_query"));
    (void)ok;
    assert(ok);
    return SingleQueryType::get(parser.getBuilder().getContext());
  }
};

} // end namespace

TEST(DataLayout, FallbackDefault) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 42)), 6u);
  EXPECT_EQ(layout.getTypeSize(Float16Type::get(&ctx)), 2u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 42)), 42u);
  EXPECT_EQ(layout.getTypeSizeInBits(Float16Type::get(&ctx)), 16u);
  EXPECT_EQ(layout.getTypeABIAlignment(IntegerType::get(&ctx, 42)), 8u);
  EXPECT_EQ(layout.getTypeABIAlignment(Float16Type::get(&ctx)), 2u);
  EXPECT_EQ(layout.getTypePreferredAlignment(IntegerType::get(&ctx, 42)), 8u);
  EXPECT_EQ(layout.getTypePreferredAlignment(Float16Type::get(&ctx)), 2u);
}

TEST(DataLayout, EmptySpec) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec< > } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 42)), 42u);
  EXPECT_EQ(layout.getTypeSize(Float16Type::get(&ctx)), 16u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 42)), 8u * 42u);
  EXPECT_EQ(layout.getTypeSizeInBits(Float16Type::get(&ctx)), 8u * 16u);
  EXPECT_EQ(layout.getTypeABIAlignment(IntegerType::get(&ctx, 42)), 64u);
  EXPECT_EQ(layout.getTypeABIAlignment(Float16Type::get(&ctx)), 16u);
  EXPECT_EQ(layout.getTypePreferredAlignment(IntegerType::get(&ctx, 42)), 128u);
  EXPECT_EQ(layout.getTypePreferredAlignment(Float16Type::get(&ctx)), 32u);
}

TEST(DataLayout, SpecWithEntries) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec<
  #dlti.dl_entry<i42, 5>,
  #dlti.dl_entry<i16, 6>
> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 42)), 5u);
  EXPECT_EQ(layout.getTypeSize(Float16Type::get(&ctx)), 6u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 42)), 40u);
  EXPECT_EQ(layout.getTypeSizeInBits(Float16Type::get(&ctx)), 48u);
  EXPECT_EQ(layout.getTypeABIAlignment(IntegerType::get(&ctx, 42)), 8u);
  EXPECT_EQ(layout.getTypeABIAlignment(Float16Type::get(&ctx)), 8u);
  EXPECT_EQ(layout.getTypePreferredAlignment(IntegerType::get(&ctx, 42)), 16u);
  EXPECT_EQ(layout.getTypePreferredAlignment(Float16Type::get(&ctx)), 16u);

  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 32)), 32u);
  EXPECT_EQ(layout.getTypeSize(Float32Type::get(&ctx)), 32u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 32)), 256u);
  EXPECT_EQ(layout.getTypeSizeInBits(Float32Type::get(&ctx)), 256u);
  EXPECT_EQ(layout.getTypeABIAlignment(IntegerType::get(&ctx, 32)), 32u);
  EXPECT_EQ(layout.getTypeABIAlignment(Float32Type::get(&ctx)), 32u);
  EXPECT_EQ(layout.getTypePreferredAlignment(IntegerType::get(&ctx, 32)), 64u);
  EXPECT_EQ(layout.getTypePreferredAlignment(Float32Type::get(&ctx)), 64u);
}

TEST(DataLayout, Caching) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec<> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);

  unsigned sum = 0;
  sum += layout.getTypeSize(SingleQueryType::get(&ctx));
  // The second call should hit the cache. If it does not, the function in
  // SingleQueryType will be called and will abort the process.
  sum += layout.getTypeSize(SingleQueryType::get(&ctx));
  // Make sure the complier doesn't optimize away the query code.
  EXPECT_EQ(sum, 2u);

  // A fresh data layout has a new cache, so the call to it should be dispatched
  // down to the type and abort the proces.
  DataLayout second(op);
  ASSERT_DEATH(second.getTypeSize(SingleQueryType::get(&ctx)), "repeated call");
}

TEST(DataLayout, CacheInvalidation) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec<
  #dlti.dl_entry<i42, 5>,
  #dlti.dl_entry<i16, 6>
> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);

  // Normal query is fine.
  EXPECT_EQ(layout.getTypeSize(Float16Type::get(&ctx)), 6u);

  // Replace the data layout spec with a new, empty spec.
  op->setAttr(kAttrName, CustomDataLayoutSpec::get(&ctx, {}));

  // Data layout is no longer valid and should trigger assertion when queried.
#ifndef NDEBUG
  ASSERT_DEATH(layout.getTypeSize(Float16Type::get(&ctx)), "no longer valid");
#endif
}

TEST(DataLayout, UnimplementedTypeInterface) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec<> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);

  ASSERT_DEATH(layout.getTypeSize(TypeNoLayout::get(&ctx)),
               "neither the scoping op nor the type class provide data layout "
               "information");
}

TEST(DataLayout, SevenBitByte) {
  const char *ir = R"MLIR(
"dltest.op_with_7bit_byte"() { dltest.layout = #dltest.spec<> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningModuleRef module = parseSourceString(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);

  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 42)), 42u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 32)), 32u);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 42)), 6u);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 32)), 5u);
}
