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
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using namespace mlir;

namespace {
constexpr static llvm::StringLiteral kAttrName = "dltest.layout";
constexpr static llvm::StringLiteral kEndiannesKeyName = "dltest.endianness";
constexpr static llvm::StringLiteral kAllocaKeyName =
    "dltest.alloca_memory_space";
constexpr static llvm::StringLiteral kProgramKeyName =
    "dltest.program_memory_space";
constexpr static llvm::StringLiteral kGlobalKeyName =
    "dltest.global_memory_space";
constexpr static llvm::StringLiteral kStackAlignmentKeyName =
    "dltest.stack_alignment";

constexpr static llvm::StringLiteral kTargetSystemDescAttrName =
    "dl_target_sys_desc_test.target_system_spec";

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
    : public Attribute::AttrBase<
          CustomDataLayoutSpec, Attribute, DataLayoutSpecStorage,
          DLTIQueryInterface::Trait, DataLayoutSpecInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomDataLayoutSpec)

  using Base::Base;

  static constexpr StringLiteral name = "test.custom_data_layout_spec";

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
  StringAttr getEndiannessIdentifier(MLIRContext *context) const {
    return Builder(context).getStringAttr(kEndiannesKeyName);
  }
  StringAttr getAllocaMemorySpaceIdentifier(MLIRContext *context) const {
    return Builder(context).getStringAttr(kAllocaKeyName);
  }
  StringAttr getProgramMemorySpaceIdentifier(MLIRContext *context) const {
    return Builder(context).getStringAttr(kProgramKeyName);
  }
  StringAttr getGlobalMemorySpaceIdentifier(MLIRContext *context) const {
    return Builder(context).getStringAttr(kGlobalKeyName);
  }
  StringAttr getStackAlignmentIdentifier(MLIRContext *context) const {
    return Builder(context).getStringAttr(kStackAlignmentKeyName);
  }
  FailureOr<Attribute> query(DataLayoutEntryKey key) const {
    return llvm::cast<mlir::DataLayoutSpecInterface>(*this).queryHelper(key);
  }
};

class TargetSystemSpecStorage : public AttributeStorage {
public:
  using KeyTy = ArrayRef<DataLayoutEntryInterface>;

  TargetSystemSpecStorage(ArrayRef<DataLayoutEntryInterface> entries)
      : entries(entries) {}

  bool operator==(const KeyTy &key) const { return key == entries; }

  static TargetSystemSpecStorage *
  construct(AttributeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<TargetSystemSpecStorage>())
        TargetSystemSpecStorage(allocator.copyInto(key));
  }

  ArrayRef<DataLayoutEntryInterface> entries;
};

struct CustomTargetSystemSpec
    : public Attribute::AttrBase<
          CustomTargetSystemSpec, Attribute, TargetSystemSpecStorage,
          DLTIQueryInterface::Trait, TargetSystemSpecInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CustomDataLayoutSpec)

  using Base::Base;

  static constexpr StringLiteral name = "test.custom_target_system_spec";

  static CustomTargetSystemSpec
  get(MLIRContext *ctx, ArrayRef<DataLayoutEntryInterface> entries) {
    return Base::get(ctx, entries);
  }
  ArrayRef<DataLayoutEntryInterface> getEntries() const {
    return getImpl()->entries;
  }
  LogicalResult verifySpec(Location loc) { return success(); }
  std::optional<TargetDeviceSpecInterface>
  getDeviceSpecForDeviceID(TargetSystemSpecInterface::DeviceID deviceID) {
    for (const auto &entry : getEntries()) {
      if (entry.getKey() == DataLayoutEntryKey(deviceID))
        if (auto deviceSpec =
                llvm::dyn_cast<TargetDeviceSpecInterface>(entry.getValue()))
          return deviceSpec;
    }
    return std::nullopt;
  }
  FailureOr<Attribute> query(DataLayoutEntryKey key) const {
    return llvm::cast<mlir::TargetSystemSpecInterface>(*this).queryHelper(key);
  }
};

/// A type subject to data layout that exits the program if it is queried more
/// than once. Handy to check if the cache works.
struct SingleQueryType
    : public Type::TypeBase<SingleQueryType, Type, TypeStorage,
                            DataLayoutTypeInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SingleQueryType)

  using Base::Base;

  static constexpr StringLiteral name = "test.single_query";

  static SingleQueryType get(MLIRContext *ctx) { return Base::get(ctx); }

  llvm::TypeSize getTypeSizeInBits(const DataLayout &layout,
                                   DataLayoutEntryListRef params) const {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return llvm::TypeSize::getFixed(1);
  }

  uint64_t getABIAlignment(const DataLayout &layout,
                           DataLayoutEntryListRef params) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return 2;
  }

  uint64_t getPreferredAlignment(const DataLayout &layout,
                                 DataLayoutEntryListRef params) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return 4;
  }

  Attribute getEndianness(DataLayoutEntryInterface entry) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return Attribute();
  }

  Attribute getAllocaMemorySpace(DataLayoutEntryInterface entry) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return Attribute();
  }

  Attribute getProgramMemorySpace(DataLayoutEntryInterface entry) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return Attribute();
  }

  Attribute getGlobalMemorySpace(DataLayoutEntryInterface entry) {
    static bool executed = false;
    if (executed)
      llvm::report_fatal_error("repeated call");

    executed = true;
    return Attribute();
  }
};

/// A types that is not subject to data layout.
struct TypeNoLayout : public Type::TypeBase<TypeNoLayout, Type, TypeStorage> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TypeNoLayout)

  using Base::Base;

  static constexpr StringLiteral name = "test.no_layout";

  static TypeNoLayout get(MLIRContext *ctx) { return Base::get(ctx); }
};

/// An op that serves as scope for data layout queries with the relevant
/// attribute attached. This can handle data layout requests for the built-in
/// types itself.
struct OpWithLayout : public Op<OpWithLayout, DataLayoutOpInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWithLayout)

  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "dltest.op_with_layout"; }

  DataLayoutSpecInterface getDataLayoutSpec() {
    return getOperation()->getAttrOfType<DataLayoutSpecInterface>(kAttrName);
  }

  TargetSystemSpecInterface getTargetSystemSpec() {
    return getOperation()->getAttrOfType<TargetSystemSpecInterface>(
        kTargetSystemDescAttrName);
  }

  static llvm::TypeSize getTypeSizeInBits(Type type,
                                          const DataLayout &dataLayout,
                                          DataLayoutEntryListRef params) {
    // Make a recursive query.
    if (isa<FloatType>(type))
      return dataLayout.getTypeSizeInBits(
          IntegerType::get(type.getContext(), type.getIntOrFloatBitWidth()));

    // Handle built-in types that are not handled by the default process.
    if (auto iType = dyn_cast<IntegerType>(type)) {
      for (DataLayoutEntryInterface entry : params)
        if (llvm::dyn_cast_if_present<Type>(entry.getKey()) == type)
          return llvm::TypeSize::getFixed(
              8 *
              cast<IntegerAttr>(entry.getValue()).getValue().getZExtValue());
      return llvm::TypeSize::getFixed(8 * iType.getIntOrFloatBitWidth());
    }

    // Use the default process for everything else.
    return detail::getDefaultTypeSize(type, dataLayout, params);
  }

  static uint64_t getTypeABIAlignment(Type type, const DataLayout &dataLayout,
                                      DataLayoutEntryListRef params) {
    return llvm::PowerOf2Ceil(getTypeSize(type, dataLayout, params));
  }

  static uint64_t getTypePreferredAlignment(Type type,
                                            const DataLayout &dataLayout,
                                            DataLayoutEntryListRef params) {
    return 2 * getTypeABIAlignment(type, dataLayout, params);
  }
};

struct OpWith7BitByte
    : public Op<OpWith7BitByte, DataLayoutOpInterface::Trait> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpWith7BitByte)

  using Op::Op;
  static ArrayRef<StringRef> getAttributeNames() { return {}; }

  static StringRef getOperationName() { return "dltest.op_with_7bit_byte"; }

  DataLayoutSpecInterface getDataLayoutSpec() {
    return getOperation()->getAttrOfType<DataLayoutSpecInterface>(kAttrName);
  }

  TargetSystemSpecInterface getTargetSystemSpec() {
    return getOperation()->getAttrOfType<TargetSystemSpecInterface>(
        kTargetSystemDescAttrName);
  }

  // Bytes are assumed to be 7-bit here.
  static llvm::TypeSize getTypeSize(Type type, const DataLayout &dataLayout,
                                    DataLayoutEntryListRef params) {
    return mlir::detail::divideCeil(dataLayout.getTypeSizeInBits(type), 7);
  }
};

/// A dialect putting all the above together.
struct DLTestDialect : Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DLTestDialect)

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
    llvm::interleaveComma(cast<CustomDataLayoutSpec>(attr).getEntries(),
                          printer);
    printer << ">";
  }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override {
    bool ok =
        succeeded(parser.parseKeyword("spec")) && succeeded(parser.parseLess());
    (void)ok;
    assert(ok);
    if (succeeded(parser.parseOptionalGreater()))
      return CustomDataLayoutSpec::get(parser.getContext(), {});

    SmallVector<DataLayoutEntryInterface> entries;
    ok = succeeded(parser.parseCommaSeparatedList([&]() {
      entries.emplace_back();
      ok = succeeded(parser.parseAttribute(entries.back()));
      assert(ok);
      return success();
    }));
    assert(ok);
    ok = succeeded(parser.parseGreater());
    assert(ok);
    return CustomDataLayoutSpec::get(parser.getContext(), entries);
  }

  void printType(Type type, DialectAsmPrinter &printer) const override {
    if (isa<SingleQueryType>(type))
      printer << "single_query";
    else
      printer << "no_layout";
  }

  Type parseType(DialectAsmParser &parser) const override {
    bool ok = succeeded(parser.parseKeyword("single_query"));
    (void)ok;
    assert(ok);
    return SingleQueryType::get(parser.getContext());
  }
};

/// A dialect to test DLTI's target system spec and related attributes
struct DLTargetSystemDescTestDialect : public Dialect {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DLTargetSystemDescTestDialect)

  explicit DLTargetSystemDescTestDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx,
                TypeID::get<DLTargetSystemDescTestDialect>()) {
    ctx->getOrLoadDialect<DLTIDialect>();
    addAttributes<CustomTargetSystemSpec>();
  }
  static StringRef getDialectNamespace() { return "dl_target_sys_desc_test"; }

  void printAttribute(Attribute attr,
                      DialectAsmPrinter &printer) const override {
    printer << "target_system_spec<";
    llvm::interleaveComma(cast<CustomTargetSystemSpec>(attr).getEntries(),
                          printer, [&](const auto &it) {
                            printer << dyn_cast<StringAttr>(it.getKey()) << ":"
                                    << it.getValue();
                          });
    printer << ">";
  }

  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override {
    bool ok = succeeded(parser.parseKeyword("target_system_spec")) &&
              succeeded(parser.parseLess());
    (void)ok;
    assert(ok);
    if (succeeded(parser.parseOptionalGreater()))
      return CustomTargetSystemSpec::get(parser.getContext(), {});

    auto parseTargetDeviceSpecEntry =
        [&](AsmParser &parser) -> FailureOr<TargetDeviceSpecEntry> {
      std::string deviceID;
      if (failed(parser.parseString(&deviceID))) {
        parser.emitError(parser.getCurrentLocation())
            << "DeviceID is missing, or is not of string type";
        return failure();
      }
      if (failed(parser.parseColon())) {
        parser.emitError(parser.getCurrentLocation()) << "Missing colon";
        return failure();
      }

      TargetDeviceSpecInterface targetDeviceSpec;
      if (failed(parser.parseAttribute(targetDeviceSpec))) {
        parser.emitError(parser.getCurrentLocation())
            << "Error in parsing target device spec";
        return failure();
      }
      return std::make_pair(parser.getBuilder().getStringAttr(deviceID),
                            targetDeviceSpec);
    };

    SmallVector<DataLayoutEntryInterface> entries;
    ok = succeeded(parser.parseCommaSeparatedList([&]() {
      auto deviceIDAndTargetDeviceSpecPair = parseTargetDeviceSpecEntry(parser);
      ok = succeeded(deviceIDAndTargetDeviceSpecPair);
      assert(ok);
      auto entry =
          DataLayoutEntryAttr::get(deviceIDAndTargetDeviceSpecPair->first,
                                   deviceIDAndTargetDeviceSpecPair->second);
      entries.push_back(entry);
      return success();
    }));
    assert(ok);
    ok = succeeded(parser.parseGreater());
    assert(ok);
    return CustomTargetSystemSpec::get(parser.getContext(), entries);
  }
};

} // namespace

TEST(DataLayout, FallbackDefault) {
  const char *ir = R"MLIR(
module {}
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  DataLayout layout(module.get());
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 42)), 6u);
  EXPECT_EQ(layout.getTypeSize(Float16Type::get(&ctx)), 2u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 42)), 42u);
  EXPECT_EQ(layout.getTypeSizeInBits(Float16Type::get(&ctx)), 16u);
  EXPECT_EQ(layout.getTypeABIAlignment(IntegerType::get(&ctx, 42)), 8u);
  EXPECT_EQ(layout.getTypeABIAlignment(Float16Type::get(&ctx)), 2u);
  EXPECT_EQ(layout.getTypePreferredAlignment(IntegerType::get(&ctx, 42)), 8u);
  EXPECT_EQ(layout.getTypePreferredAlignment(Float16Type::get(&ctx)), 2u);

  EXPECT_EQ(layout.getEndianness(), Attribute());
  EXPECT_EQ(layout.getAllocaMemorySpace(), Attribute());
  EXPECT_EQ(layout.getProgramMemorySpace(), Attribute());
  EXPECT_EQ(layout.getGlobalMemorySpace(), Attribute());
  EXPECT_EQ(layout.getStackAlignment(), 0u);
}

TEST(DataLayout, NullSpec) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
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
  EXPECT_EQ(layout.getTypeIndexBitwidth(Float16Type::get(&ctx)), std::nullopt);
  EXPECT_EQ(layout.getTypeIndexBitwidth(IndexType::get(&ctx)), 64u);

  EXPECT_EQ(layout.getEndianness(), Attribute());
  EXPECT_EQ(layout.getAllocaMemorySpace(), Attribute());
  EXPECT_EQ(layout.getProgramMemorySpace(), Attribute());
  EXPECT_EQ(layout.getGlobalMemorySpace(), Attribute());
  EXPECT_EQ(layout.getStackAlignment(), 0u);

  EXPECT_EQ(layout.getDevicePropertyValue(
                Builder(&ctx).getStringAttr("CPU" /* device ID*/),
                Builder(&ctx).getStringAttr("L1_cache_size_in_bytes")),
            std::nullopt);
  EXPECT_EQ(layout.getDevicePropertyValue(
                Builder(&ctx).getStringAttr("CPU" /* device ID*/),
                Builder(&ctx).getStringAttr("max_vector_width")),
            std::nullopt);
}

TEST(DataLayout, EmptySpec) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec< > } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
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
  EXPECT_EQ(layout.getTypeIndexBitwidth(Float16Type::get(&ctx)), std::nullopt);
  EXPECT_EQ(layout.getTypeIndexBitwidth(IndexType::get(&ctx)), 64u);

  EXPECT_EQ(layout.getEndianness(), Attribute());
  EXPECT_EQ(layout.getAllocaMemorySpace(), Attribute());
  EXPECT_EQ(layout.getProgramMemorySpace(), Attribute());
  EXPECT_EQ(layout.getGlobalMemorySpace(), Attribute());
  EXPECT_EQ(layout.getStackAlignment(), 0u);

  EXPECT_EQ(layout.getDevicePropertyValue(
                Builder(&ctx).getStringAttr("CPU" /* device ID*/),
                Builder(&ctx).getStringAttr("L1_cache_size_in_bytes")),
            std::nullopt);
  EXPECT_EQ(layout.getDevicePropertyValue(
                Builder(&ctx).getStringAttr("CPU" /* device ID*/),
                Builder(&ctx).getStringAttr("max_vector_width")),
            std::nullopt);
}

TEST(DataLayout, SpecWithEntries) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec<
  #dlti.dl_entry<i42, 5>,
  #dlti.dl_entry<i16, 6>,
  #dlti.dl_entry<index, 42>,
  #dlti.dl_entry<"dltest.endianness", "little">,
  #dlti.dl_entry<"dltest.alloca_memory_space", 5 : i32>,
  #dlti.dl_entry<"dltest.program_memory_space", 3 : i32>,
  #dlti.dl_entry<"dltest.global_memory_space", 2 : i32>,
  #dlti.dl_entry<"dltest.stack_alignment", 128 : i32>
> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
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
  EXPECT_EQ(layout.getTypeIndexBitwidth(Float16Type::get(&ctx)), std::nullopt);
  EXPECT_EQ(layout.getTypeIndexBitwidth(IndexType::get(&ctx)), 42u);

  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 32)), 32u);
  EXPECT_EQ(layout.getTypeSize(Float32Type::get(&ctx)), 32u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 32)), 256u);
  EXPECT_EQ(layout.getTypeSizeInBits(Float32Type::get(&ctx)), 256u);
  EXPECT_EQ(layout.getTypeABIAlignment(IntegerType::get(&ctx, 32)), 32u);
  EXPECT_EQ(layout.getTypeABIAlignment(Float32Type::get(&ctx)), 32u);
  EXPECT_EQ(layout.getTypePreferredAlignment(IntegerType::get(&ctx, 32)), 64u);
  EXPECT_EQ(layout.getTypePreferredAlignment(Float32Type::get(&ctx)), 64u);

  EXPECT_EQ(layout.getEndianness(), Builder(&ctx).getStringAttr("little"));
  EXPECT_EQ(layout.getAllocaMemorySpace(), Builder(&ctx).getI32IntegerAttr(5));
  EXPECT_EQ(layout.getProgramMemorySpace(), Builder(&ctx).getI32IntegerAttr(3));
  EXPECT_EQ(layout.getGlobalMemorySpace(), Builder(&ctx).getI32IntegerAttr(2));
  EXPECT_EQ(layout.getStackAlignment(), 128u);
}

TEST(DataLayout, SpecWithTargetSystemDescEntries) {
  const char *ir = R"MLIR(
  module attributes { dl_target_sys_desc_test.target_system_spec =
    #dl_target_sys_desc_test.target_system_spec<
      "CPU": #dlti.target_device_spec<
              #dlti.dl_entry<"L1_cache_size_in_bytes", "4096">,
              #dlti.dl_entry<"max_vector_op_width", "128">>
    > } {}
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTargetSystemDescTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  DataLayout layout(*module);
  EXPECT_EQ(layout.getDevicePropertyValue(
                Builder(&ctx).getStringAttr("CPU") /* device ID*/,
                Builder(&ctx).getStringAttr("L1_cache_size_in_bytes")),
            std::optional<Attribute>(Builder(&ctx).getStringAttr("4096")));
  EXPECT_EQ(layout.getDevicePropertyValue(
                Builder(&ctx).getStringAttr("CPU") /* device ID*/,
                Builder(&ctx).getStringAttr("max_vector_op_width")),
            std::optional<Attribute>(Builder(&ctx).getStringAttr("128")));
}

TEST(DataLayout, Caching) {
  const char *ir = R"MLIR(
"dltest.op_with_layout"() { dltest.layout = #dltest.spec<> } : () -> ()
  )MLIR";

  DialectRegistry registry;
  registry.insert<DLTIDialect, DLTestDialect>();
  MLIRContext ctx(registry);

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
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
  // down to the type and abort the process.
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

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
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

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
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

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(ir, &ctx);
  auto op =
      cast<DataLayoutOpInterface>(module->getBody()->getOperations().front());
  DataLayout layout(op);

  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 42)), 42u);
  EXPECT_EQ(layout.getTypeSizeInBits(IntegerType::get(&ctx, 32)), 32u);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 42)), 6u);
  EXPECT_EQ(layout.getTypeSize(IntegerType::get(&ctx, 32)), 5u);
}
