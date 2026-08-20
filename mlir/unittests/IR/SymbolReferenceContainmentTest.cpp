//===- SymbolReferenceContainmentTest.cpp - Containment query unit tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for SymbolTable::mayContainSymbolRefs, which records per uniqued type
// or attribute whether it transitively contains a SymbolRefAttr (conservatively
// true for mutable storage). Symbol-table verification relies on it to prune
// types and attributes that provably hold no symbol reference. Because a
// SymbolUserTypeInterface / SymbolUserAttrInterface implementation must spell
// its references as SymbolRefAttr sub-elements, a false answer is a sound
// reason to skip an instance even after the interface is attached late.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

#include "../../lib/IR/SymbolRefContainmentCache.h"
#include "../../test/lib/Dialect/Test/TestAttributes.h"
#include "../../test/lib/Dialect/Test/TestDialect.h"
#include "../../test/lib/Dialect/Test/TestTypes.h"

#include <atomic>
#include <thread>
#include <vector>

using namespace mlir;

namespace {

// Symbol-user models whose verification always fails, attached externally to
// exercise late interface attachment. One targets a type that structurally
// holds a SymbolRefAttr (a tensor with a symbol-ref encoding); the other
// targets f32, which holds none.
struct FailingTensorSymbolUserModel
    : public SymbolUserTypeInterface::ExternalModel<
          FailingTensorSymbolUserModel, RankedTensorType> {
  LogicalResult verifySymbolUses(Type type, Operation *op,
                                 SymbolTableCollection &symbolTable) const {
    return op->emitError("tensor rejected by its attached symbol-user model");
  }
};
struct FailingF32SymbolUserModel
    : public SymbolUserTypeInterface::ExternalModel<FailingF32SymbolUserModel,
                                                    Float32Type> {
  LogicalResult verifySymbolUses(Type type, Operation *op,
                                 SymbolTableCollection &symbolTable) const {
    return op->emitError("f32 rejected by its attached symbol-user model");
  }
};

class SymbolReferenceContainmentTest : public ::testing::Test {
protected:
  SymbolReferenceContainmentTest() {
    context.loadDialect<test::TestDialect>();
    context.allowUnregisteredDialects();
  }

  FlatSymbolRefAttr symbolRef() {
    return FlatSymbolRefAttr::get(&context, "sym");
  }

  // A conforming type implementing SymbolUserTypeInterface, spelling its
  // reference as a FlatSymbolRefAttr parameter: !test.symbol_ref<@sym>.
  test::TestSymbolUserType symbolUserType() {
    return test::TestSymbolUserType::get(&context, symbolRef());
  }

  // A conforming attribute implementing SymbolUserAttrInterface, spelling its
  // reference as a FlatSymbolRefAttr parameter: #test.symbol_ref_attr<@sym>.
  test::TestSymbolRefAttr symbolUserAttr() {
    return test::TestSymbolRefAttr::get(&context, symbolRef());
  }

  MLIRContext context;
};

// A leaf type holding no symbol reference answers false.
TEST_F(SymbolReferenceContainmentTest, LeafTypeIsClear) {
  EXPECT_FALSE(
      SymbolTable::mayContainSymbolRefs(IntegerType::get(&context, 32)));
}

// A plain attribute holding no symbol reference answers false.
TEST_F(SymbolReferenceContainmentTest, LeafAttrIsClear) {
  EXPECT_FALSE(
      SymbolTable::mayContainSymbolRefs(StringAttr::get(&context, "hi")));
  EXPECT_FALSE(SymbolTable::mayContainSymbolRefs(
      TypeAttr::get(IntegerType::get(&context, 32))));
}

// A SymbolRefAttr itself answers true.
TEST_F(SymbolReferenceContainmentTest, FlatSymbolRefAttrIsTrue) {
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(symbolRef()));
}

// A non-flat SymbolRefAttr, which nests further references, answers true.
TEST_F(SymbolReferenceContainmentTest, NestedSymbolRefAttrIsTrue) {
  SymbolRefAttr ref =
      SymbolRefAttr::get(StringAttr::get(&context, "root"),
                         {FlatSymbolRefAttr::get(&context, "n")});
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(ref));
}

// A conforming symbol-user type answers true through its SymbolRefAttr
// parameter (not through the interface, which plays no part in the answer).
TEST_F(SymbolReferenceContainmentTest, ConformingSymbolUserTypeIsTrue) {
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(symbolUserType()));
}

// A conforming symbol-user attribute answers true through its SymbolRefAttr
// parameter.
TEST_F(SymbolReferenceContainmentTest, ConformingSymbolUserAttrIsTrue) {
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(symbolUserAttr()));
}

// A type nesting a symbol-ref-bearing type propagates true.
TEST_F(SymbolReferenceContainmentTest, TypeNestingSymbolRefBearingType) {
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(
      TupleType::get(&context, {symbolUserType()})));
}

// A tuple of ordinary types stays false.
TEST_F(SymbolReferenceContainmentTest, TypeNestingOrdinaryTypesIsClear) {
  Type i32 = IntegerType::get(&context, 32);
  EXPECT_FALSE(
      SymbolTable::mayContainSymbolRefs(TupleType::get(&context, {i32, i32})));
}

// A type reaches a SymbolRefAttr two levels deep, through an attribute
// sub-element (a tensor encoding holding a TypeAttr of a symbol-ref type).
TEST_F(SymbolReferenceContainmentTest, TypeReachesSymbolRefThroughAttribute) {
  Attribute encoding = TypeAttr::get(symbolUserType());
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(encoding));
  RankedTensorType tensor =
      RankedTensorType::get({2}, IntegerType::get(&context, 32), encoding);
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(tensor));
}

// A type reaches a plain SymbolRefAttr through an attribute parameter (a tensor
// encoding).
TEST_F(SymbolReferenceContainmentTest, TypeWithSymbolRefAttrParameter) {
  RankedTensorType tensor =
      RankedTensorType::get({2}, IntegerType::get(&context, 32), symbolRef());
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(tensor));
}

// A dictionary attribute containing a plain SymbolRefAttr answers true.
TEST_F(SymbolReferenceContainmentTest, DictionaryAttrContainingSymbolRef) {
  NamedAttribute named(StringAttr::get(&context, "callee"), symbolRef());
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(
      DictionaryAttr::get(&context, {named})));
}

// A dictionary attribute containing a symbol-ref-bearing type inside a TypeAttr
// answers true; the answer on the dictionary summarizes its whole nested tree.
TEST_F(SymbolReferenceContainmentTest, DictionaryAttrContainingSymbolRefType) {
  NamedAttribute named(StringAttr::get(&context, "key"),
                       TypeAttr::get(symbolUserType()));
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(
      DictionaryAttr::get(&context, {named})));
}

// A dictionary attribute with no symbol reference stays false.
TEST_F(SymbolReferenceContainmentTest, DictionaryAttrIsClear) {
  NamedAttribute named(StringAttr::get(&context, "key"),
                       TypeAttr::get(IntegerType::get(&context, 32)));
  EXPECT_FALSE(SymbolTable::mayContainSymbolRefs(
      DictionaryAttr::get(&context, {named})));
}

// A type carrying a mutable component answers true conservatively, since its
// sub-elements may change after the answer is computed at first query; the
// fill never reads its contents.
TEST_F(SymbolReferenceContainmentTest, MutableTypeReportsConservatively) {
  test::TestRecursiveType recursive =
      test::TestRecursiveType::get(&context, "rec");
  EXPECT_TRUE(SymbolTable::mayContainSymbolRefs(recursive));
}

// A container of a mutable-storage kind inherits true, even before the mutable
// body is populated, so no later mutation can turn a cached false stale.
TEST_F(SymbolReferenceContainmentTest, ContainerOfMutableTypeIsTrue) {
  test::TestRecursiveType recursive =
      test::TestRecursiveType::get(&context, "rec2");
  EXPECT_TRUE(
      SymbolTable::mayContainSymbolRefs(TupleType::get(&context, {recursive})));
}

// Interface membership plays no part in the answer, so late attachment needs no
// fallback: a type that structurally holds a SymbolRefAttr (a tensor with a
// symbol-ref encoding) answers true from interning, so verification visits it
// and the newly-attached verifySymbolUses fires.
TEST_F(SymbolReferenceContainmentTest, LateInterfaceAttachmentStillVerifies) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      "module { \"foo.op\"() : () -> tensor<4xf32, @sym> }", &context);
  ASSERT_TRUE(module);

  RankedTensorType::attachInterface<FailingTensorSymbolUserModel>(context);
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(*module)));
}

// The contract boundary: a type that references a symbol without spelling it as
// a SymbolRefAttr (here f32, standing in for a non-conforming symbol-user type)
// answers false and is therefore skipped -- its verifySymbolUses never fires,
// so verification succeeds. This is the documented cost of the interface
// contract.
TEST_F(SymbolReferenceContainmentTest, NonConformingSymbolUserTypeIsSkipped) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      "module { \"foo.op\"() : () -> f32 }", &context);
  ASSERT_TRUE(module);

  Float32Type::attachInterface<FailingF32SymbolUserModel>(context);
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(succeeded(verify(*module)));
}

// Concurrency crux: many threads query the same cold, shared objects at once,
// exactly as parallel symbol-table verification fills the context cache from
// several isolated-op workers simultaneously. Every thread must agree with the
// single-threaded ground truth, and the run must be clean under
// ThreadSanitizer. Built cold (constructed but never queried before the
// threads start) so the fills genuinely race.
TEST_F(SymbolReferenceContainmentTest, ConcurrentFillIsRaceFree) {
  ASSERT_TRUE(context.isMultithreadingEnabled());

  // A set of shared objects spanning the interesting fill paths and sharing
  // sub-elements, so concurrent fills overlap on the same interior objects.
  Type i32 = IntegerType::get(&context, 32);
  std::vector<Attribute> attrs = {
      StringAttr::get(&context, "leaf"),
      symbolRef(),
      symbolUserAttr(),
      TypeAttr::get(symbolUserType()),
      DictionaryAttr::get(
          &context,
          {NamedAttribute(StringAttr::get(&context, "callee"), symbolRef())}),
      DictionaryAttr::get(&context,
                          {NamedAttribute(StringAttr::get(&context, "plain"),
                                          TypeAttr::get(i32))}),
  };
  std::vector<Type> types = {
      i32,
      symbolUserType(),
      TupleType::get(&context, {i32, symbolUserType()}),
      TupleType::get(&context, {i32, i32}),
      RankedTensorType::get({2}, i32, symbolRef()),
  };

  // Ground truth computed single-threaded (this also warms nothing new for the
  // threads, since these very objects are what they race to fill).
  std::vector<bool> attrTruth, typeTruth;
  for (Attribute a : attrs)
    attrTruth.push_back(SymbolTable::mayContainSymbolRefs(a));
  for (Type t : types)
    typeTruth.push_back(SymbolTable::mayContainSymbolRefs(t));

  // Fresh context so the threads hit a cold cache and genuinely race the fills.
  MLIRContext raced;
  raced.loadDialect<test::TestDialect>();
  raced.allowUnregisteredDialects();
  ASSERT_TRUE(raced.isMultithreadingEnabled());
  auto rebuildAttrs = [&](MLIRContext &ctx) {
    Type ri32 = IntegerType::get(&ctx, 32);
    FlatSymbolRefAttr rsym = FlatSymbolRefAttr::get(&ctx, "sym");
    return std::vector<Attribute>{
        StringAttr::get(&ctx, "leaf"),
        rsym,
        test::TestSymbolRefAttr::get(&ctx, rsym),
        TypeAttr::get(test::TestSymbolUserType::get(&ctx, rsym)),
        DictionaryAttr::get(
            &ctx, {NamedAttribute(StringAttr::get(&ctx, "callee"), rsym)}),
        DictionaryAttr::get(&ctx,
                            {NamedAttribute(StringAttr::get(&ctx, "plain"),
                                            TypeAttr::get(ri32))}),
    };
  };
  auto rebuildTypes = [&](MLIRContext &ctx) {
    Type ri32 = IntegerType::get(&ctx, 32);
    FlatSymbolRefAttr rsym = FlatSymbolRefAttr::get(&ctx, "sym");
    test::TestSymbolUserType user = test::TestSymbolUserType::get(&ctx, rsym);
    return std::vector<Type>{
        ri32,
        user,
        TupleType::get(&ctx, {ri32, user}),
        TupleType::get(&ctx, {ri32, ri32}),
        RankedTensorType::get({2}, ri32, rsym),
    };
  };
  std::vector<Attribute> racedAttrs = rebuildAttrs(raced);
  std::vector<Type> racedTypes = rebuildTypes(raced);

  const int numThreads = 16;
  std::vector<std::vector<bool>> attrResults(numThreads);
  std::vector<std::vector<bool>> typeResults(numThreads);
  std::atomic<int> ready{0};
  std::atomic<bool> go{false};
  std::vector<std::thread> threads;
  for (int i = 0; i < numThreads; ++i)
    threads.emplace_back([&, i] {
      ready.fetch_add(1);
      while (!go.load())
        ;
      for (Attribute a : racedAttrs)
        attrResults[i].push_back(SymbolTable::mayContainSymbolRefs(a));
      for (Type t : racedTypes)
        typeResults[i].push_back(SymbolTable::mayContainSymbolRefs(t));
    });
  while (ready.load() < numThreads)
    ;
  go.store(true);
  for (std::thread &t : threads)
    t.join();

  for (int i = 0; i < numThreads; ++i) {
    ASSERT_EQ(attrResults[i].size(), attrTruth.size());
    ASSERT_EQ(typeResults[i].size(), typeTruth.size());
    EXPECT_EQ(attrResults[i], attrTruth) << "thread " << i;
    EXPECT_EQ(typeResults[i], typeTruth) << "thread " << i;
  }
}

// Growth preserves every recorded clear fact. The clear-object set starts empty
// and grows as it fills, so inserting far past its initial capacity forces
// several rehashes; every live key must survive each grow. Synthetic pointers
// in a regular stride stand in for the bump-allocated storage addresses the
// cache keys on; the keys are opaque and never dereferenced. Only clear facts
// are recorded, so this checks that each survives the rehash and that
// may-contain facts, which the store never keeps, stay misses.
TEST_F(SymbolReferenceContainmentTest, SetGrowthPreservesEntries) {
  detail::SymbolRefContainmentCache cache;
  const int n = 5000; // many doublings past the minimal set
  std::vector<Type> clearKeys;
  std::vector<Type> mayContainKeys;
  for (int i = 0; i < n; ++i) {
    auto *p = reinterpret_cast<const void *>(
        static_cast<uintptr_t>(0x100000 + i * 8));
    Type key = Type::getFromOpaquePointer(p);
    if (i % 3 == 0) {
      // A may-contain fact is never stored: insert returns true but records
      // nothing, so a later lookup stays a miss and the caller recomputes.
      EXPECT_TRUE(cache.insert(key, /*value=*/true, /*lock=*/false));
      mayContainKeys.push_back(key);
    } else {
      // A clear fact is recorded and returned unchanged.
      EXPECT_FALSE(cache.insert(key, /*value=*/false, /*lock=*/false));
      clearKeys.push_back(key);
    }
  }
  // After all the growth, every clear fact is still found as false.
  for (Type key : clearKeys) {
    std::optional<bool> got = cache.lookup(key, /*lock=*/false);
    ASSERT_TRUE(got.has_value()) << "lost clear entry";
    EXPECT_FALSE(*got);
  }
  // A may-contain fact was never stored, so it stays a miss.
  for (Type key : mayContainKeys)
    EXPECT_FALSE(cache.lookup(key, /*lock=*/false).has_value());
  // A key never inserted is a miss, not a stray cluster hit.
  Type absent = Type::getFromOpaquePointer(
      reinterpret_cast<const void *>(static_cast<uintptr_t>(0x100000 + n * 8)));
  EXPECT_FALSE(cache.lookup(absent, /*lock=*/false).has_value());
}

// A DistinctAttr is keyed by the address of its own storage, which comes from a
// separate always-allocating allocator rather than the attribute uniquer; the
// cache must record and recover that pointer just like a uniqued one.
TEST_F(SymbolReferenceContainmentTest, DistinctAttrIsHandled) {
  DistinctAttr d1 = DistinctAttr::create(UnitAttr::get(&context));
  DistinctAttr d2 = DistinctAttr::create(UnitAttr::get(&context));
  ASSERT_NE(d1, d2); // always-allocating: distinct instances, distinct pointers

  // Through the public query a DistinctAttr exposes no SymbolRefAttr
  // sub-element, so it answers false, stably across the cold fill and the warm
  // cache hit.
  bool cold = SymbolTable::mayContainSymbolRefs(d1);
  EXPECT_FALSE(cold);
  EXPECT_EQ(SymbolTable::mayContainSymbolRefs(d1), cold);
  EXPECT_FALSE(SymbolTable::mayContainSymbolRefs(d2));

  // A may-contain answer is never recorded: a forced-true insert on a
  // distinct-allocator pointer returns true but stores nothing, so the warm
  // lookup stays a miss and the caller recomputes. A clear answer, by contrast,
  // is recorded and returned false warm, while an uninserted distinct pointer
  // stays a miss -- the pointer round-trips just like a uniqued one.
  detail::SymbolRefContainmentCache cache;
  EXPECT_TRUE(cache.insert(d1, /*value=*/true, /*lock=*/false));
  EXPECT_FALSE(cache.lookup(d1, /*lock=*/false).has_value());
  EXPECT_FALSE(cache.insert(d1, /*value=*/false, /*lock=*/false));
  std::optional<bool> warm = cache.lookup(d1, /*lock=*/false);
  ASSERT_TRUE(warm.has_value());
  EXPECT_FALSE(*warm);
  EXPECT_FALSE(cache.lookup(d2, /*lock=*/false).has_value());
}

} // namespace
