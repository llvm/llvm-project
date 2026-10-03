//===- SymbolReferenceContainmentTest.cpp - Containment query unit tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the per-context symbol-reference containment cache, whose
// isSymbolRefFree query answers true for a uniqued type or attribute proven
// free of any transitive SymbolRefAttr, and false conservatively otherwise
// (including mutable storage). Symbol-table verification relies on it to prune
// the symbol-using types it would otherwise walk. Because a
// SymbolUserTypeInterface implementation must spell its references as
// SymbolRefAttr sub-elements, a true (free) answer is a sound reason to skip
// such a type even after the interface is attached late.
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
#include <string>
#include <thread>
#include <vector>

using namespace mlir;

namespace {

// A symbol-user model whose verification always fails, attached externally to
// exercise late interface attachment on an arbitrary concrete type.
template <typename ConcreteType>
struct FailingSymbolUserModel
    : public SymbolUserTypeInterface::ExternalModel<
          FailingSymbolUserModel<ConcreteType>, ConcreteType> {
  LogicalResult verifySymbolUses(Type type, Operation *op,
                                 SymbolTableCollection &symbolTable) const {
    return op->emitError("rejected by its attached symbol-user model");
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

  // Query the cache for `obj`; `expected` is whether it is proven symbol-ref-
  // free. The row index names a failed row without printing `obj`, which a
  // mutable-storage kind with an unset body cannot survive.
  template <typename T>
  void expect(T obj, bool expected) {
    SCOPED_TRACE("containment row " + std::to_string(++row));
    EXPECT_EQ(
        detail::getSymbolRefContainmentCache(&context).isSymbolRefFree(obj),
        expected);
  }

  unsigned row = 0;
  MLIRContext context;
};

// The symbol-ref-free answer across leaves, self-references, conforming user
// types/attributes, and every nesting path a symbol reference can hide behind.
TEST_F(SymbolReferenceContainmentTest, ContainmentTruthTable) {
  Type i32 = IntegerType::get(&context, 32);

  // Leaves hold no reference and are free.
  expect(i32, true);
  expect(StringAttr::get(&context, "hi"), true);
  expect(TypeAttr::get(i32), true);

  // A SymbolRefAttr, flat or nesting further references, is itself a reference,
  // so it is not free.
  expect(symbolRef(), false);
  expect(SymbolRefAttr::get(StringAttr::get(&context, "root"),
                            {FlatSymbolRefAttr::get(&context, "n")}),
         false);

  // A conforming symbol-user type/attribute is not free -- it carries a
  // SymbolRefAttr parameter (the interface plays no part in the answer).
  expect(symbolUserType(), false);
  expect(symbolUserAttr(), false);

  // Nesting propagates the reference; ordinary nesting stays free.
  expect(TupleType::get(&context, {symbolUserType()}), false);
  expect(TupleType::get(&context, {i32, i32}), true);

  // A type reaches a reference through an attribute sub-element -- a plain
  // SymbolRefAttr encoding, or a TypeAttr of a symbol-ref-bearing type.
  expect(RankedTensorType::get({2}, i32, symbolRef()), false);
  expect(TypeAttr::get(symbolUserType()), false);
  expect(RankedTensorType::get({2}, i32, TypeAttr::get(symbolUserType())),
         false);

  // A dictionary summarizes its whole nested tree, whichever way a reference
  // hides, and stays free when none does.
  expect(DictionaryAttr::get(
             &context, {NamedAttribute(StringAttr::get(&context, "callee"),
                                       symbolRef())}),
         false);
  expect(DictionaryAttr::get(&context,
                             {NamedAttribute(StringAttr::get(&context, "key"),
                                             TypeAttr::get(symbolUserType()))}),
         false);
  expect(DictionaryAttr::get(&context,
                             {NamedAttribute(StringAttr::get(&context, "key"),
                                             TypeAttr::get(i32))}),
         true);

  // A mutable-storage kind, and any container of one, answers not-free
  // conservatively: its sub-elements may change after the query, so its
  // contents are never read and no later mutation can turn a cached-free answer
  // stale.
  expect(test::TestRecursiveType::get(&context, "rec"), false);
  expect(
      TupleType::get(&context, {test::TestRecursiveType::get(&context, "c")}),
      false);

  // A DistinctAttr is keyed by its own always-allocated storage address, not
  // the attribute uniquer; two instances answer free independently, and a
  // repeat query is stable across the cold fill and the warm hit.
  DistinctAttr d1 = DistinctAttr::create(UnitAttr::get(&context));
  DistinctAttr d2 = DistinctAttr::create(UnitAttr::get(&context));
  ASSERT_NE(d1, d2);
  expect(d1, true);
  expect(d1, true);
  expect(d2, true);
}

// Interface membership plays no part in the answer, so late attachment needs no
// fallback: a type that structurally holds a SymbolRefAttr (a tensor with a
// symbol-ref encoding) is not symbol-ref-free from interning, so verification
// visits it and the newly-attached verifySymbolUses fires.
TEST_F(SymbolReferenceContainmentTest, LateInterfaceAttachmentStillVerifies) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      "module { \"foo.op\"() : () -> tensor<4xf32, @sym> }", &context);
  ASSERT_TRUE(module);

  RankedTensorType::attachInterface<FailingSymbolUserModel<RankedTensorType>>(
      context);
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(failed(verify(*module)));
}

// The contract boundary: a type that references a symbol without spelling it as
// a SymbolRefAttr (here f32, standing in for a non-conforming symbol-user type)
// answers symbol-ref-free and is therefore skipped -- its verifySymbolUses
// never fires, so verification succeeds. This is the documented cost of the
// interface contract.
TEST_F(SymbolReferenceContainmentTest, NonConformingSymbolUserTypeIsSkipped) {
  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      "module { \"foo.op\"() : () -> f32 }", &context);
  ASSERT_TRUE(module);

  Float32Type::attachInterface<FailingSymbolUserModel<Float32Type>>(context);
  ScopedDiagnosticHandler handler(&context,
                                  [](Diagnostic &) { return success(); });
  EXPECT_TRUE(succeeded(verify(*module)));
}

// The corpus both the ground-truth and the raced context are filled from,
// spanning the interesting fill paths and sharing sub-elements so concurrent
// fills overlap on the same interior objects. Built from one helper so the two
// contexts stay in lock-step by construction.
std::vector<Attribute> buildAttrs(MLIRContext &ctx) {
  Type i32 = IntegerType::get(&ctx, 32);
  FlatSymbolRefAttr sym = FlatSymbolRefAttr::get(&ctx, "sym");
  return {
      StringAttr::get(&ctx, "leaf"),
      sym,
      test::TestSymbolRefAttr::get(&ctx, sym),
      TypeAttr::get(test::TestSymbolUserType::get(&ctx, sym)),
      DictionaryAttr::get(
          &ctx, {NamedAttribute(StringAttr::get(&ctx, "callee"), sym)}),
      DictionaryAttr::get(&ctx, {NamedAttribute(StringAttr::get(&ctx, "plain"),
                                                TypeAttr::get(i32))}),
  };
}
std::vector<Type> buildTypes(MLIRContext &ctx) {
  Type i32 = IntegerType::get(&ctx, 32);
  FlatSymbolRefAttr sym = FlatSymbolRefAttr::get(&ctx, "sym");
  test::TestSymbolUserType user = test::TestSymbolUserType::get(&ctx, sym);
  return {
      i32,
      user,
      TupleType::get(&ctx, {i32, user}),
      TupleType::get(&ctx, {i32, i32}),
      RankedTensorType::get({2}, i32, sym),
  };
}

// Concurrency crux: many threads query the same cold, shared objects at once,
// exactly as parallel symbol-table verification fills the context cache from
// several isolated-op workers simultaneously. Every thread must agree with the
// single-threaded ground truth, and the run must be clean under
// ThreadSanitizer. The raced context is built cold (constructed but never
// queried before the threads start) so the fills genuinely race.
TEST_F(SymbolReferenceContainmentTest, ConcurrentFillIsRaceFree) {
  ASSERT_TRUE(context.isMultithreadingEnabled());

  auto &truthCache = detail::getSymbolRefContainmentCache(&context);
  std::vector<Attribute> attrs = buildAttrs(context);
  std::vector<Type> types = buildTypes(context);
  std::vector<bool> attrTruth, typeTruth;
  for (Attribute a : attrs)
    attrTruth.push_back(truthCache.isSymbolRefFree(a));
  for (Type t : types)
    typeTruth.push_back(truthCache.isSymbolRefFree(t));

  MLIRContext raced;
  raced.loadDialect<test::TestDialect>();
  raced.allowUnregisteredDialects();
  ASSERT_TRUE(raced.isMultithreadingEnabled());
  std::vector<Attribute> racedAttrs = buildAttrs(raced);
  std::vector<Type> racedTypes = buildTypes(raced);
  auto &racedCache = detail::getSymbolRefContainmentCache(&raced);

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
        attrResults[i].push_back(racedCache.isSymbolRefFree(a));
      for (Type t : racedTypes)
        typeResults[i].push_back(racedCache.isSymbolRefFree(t));
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

} // namespace
