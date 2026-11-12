//===- RemarkPolicyTest.cpp - Remark policy tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Remarks.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/llvm-config.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <atomic>
#include <string>
#include <thread>
#include <vector>

using namespace mlir;
using namespace testing;

namespace {
using remark::RemarkId;
using remark::detail::Remark;

static std::vector<std::string> relatedArgs(ArrayRef<Remark::Arg> args) {
  std::vector<std::string> values;
  for (const auto &arg : args)
    if (arg.key == "RelatedTo")
      values.push_back(arg.val);
  return values;
}

static std::vector<std::string>
relatedArgs(const llvm::remarks::Remark &remark) {
  std::vector<std::string> values;
  for (const auto &arg : remark.Args)
    if (arg.Key == "RelatedTo")
      values.push_back(arg.Val.str());
  return values;
}

class FinalRemarkPolicyTest : public Test {
protected:
  void SetUp() override {
    policy.initialize(
        [this](const Remark &remark) { emitted.push_back(remark); });
  }

  Remark makeRemark(StringRef name, uint64_t id,
                    ArrayRef<uint64_t> related = {}) {
    Remark result(remark::RemarkKind::RemarkMissed, DiagnosticSeverity::Remark,
                  loc, remark::RemarkOpts::name(name).category("Opt"));
    result.setId(RemarkId(id));
    for (uint64_t target : related)
      result.addRelatedRemark(RemarkId(target));
    return result;
  }

  void report(StringRef name, uint64_t id, ArrayRef<uint64_t> related = {}) {
    policy.reportRemark(makeRemark(name, id, related));
  }

  SmallVector<uint64_t> emittedIds() const {
    SmallVector<uint64_t> ids;
    for (const Remark &remark : emitted)
      ids.push_back(remark.getId().getValue());
    return ids;
  }

  void expectLinks(uint64_t id, ArrayRef<uint64_t> expected,
                   ArrayRef<StringRef> userArgs = {}) {
    auto it = llvm::find_if(emitted, [&](const Remark &remark) {
      return remark.getId().getValue() == id;
    });
    ASSERT_NE(it, emitted.end());
    SmallVector<uint64_t> actual;
    for (RemarkId related : it->getRelatedRemarkIds())
      actual.push_back(related.getValue());
    EXPECT_THAT(actual, ElementsAreArray(expected));

    std::vector<std::string> expectedArgs;
    for (StringRef value : userArgs)
      expectedArgs.push_back(value.str());
    for (uint64_t related : expected)
      expectedArgs.push_back(std::to_string(related));
    EXPECT_THAT(relatedArgs(it->getArgs()),
                UnorderedElementsAreArray(expectedArgs));
    EXPECT_THAT(relatedArgs(it->generateRemark()),
                UnorderedElementsAreArray(expectedArgs));
  }

  MLIRContext context;
  Location loc = UnknownLoc::get(&context);
  remark::RemarkEmittingPolicyFinal policy;
  std::vector<Remark> emitted;
};

TEST_F(FinalRemarkPolicyTest, DiscardedTargetDoesNotDiscardOrRetargetConsumer) {
  report("Candidate", 1);
  auto consumer = makeRemark("Consumer", 2, {1});
  EXPECT_THAT(relatedArgs(consumer.getArgs()), ElementsAre("1"));
  policy.reportRemark(consumer);
  auto withUserArgs = makeRemark("ConsumerWithUserArgs", 3, {1});
  withUserArgs.insert(Remark::Arg("RelatedTo", "1"));
  withUserArgs.insert(Remark::Arg("RelatedTo", "caller-supplied value"));
  policy.reportRemark(withUserArgs);
  report("Candidate", 4);

  policy.finalize();
  EXPECT_THAT(emittedIds(), ElementsAre(2, 3, 4));
  expectLinks(2, {});
  expectLinks(3, {}, {"1", "caller-supplied value"});
  ASSERT_EQ(emitted.size(), 3u);
  EXPECT_THAT(emitted.front().getMsg(), Not(HasSubstr("RelatedTo=")));
  EXPECT_THAT(emitted[1].getMsg(), HasSubstr("RelatedTo=1"));
  EXPECT_THAT(emitted[1].getMsg(),
              HasSubstr("RelatedTo=\"caller-supplied value\""));
}

TEST_F(FinalRemarkPolicyTest, LinkToReplacementSurvives) {
  report("Candidate", 1);
  report("Candidate", 2);
  report("Consumer", 3, {2});

  policy.finalize();
  EXPECT_THAT(emittedIds(), ElementsAre(3, 2));
  expectLinks(3, {2});
}

TEST_F(FinalRemarkPolicyTest, DiscardedIDsAreScopedToOneBatch) {
  report("Candidate", 1);
  report("Candidate", 2);
  report("Consumer", 3, {1});
  report("EarlierTarget", 4);
  // IDs 90 and 91 are unknown in this batch. The policy preserves them.
  report("ExternalConsumer", 5, {4, 90, 91});
  policy.finalize();
  expectLinks(3, {});
  expectLinks(5, {4, 90, 91});
  EXPECT_THAT(emittedIds(), Not(Contains(1)));

  // The previous batch's overwritten ID is no longer in this batch's scope.
  report("NextConsumer", 6, {1, 4, 90, 91});
  report("FutureTarget", 91);
  policy.finalize();
  expectLinks(6, {1, 4, 90, 91});
  EXPECT_EQ(llvm::count(emittedIds(), 4), 1);
  EXPECT_EQ(llvm::count(emittedIds(), 91), 1);
  size_t count = emitted.size();
  policy.finalize();
  EXPECT_EQ(emitted.size(), count);
}

TEST_F(FinalRemarkPolicyTest, RereportedAndSurvivingIDsRemainValid) {
  auto target = makeRemark("Target", 1);
  policy.reportRemark(target);
  target.insert("reported again");
  policy.reportRemark(target);
  report("Consumer", 2, {1});

  report("ReplacedIdentity", 3);
  report("ReplacedIdentity", 4);
  // A surviving event with ID 3 remains a valid target despite the overwrite.
  report("SurvivingIdentity", 3);
  report("OtherConsumer", 5, {3});
  policy.finalize();

  EXPECT_THAT(emittedIds(), UnorderedElementsAre(1, 2, 3, 4, 5));
  expectLinks(2, {1});
  expectLinks(5, {3});
}

TEST_F(FinalRemarkPolicyTest, TransitiveSharedAndCyclicTargetsEmitOnce) {
  report("Leaf", 1);
  report("Middle", 2, {1});
  report("Root", 3, {2});
  report("OtherRoot", 4, {1});
  report("CycleA", 5, {6});
  report("CycleB", 6, {5});
  report("Self", 7, {7});
  policy.finalize();

  EXPECT_THAT(emittedIds(), ElementsAre(3, 2, 1, 4, 5, 6, 7));
  expectLinks(2, {1});
  expectLinks(3, {2});
  expectLinks(4, {1});
  expectLinks(5, {6});
  expectLinks(6, {5});
  expectLinks(7, {7});
}

TEST_F(FinalRemarkPolicyTest, RemarksWithoutIDsSurvive) {
  report("FirstWithoutID", 0);
  report("SecondWithoutID", 0);
  report("WithID", 1);
  policy.finalize();

  EXPECT_THAT(emittedIds(), ElementsAre(1, 0, 0));
  std::vector<std::string> names;
  for (const Remark &remark : emitted)
    names.push_back(remark.getRemarkName().str());
  EXPECT_THAT(names, UnorderedElementsAre("FirstWithoutID", "SecondWithoutID",
                                          "WithID"));
}

TEST_F(FinalRemarkPolicyTest, RelatedLinksSerializeAndCopy) {
  auto original = makeRemark("Original", 1, {2});
  original.insert(Remark::Arg("RelatedTo", "caller"));
  original.insert(Remark::Arg("Payload", std::string(256, 'x')));
  auto args = original.getArgs();
  auto serialized = original.generateRemark();
  EXPECT_THAT(original.getMsg(), HasSubstr("RelatedTo=2"));
  EXPECT_THAT(relatedArgs(args), UnorderedElementsAre("caller", "2"));
  EXPECT_THAT(relatedArgs(serialized), UnorderedElementsAre("caller", "2"));

  Remark copy = original;
  copy.addRelatedRemark(RemarkId(3));
  copy.insert(Remark::Arg("RelatedTo", "copy"));
  auto copyArgs = copy.getArgs();
  EXPECT_THAT(relatedArgs(copyArgs),
              UnorderedElementsAre("caller", "copy", "2", "3"));
  Remark moved = std::move(copy);
  EXPECT_THAT(relatedArgs(moved.generateRemark()),
              UnorderedElementsAre("caller", "copy", "2", "3"));
  EXPECT_THAT(relatedArgs(serialized), UnorderedElementsAre("caller", "2"));

  // Mutations leave the copied remark unchanged.
  original.addRelatedRemark(RemarkId(4));
  original.insert(Remark::Arg("RelatedTo", "new caller"));
  EXPECT_THAT(relatedArgs(original.getArgs()),
              UnorderedElementsAre("caller", "new caller", "2", "4"));
  EXPECT_THAT(relatedArgs(original.generateRemark()),
              UnorderedElementsAre("caller", "new caller", "2", "4"));
  EXPECT_THAT(relatedArgs(moved.getArgs()),
              UnorderedElementsAre("caller", "copy", "2", "3"));

  moved.setRelatedRemarkIds(moved.getRelatedRemarkIds().drop_front());
  EXPECT_THAT(relatedArgs(moved.generateRemark()),
              UnorderedElementsAre("caller", "copy", "3"));
  moved.setRelatedRemarkIds({});
  EXPECT_THAT(relatedArgs(moved.getArgs()),
              UnorderedElementsAre("caller", "copy"));
}

#if LLVM_ENABLE_THREADS
TEST_F(FinalRemarkPolicyTest, ConcurrentReportsDeduplicate) {
  constexpr unsigned numWriters = 4;
  constexpr unsigned numReports = 64;
  std::atomic<bool> start{false};
  std::vector<std::thread> writers;
  for (unsigned writer = 0; writer < numWriters; ++writer) {
    writers.emplace_back([&, writer] {
      std::string name = "Writer" + std::to_string(writer);
      while (!start.load())
        std::this_thread::yield();
      for (unsigned reportIndex = 1; reportIndex <= numReports; ++reportIndex)
        report(name, writer * numReports + reportIndex);
    });
  }
  start.store(true);
  for (auto &writer : writers)
    writer.join();
  policy.finalize();

  EXPECT_THAT(emittedIds(), ElementsAre(64, 128, 192, 256));
}

#endif
} // namespace
