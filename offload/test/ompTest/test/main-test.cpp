#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include <omp-tools.h>

#include "gtest/gtest.h"

using OS = omptest::ObserveState;
using OAE = omptest::OmptAssertEvent;

TEST(CompareOperatorTests, ThreadBeginIdentity) {
  auto TBInitial =
      OAE::ThreadBegin("dflt", "", OS::always, ompt_thread_initial);
  auto TBWorker = OAE::ThreadBegin("dflt", "", OS::always, ompt_thread_worker);
  auto TBOther = OAE::ThreadBegin("dflt", "", OS::always, ompt_thread_other);
  auto TBUnknown =
      OAE::ThreadBegin("dflt", "", OS::always, ompt_thread_unknown);

  ASSERT_EQ(TBInitial, TBInitial);
  ASSERT_EQ(TBWorker, TBWorker);
  ASSERT_EQ(TBOther, TBOther);
  ASSERT_EQ(TBUnknown, TBUnknown);
}

TEST(CompareOperatorTests, ThreadEndIdentity) {
  auto TE = OAE::ThreadEnd("dflt", "", OS::always);

  ASSERT_EQ(TE, TE);
}

TEST(CompareOperatorTests, ParallelBeginIdentity) {
  auto PBNumT = OAE::ParallelBegin("thrdenable", "", OS::always, 3);

  ASSERT_EQ(PBNumT, PBNumT);
}

TEST(CompareOperatorTests, ParallelEndIdentity) {
  auto PEDflt = OAE::ParallelEnd("dflt", "", OS::always);
  // TODO: Add cases with parallel data set, task data set, flags

  ASSERT_EQ(PEDflt, PEDflt);
}

TEST(CompareOperatorTests, WorkIdentity) {
  auto WDLoopBgn =
      OAE::Work("loopbgn", "", OS::always, ompt_work_loop, ompt_scope_begin);
  auto WDLoopEnd =
      OAE::Work("loobend", "", OS::always, ompt_work_loop, ompt_scope_end);

  ASSERT_EQ(WDLoopBgn, WDLoopBgn);
  ASSERT_EQ(WDLoopEnd, WDLoopEnd);

  auto WDSectionsBgn = OAE::Work("sectionsbgn", "", OS::always,
                                 ompt_work_sections, ompt_scope_begin);
  auto WDSectionsEnd = OAE::Work("sectionsend", "", OS::always,
                                 ompt_work_sections, ompt_scope_end);

  // TODO: singleexecutor, single_other, workshare, distribute, taskloop, scope,
  // loop_static, loop_dynamic, loop_guided, loop_other

  ASSERT_EQ(WDSectionsBgn, WDSectionsBgn);
  ASSERT_EQ(WDSectionsEnd, WDSectionsEnd);
}

TEST(CompareOperatorTests, DispatchIdentity) {
  auto DIDflt = OAE::Dispatch("dflt", "", OS::always);

  ASSERT_EQ(DIDflt, DIDflt);
}

TEST(CompareOperatorTests, TaskCreateIdentity) {
  auto TCDflt = OAE::TaskCreate("dflt", "", OS::always);

  ASSERT_EQ(TCDflt, TCDflt);
}

TEST(CompareOperatorTests, TaskScheduleIdentity) {
  auto TS = OAE::TaskSchedule("dflt", "", OS::always);

  ASSERT_EQ(TS, TS);
}

TEST(CompareOperatorTests, ImplicitTaskIdentity) {
  auto ITDfltBgn =
      OAE::ImplicitTask("dfltbgn", "", OS::always, ompt_scope_begin);
  auto ITDfltEnd = OAE::ImplicitTask("dfltend", "", OS::always, ompt_scope_end);

  ASSERT_EQ(ITDfltBgn, ITDfltBgn);
  ASSERT_EQ(ITDfltEnd, ITDfltEnd);
}

TEST(CompareOperatorTests, SyncRegionIdentity) {
  auto SRDfltBgn =
      OAE::SyncRegion("srdfltbgn", "", OS::always,
                      ompt_sync_region_barrier_explicit, ompt_scope_begin);
  auto SRDfltEnd =
      OAE::SyncRegion("srdfltend", "", OS::always,
                      ompt_sync_region_barrier_explicit, ompt_scope_end);

  ASSERT_EQ(SRDfltBgn, SRDfltBgn);
  ASSERT_EQ(SRDfltEnd, SRDfltEnd);
}

TEST(CompareOperatorTests, TargetIdentity) {
  auto TargetDfltBgn =
      OAE::Target("dfltbgn", "", OS::always, ompt_target, ompt_scope_begin);
  auto TargetDfltEnd =
      OAE::Target("dfltend", "", OS::always, ompt_target, ompt_scope_end);

  ASSERT_EQ(TargetDfltBgn, TargetDfltBgn);
  ASSERT_EQ(TargetDfltEnd, TargetDfltEnd);

  auto TargetDevBgn = OAE::Target("tgtdevbgn", "", OS::always, ompt_target,
                                  ompt_scope_begin, 1);
  auto TargetDevEnd =
      OAE::Target("tgtdevend", "", OS::always, ompt_target, ompt_scope_end, 1);

  ASSERT_EQ(TargetDevBgn, TargetDevBgn);
  ASSERT_EQ(TargetDevEnd, TargetDevEnd);
}

TEST(CompareOperatorTests, BufferRecordIdentity) {
  // Default, no time limit or anything
  auto BRDflt =
      OAE::BufferRecord("dflt", "", OS::always, ompt_callback_target_submit);

  // Minimum time set, no max time
  auto BRMinSet = OAE::BufferRecord("minset", "", OS::always,
                                    ompt_callback_target_submit, 10);

  // Minimum time and maximum time set
  auto BRMinMaxSet = OAE::BufferRecord("minmaxset", "", OS::always,
                                       ompt_callback_target_submit, {10, 100});

  ASSERT_EQ(BRDflt, BRDflt);
  ASSERT_EQ(BRMinSet, BRMinSet);
  ASSERT_EQ(BRMinMaxSet, BRMinMaxSet);
}
