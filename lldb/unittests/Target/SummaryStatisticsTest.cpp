#include "lldb/DataFormatters/TypeSummary.h"
#include "lldb/Target/Statistics.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-private-enumerations.h"
#include "lldb/lldb-private.h"
#include "lldb/Utility/Stream.h"
#include "llvm/Testing/Support/Error.h"
#include "gtest/gtest.h"
#include <thread>

using namespace lldb_private;
using Duration = std::chrono::duration<double>;

class DummySummaryImpl : public lldb_private::TypeSummaryImpl {
public:
  DummySummaryImpl(Duration sleepTime):
    TypeSummaryImpl(TypeSummaryImpl::Kind::eSummaryString, TypeSummaryImpl::Flags()),
     m_sleepTime(sleepTime) {}

  std::string GetName() override {
    return "DummySummary";
  }

  std::string GetSummaryKindName() override {
    return "dummy";
  }

  std::string GetDescription() override {
    return "";
  } 

  bool FormatObject(ValueObject *valobj, std::string &dest,
                    const TypeSummaryOptions &options) override {
    return false;
  }

  void FakeFormat() {
    std::this_thread::sleep_for(m_sleepTime);
  }

private:
  Duration m_sleepTime;
};

TEST(MultithreadFormatting, Multithread) {
  SummaryStatisticsCache statistics_cache;
  DummySummaryImpl summary(Duration(1));
  std::vector<std::thread> threads;
  for (int i = 0; i < 10; ++i) {
    threads.emplace_back(std::thread([&statistics_cache, &summary]() {
      auto sp = statistics_cache.GetSummaryStatisticsForProvider(summary);
      {
        SummaryStatistics::SummaryInvocation invocation(sp);
        summary.FakeFormat();
      }
    }));
  }

  for (auto &thread : threads) 
    thread.join();

  auto sp = statistics_cache.GetSummaryStatisticsForProvider(summary);
  ASSERT_TRUE(sp->GetDurationReference().get().count() > 10);
  ASSERT_TRUE(sp->GetSummaryCount() == 10);

  std::string stats_as_json;
  llvm::raw_string_ostream ss(stats_as_json);
  ss << sp->ToJSON();
  ASSERT_THAT(stats_as_json, ::testing::HasSubstr("\"name\":\"DummySummary\""));
  ASSERT_THAT(stats_as_json, ::testing::HasSubstr("\"type\":\"dummy\""));
}
