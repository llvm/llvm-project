//===- unittest/ProfileData/CoverageMappingTest.cpp -------------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/CoverageMapping.h"
#include "llvm/ProfileData/CoverageMappingReader.h"
#include "llvm/ProfileData/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

#include <sstream>

using namespace llvm;
using namespace coverage;

static ::testing::AssertionResult NoError(std::error_code EC) {
  if (!EC)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error " << EC.value()
                                       << ": " << EC.message();
}

namespace llvm {
namespace coverage {
void PrintTo(const Counter &C, ::std::ostream *os) {
  if (C.isZero())
    *os << "Zero";
  else if (C.isExpression())
    *os << "Expression " << C.getExpressionID();
  else
    *os << "Counter " << C.getCounterID();
}

void PrintTo(const CoverageSegment &S, ::std::ostream *os) {
  *os << "CoverageSegment(" << S.Line << ", " << S.Col << ", ";
  if (S.HasCount)
    *os << S.Count << ", ";
  *os << (S.IsRegionEntry ? "true" : "false") << ")";
}
}
}

namespace {

struct OneFunctionCoverageReader : CoverageMappingReader {
  StringRef Name;
  uint64_t Hash;
  std::vector<StringRef> Filenames;
  ArrayRef<CounterMappingRegion> Regions;
  bool Done;

  OneFunctionCoverageReader(StringRef Name, uint64_t Hash,
                            ArrayRef<StringRef> Filenames,
                            ArrayRef<CounterMappingRegion> Regions)
      : Name(Name), Hash(Hash), Filenames(Filenames), Regions(Regions),
        Done(false) {}

  std::error_code readNextRecord(CoverageMappingRecord &Record) override {
    if (Done)
      return instrprof_error::eof;
    Done = true;

    Record.FunctionName = Name;
    Record.FunctionHash = Hash;
    Record.Filenames = Filenames;
    Record.Expressions = {};
    Record.MappingRegions = Regions;
    return instrprof_error::success;
  }
};

struct CoverageMappingTest : ::testing::Test {
  StringMap<unsigned> Files;
  unsigned NextFile;
  std::vector<CounterMappingRegion> InputCMRs;

  std::vector<StringRef> OutputFiles;
  std::vector<CounterExpression> OutputExpressions;
  std::vector<CounterMappingRegion> OutputCMRs;

  InstrProfWriter ProfileWriter;
  std::unique_ptr<IndexedInstrProfReader> ProfileReader;

  std::unique_ptr<CoverageMapping> LoadedCoverage;

  void SetUp() override {
    NextFile = 0;
    ProfileWriter.setOutputSparse(false);
  }

  unsigned getFile(StringRef Name) {
    auto R = Files.find(Name);
    if (R != Files.end())
      return R->second;
    Files[Name] = NextFile;
    return NextFile++;
  }

  void addCMR(Counter C, StringRef File, unsigned LS, unsigned CS, unsigned LE,
              unsigned CE) {
    InputCMRs.push_back(
        CounterMappingRegion::makeRegion(C, getFile(File), LS, CS, LE, CE));
  }

  void addExpansionCMR(StringRef File, StringRef ExpandedFile, unsigned LS,
                       unsigned CS, unsigned LE, unsigned CE) {
    InputCMRs.push_back(CounterMappingRegion::makeExpansion(
        getFile(File), getFile(ExpandedFile), LS, CS, LE, CE));
  }

  std::string writeCoverageRegions() {
    SmallVector<unsigned, 8> FileIDs;
    for (const auto &E : Files)
      FileIDs.push_back(E.getValue());
    std::string Coverage;
    llvm::raw_string_ostream OS(Coverage);
    CoverageMappingWriter(FileIDs, None, InputCMRs).write(OS);
    return OS.str();
  }

  void readCoverageRegions(std::string Coverage) {
    SmallVector<StringRef, 8> Filenames;
    for (const auto &E : Files)
      Filenames.push_back(E.getKey());
    RawCoverageMappingReader Reader(Coverage, Filenames, OutputFiles,
                                    OutputExpressions, OutputCMRs);
    ASSERT_TRUE(NoError(Reader.read()));
  }

  void readProfCounts() {
    auto Profile = ProfileWriter.writeBuffer();
    auto ReaderOrErr = IndexedInstrProfReader::create(std::move(Profile));
    ASSERT_TRUE(NoError(ReaderOrErr.getError()));
    ProfileReader = std::move(ReaderOrErr.get());
  }

  void loadCoverageMapping(StringRef FuncName, uint64_t Hash) {
    std::string Regions = writeCoverageRegions();
    readCoverageRegions(Regions);

    SmallVector<StringRef, 8> Filenames;
    for (const auto &E : Files)
      Filenames.push_back(E.getKey());
    OneFunctionCoverageReader CovReader(FuncName, Hash, Filenames, OutputCMRs);
    auto CoverageOrErr = CoverageMapping::load(CovReader, *ProfileReader);
    ASSERT_TRUE(NoError(CoverageOrErr.getError()));
    LoadedCoverage = std::move(CoverageOrErr.get());
  }
};

struct MaybeSparseCoverageMappingTest
    : public CoverageMappingTest,
      public ::testing::WithParamInterface<bool> {
  void SetUp() {
    CoverageMappingTest::SetUp();
    ProfileWriter.setOutputSparse(GetParam());
  }
};

TEST_P(MaybeSparseCoverageMappingTest, basic_write_read) {
  addCMR(Counter::getCounter(0), "foo", 1, 1, 1, 1);
  addCMR(Counter::getCounter(1), "foo", 2, 1, 2, 2);
  addCMR(Counter::getZero(),     "foo", 3, 1, 3, 4);
  addCMR(Counter::getCounter(2), "foo", 4, 1, 4, 8);
  addCMR(Counter::getCounter(3), "bar", 1, 2, 3, 4);
  std::string Coverage = writeCoverageRegions();
  readCoverageRegions(Coverage);

  size_t N = makeArrayRef(InputCMRs).size();
  ASSERT_EQ(N, OutputCMRs.size());
  for (size_t I = 0; I < N; ++I) {
    ASSERT_EQ(InputCMRs[I].Count,      OutputCMRs[I].Count);
    ASSERT_EQ(InputCMRs[I].FileID,     OutputCMRs[I].FileID);
    ASSERT_EQ(InputCMRs[I].startLoc(), OutputCMRs[I].startLoc());
    ASSERT_EQ(InputCMRs[I].endLoc(),   OutputCMRs[I].endLoc());
    ASSERT_EQ(InputCMRs[I].Kind,       OutputCMRs[I].Kind);
  }
}

TEST_P(MaybeSparseCoverageMappingTest, expansion_gets_first_counter) {
  addCMR(Counter::getCounter(1), "foo", 10, 1, 10, 2);
  // This starts earlier in "foo", so the expansion should get its counter.
  addCMR(Counter::getCounter(2), "foo", 1, 1, 20, 1);
  addExpansionCMR("bar", "foo", 3, 3, 3, 3);
  std::string Coverage = writeCoverageRegions();
  readCoverageRegions(Coverage);

  ASSERT_EQ(CounterMappingRegion::ExpansionRegion, OutputCMRs[2].Kind);
  ASSERT_EQ(Counter::getCounter(2), OutputCMRs[2].Count);
  ASSERT_EQ(3U, OutputCMRs[2].LineStart);
}

TEST_P(MaybeSparseCoverageMappingTest, basic_coverage_iteration) {
  InstrProfRecord Record("func", 0x1234, {30, 20, 10, 0});
  ProfileWriter.addRecord(std::move(Record));
  readProfCounts();

  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 4, 7);
  addCMR(Counter::getCounter(2), "file1", 5, 8, 9, 1);
  addCMR(Counter::getCounter(3), "file1", 10, 10, 11, 11);
  loadCoverageMapping("func", 0x1234);

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(7U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 20, true),  Segments[0]);
  ASSERT_EQ(CoverageSegment(4, 7, 30, false), Segments[1]);
  ASSERT_EQ(CoverageSegment(5, 8, 10, true),  Segments[2]);
  ASSERT_EQ(CoverageSegment(9, 1, 30, false), Segments[3]);
  ASSERT_EQ(CoverageSegment(9, 9, false),     Segments[4]);
  ASSERT_EQ(CoverageSegment(10, 10, 0, true), Segments[5]);
  ASSERT_EQ(CoverageSegment(11, 11, false),   Segments[6]);
}

TEST_P(MaybeSparseCoverageMappingTest, uncovered_function) {
  readProfCounts();

  addCMR(Counter::getZero(), "file1", 1, 2, 3, 4);
  loadCoverageMapping("func", 0x1234);

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(2U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 2, 0, true), Segments[0]);
  ASSERT_EQ(CoverageSegment(3, 4, false),   Segments[1]);
}

TEST_P(MaybeSparseCoverageMappingTest, uncovered_function_with_mapping) {
  readProfCounts();

  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 4, 7);
  loadCoverageMapping("func", 0x1234);

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(3U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 0, true),  Segments[0]);
  ASSERT_EQ(CoverageSegment(4, 7, 0, false), Segments[1]);
  ASSERT_EQ(CoverageSegment(9, 9, false),    Segments[2]);
}

TEST_P(MaybeSparseCoverageMappingTest, combine_regions) {
  InstrProfRecord Record("func", 0x1234, {10, 20, 30});
  ProfileWriter.addRecord(std::move(Record));
  readProfCounts();

  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 3, 3, 4, 4);
  addCMR(Counter::getCounter(2), "file1", 3, 3, 4, 4);
  loadCoverageMapping("func", 0x1234);

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(4U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 10, true), Segments[0]);
  ASSERT_EQ(CoverageSegment(3, 3, 50, true), Segments[1]);
  ASSERT_EQ(CoverageSegment(4, 4, 10, false), Segments[2]);
  ASSERT_EQ(CoverageSegment(9, 9, false), Segments[3]);
}

TEST_P(MaybeSparseCoverageMappingTest, dont_combine_expansions) {
  InstrProfRecord Record1("func", 0x1234, {10, 20});
  InstrProfRecord Record2("func", 0x1234, {0, 0});
  ProfileWriter.addRecord(std::move(Record1));
  ProfileWriter.addRecord(std::move(Record2));
  readProfCounts();

  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 3, 3, 4, 4);
  addCMR(Counter::getCounter(1), "include1", 6, 6, 7, 7);
  addExpansionCMR("file1", "include1", 3, 3, 4, 4);
  loadCoverageMapping("func", 0x1234);

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(4U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 10, true), Segments[0]);
  ASSERT_EQ(CoverageSegment(3, 3, 20, true), Segments[1]);
  ASSERT_EQ(CoverageSegment(4, 4, 10, false), Segments[2]);
  ASSERT_EQ(CoverageSegment(9, 9, false), Segments[3]);
}

TEST_P(MaybeSparseCoverageMappingTest, strip_filename_prefix) {
  InstrProfRecord Record("file1:func", 0x1234, {0});
  ProfileWriter.addRecord(std::move(Record));
  readProfCounts();

  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  loadCoverageMapping("file1:func", 0x1234);

  std::vector<std::string> Names;
  for (const auto &Func : LoadedCoverage->getCoveredFunctions())
    Names.push_back(Func.Name);
  ASSERT_EQ(1U, Names.size());
  ASSERT_EQ("func", Names[0]);
}

INSTANTIATE_TEST_CASE_P(MaybeSparse, MaybeSparseCoverageMappingTest,
                        ::testing::Bool());

} // end anonymous namespace
