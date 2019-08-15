//===- unittest/ProfileData/CoverageMappingTest.cpp -------------------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/Coverage/CoverageMappingWriter.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ProfileData/InstrProfWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

#include <ostream>
#include <utility>

using namespace llvm;
using namespace coverage;

LLVM_NODISCARD static ::testing::AssertionResult
ErrorEquals(coveragemap_error Expected, Error E) {
  coveragemap_error Found;
  std::string FoundMsg;
  handleAllErrors(std::move(E), [&](const CoverageMapError &CME) {
    Found = CME.get();
    FoundMsg = CME.message();
  });
  if (Expected == Found)
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure() << "error: " << FoundMsg << "\n";
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

struct OutputFunctionCoverageData {
  StringRef Name;
  uint64_t Hash;
  std::vector<StringRef> Filenames;
  std::vector<CounterMappingRegion> Regions;

  OutputFunctionCoverageData() : Hash(0) {}

  OutputFunctionCoverageData(OutputFunctionCoverageData &&OFCD)
      : Name(OFCD.Name), Hash(OFCD.Hash), Filenames(std::move(OFCD.Filenames)),
        Regions(std::move(OFCD.Regions)) {}

  OutputFunctionCoverageData(const OutputFunctionCoverageData &) = delete;
  OutputFunctionCoverageData &
  operator=(const OutputFunctionCoverageData &) = delete;
  OutputFunctionCoverageData &operator=(OutputFunctionCoverageData &&) = delete;

  void fillCoverageMappingRecord(CoverageMappingRecord &Record) const {
    Record.FunctionName = Name;
    Record.FunctionHash = Hash;
    Record.Filenames = Filenames;
    Record.Expressions = {};
    Record.MappingRegions = Regions;
  }
};

struct CoverageMappingReaderMock : CoverageMappingReader {
  ArrayRef<OutputFunctionCoverageData> Functions;

  CoverageMappingReaderMock(ArrayRef<OutputFunctionCoverageData> Functions)
      : Functions(Functions) {}

  Error readNextRecord(CoverageMappingRecord &Record) override {
    if (Functions.empty())
      return make_error<CoverageMapError>(coveragemap_error::eof);

    Functions.front().fillCoverageMappingRecord(Record);
    Functions = Functions.slice(1);

    return Error::success();
  }
};

struct InputFunctionCoverageData {
  // Maps the global file index from CoverageMappingTest.Files
  // to the index of that file within this function. We can't just use
  // global file indexes here because local indexes have to be dense.
  // This map is used during serialization to create the virtual file mapping
  // (from local fileId to global Index) in the head of the per-function
  // coverage mapping data.
  SmallDenseMap<unsigned, unsigned> ReverseVirtualFileMapping;
  std::string Name;
  uint64_t Hash;
  std::vector<CounterMappingRegion> Regions;

  InputFunctionCoverageData(std::string Name, uint64_t Hash)
      : Name(std::move(Name)), Hash(Hash) {}

  InputFunctionCoverageData(InputFunctionCoverageData &&IFCD)
      : ReverseVirtualFileMapping(std::move(IFCD.ReverseVirtualFileMapping)),
        Name(std::move(IFCD.Name)), Hash(IFCD.Hash),
        Regions(std::move(IFCD.Regions)) {}

  InputFunctionCoverageData(const InputFunctionCoverageData &) = delete;
  InputFunctionCoverageData &
  operator=(const InputFunctionCoverageData &) = delete;
  InputFunctionCoverageData &operator=(InputFunctionCoverageData &&) = delete;
};

struct CoverageMappingTest : ::testing::TestWithParam<std::pair<bool, bool>> {
  bool UseMultipleReaders;
  StringMap<unsigned> Files;
  std::vector<InputFunctionCoverageData> InputFunctions;
  std::vector<OutputFunctionCoverageData> OutputFunctions;

  InstrProfWriter ProfileWriter;
  std::unique_ptr<IndexedInstrProfReader> ProfileReader;

  std::unique_ptr<CoverageMapping> LoadedCoverage;

  void SetUp() override {
    ProfileWriter.setOutputSparse(GetParam().first);
    UseMultipleReaders = GetParam().second;
  }

  unsigned getGlobalFileIndex(StringRef Name) {
    auto R = Files.find(Name);
    if (R != Files.end())
      return R->second;
    unsigned Index = Files.size();
    Files.try_emplace(Name, Index);
    return Index;
  }

  // Return the file index of file 'Name' for the current function.
  // Add the file into the global map if necessary.
  // See also InputFunctionCoverageData::ReverseVirtualFileMapping
  // for additional comments.
  unsigned getFileIndexForFunction(StringRef Name) {
    unsigned GlobalIndex = getGlobalFileIndex(Name);
    auto &CurrentFunctionFileMapping =
        InputFunctions.back().ReverseVirtualFileMapping;
    auto R = CurrentFunctionFileMapping.find(GlobalIndex);
    if (R != CurrentFunctionFileMapping.end())
      return R->second;
    unsigned IndexInFunction = CurrentFunctionFileMapping.size();
    CurrentFunctionFileMapping.insert(
        std::make_pair(GlobalIndex, IndexInFunction));
    return IndexInFunction;
  }

  void startFunction(StringRef FuncName, uint64_t Hash) {
    InputFunctions.emplace_back(FuncName.str(), Hash);
  }

  void addCMR(Counter C, StringRef File, unsigned LS, unsigned CS, unsigned LE,
              unsigned CE, bool Skipped = false) {
    auto &Regions = InputFunctions.back().Regions;
    unsigned FileID = getFileIndexForFunction(File);
    Regions.push_back(
        Skipped ? CounterMappingRegion::makeSkipped(FileID, LS, CS, LE, CE)
                : CounterMappingRegion::makeRegion(C, FileID, LS, CS, LE, CE));
  }

  void addExpansionCMR(StringRef File, StringRef ExpandedFile, unsigned LS,
                       unsigned CS, unsigned LE, unsigned CE) {
    InputFunctions.back().Regions.push_back(CounterMappingRegion::makeExpansion(
        getFileIndexForFunction(File), getFileIndexForFunction(ExpandedFile),
        LS, CS, LE, CE));
  }

  std::string writeCoverageRegions(InputFunctionCoverageData &Data) {
    SmallVector<unsigned, 8> FileIDs(Data.ReverseVirtualFileMapping.size());
    for (const auto &E : Data.ReverseVirtualFileMapping)
      FileIDs[E.second] = E.first;
    std::string Coverage;
    llvm::raw_string_ostream OS(Coverage);
    CoverageMappingWriter(FileIDs, None, Data.Regions).write(OS);
    return OS.str();
  }

  void readCoverageRegions(const std::string &Coverage,
                           OutputFunctionCoverageData &Data) {
    SmallVector<StringRef, 8> Filenames(Files.size());
    for (const auto &E : Files)
      Filenames[E.getValue()] = E.getKey();
    std::vector<CounterExpression> Expressions;
    RawCoverageMappingReader Reader(Coverage, Filenames, Data.Filenames,
                                    Expressions, Data.Regions);
    EXPECT_THAT_ERROR(Reader.read(), Succeeded());
  }

  void writeAndReadCoverageRegions(bool EmitFilenames = true) {
    OutputFunctions.resize(InputFunctions.size());
    for (unsigned I = 0; I < InputFunctions.size(); ++I) {
      std::string Regions = writeCoverageRegions(InputFunctions[I]);
      readCoverageRegions(Regions, OutputFunctions[I]);
      OutputFunctions[I].Name = InputFunctions[I].Name;
      OutputFunctions[I].Hash = InputFunctions[I].Hash;
      if (!EmitFilenames)
        OutputFunctions[I].Filenames.clear();
    }
  }

  void readProfCounts() {
    auto Profile = ProfileWriter.writeBuffer();
    auto ReaderOrErr = IndexedInstrProfReader::create(std::move(Profile));
    EXPECT_THAT_ERROR(ReaderOrErr.takeError(), Succeeded());
    ProfileReader = std::move(ReaderOrErr.get());
  }

  Expected<std::unique_ptr<CoverageMapping>> readOutputFunctions() {
    std::vector<std::unique_ptr<CoverageMappingReader>> CoverageReaders;
    if (UseMultipleReaders) {
      for (const auto &OF : OutputFunctions) {
        ArrayRef<OutputFunctionCoverageData> Funcs(OF);
        CoverageReaders.push_back(
            std::make_unique<CoverageMappingReaderMock>(Funcs));
      }
    } else {
      ArrayRef<OutputFunctionCoverageData> Funcs(OutputFunctions);
      CoverageReaders.push_back(
          std::make_unique<CoverageMappingReaderMock>(Funcs));
    }
    return CoverageMapping::load(CoverageReaders, *ProfileReader);
  }

  Error loadCoverageMapping(bool EmitFilenames = true) {
    readProfCounts();
    writeAndReadCoverageRegions(EmitFilenames);
    auto CoverageOrErr = readOutputFunctions();
    if (!CoverageOrErr)
      return CoverageOrErr.takeError();
    LoadedCoverage = std::move(CoverageOrErr.get());
    return Error::success();
  }
};

TEST_P(CoverageMappingTest, basic_write_read) {
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "foo", 1, 1, 1, 1);
  addCMR(Counter::getCounter(1), "foo", 2, 1, 2, 2);
  addCMR(Counter::getZero(),     "foo", 3, 1, 3, 4);
  addCMR(Counter::getCounter(2), "foo", 4, 1, 4, 8);
  addCMR(Counter::getCounter(3), "bar", 1, 2, 3, 4);

  writeAndReadCoverageRegions();
  ASSERT_EQ(1u, InputFunctions.size());
  ASSERT_EQ(1u, OutputFunctions.size());
  InputFunctionCoverageData &Input = InputFunctions.back();
  OutputFunctionCoverageData &Output = OutputFunctions.back();

  size_t N = makeArrayRef(Input.Regions).size();
  ASSERT_EQ(N, Output.Regions.size());
  for (size_t I = 0; I < N; ++I) {
    ASSERT_EQ(Input.Regions[I].Count, Output.Regions[I].Count);
    ASSERT_EQ(Input.Regions[I].FileID, Output.Regions[I].FileID);
    ASSERT_EQ(Input.Regions[I].startLoc(), Output.Regions[I].startLoc());
    ASSERT_EQ(Input.Regions[I].endLoc(), Output.Regions[I].endLoc());
    ASSERT_EQ(Input.Regions[I].Kind, Output.Regions[I].Kind);
  }
}

TEST_P(CoverageMappingTest, correct_deserialize_for_more_than_two_files) {
  const char *FileNames[] = {"bar", "baz", "foo"};
  static const unsigned N = array_lengthof(FileNames);

  startFunction("func", 0x1234);
  for (unsigned I = 0; I < N; ++I)
    // Use LineStart to hold the index of the file name
    // in order to preserve that information during possible sorting of CMRs.
    addCMR(Counter::getCounter(0), FileNames[I], I, 1, I, 1);

  writeAndReadCoverageRegions();
  ASSERT_EQ(1u, OutputFunctions.size());
  OutputFunctionCoverageData &Output = OutputFunctions.back();

  ASSERT_EQ(N, Output.Regions.size());
  ASSERT_EQ(N, Output.Filenames.size());

  for (unsigned I = 0; I < N; ++I) {
    ASSERT_GT(N, Output.Regions[I].FileID);
    ASSERT_GT(N, Output.Regions[I].LineStart);
    EXPECT_EQ(FileNames[Output.Regions[I].LineStart],
              Output.Filenames[Output.Regions[I].FileID]);
  }
}

static const auto Err = [](Error E) { FAIL(); };

TEST_P(CoverageMappingTest, load_coverage_for_more_than_two_files) {
  ProfileWriter.addRecord({"func", 0x1234, {0}}, Err);

  const char *FileNames[] = {"bar", "baz", "foo"};
  static const unsigned N = array_lengthof(FileNames);

  startFunction("func", 0x1234);
  for (unsigned I = 0; I < N; ++I)
    // Use LineStart to hold the index of the file name
    // in order to preserve that information during possible sorting of CMRs.
    addCMR(Counter::getCounter(0), FileNames[I], I, 1, I, 1);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  for (unsigned I = 0; I < N; ++I) {
    CoverageData Data = LoadedCoverage->getCoverageForFile(FileNames[I]);
    ASSERT_TRUE(!Data.empty());
    EXPECT_EQ(I, Data.begin()->Line);
  }
}

TEST_P(CoverageMappingTest, load_coverage_with_bogus_function_name) {
  ProfileWriter.addRecord({"", 0x1234, {10}}, Err);
  startFunction("", 0x1234);
  addCMR(Counter::getCounter(0), "foo", 1, 1, 5, 5);
  EXPECT_TRUE(ErrorEquals(coveragemap_error::malformed, loadCoverageMapping()));
}

TEST_P(CoverageMappingTest, load_coverage_for_several_functions) {
  ProfileWriter.addRecord({"func1", 0x1234, {10}}, Err);
  ProfileWriter.addRecord({"func2", 0x2345, {20}}, Err);

  startFunction("func1", 0x1234);
  addCMR(Counter::getCounter(0), "foo", 1, 1, 5, 5);

  startFunction("func2", 0x2345);
  addCMR(Counter::getCounter(0), "bar", 2, 2, 6, 6);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  EXPECT_EQ(2, std::distance(FunctionRecords.begin(), FunctionRecords.end()));
  for (const auto &FunctionRecord : FunctionRecords) {
    CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
    std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
    ASSERT_EQ(2U, Segments.size());
    if (FunctionRecord.Name == "func1") {
      EXPECT_EQ(CoverageSegment(1, 1, 10, true), Segments[0]);
      EXPECT_EQ(CoverageSegment(5, 5, false), Segments[1]);
    } else {
      ASSERT_EQ("func2", FunctionRecord.Name);
      EXPECT_EQ(CoverageSegment(2, 2, 20, true), Segments[0]);
      EXPECT_EQ(CoverageSegment(6, 6, false), Segments[1]);
    }
  }
}

TEST_P(CoverageMappingTest, create_combined_regions) {
  ProfileWriter.addRecord({"func1", 0x1234, {1, 2, 3}}, Err);
  startFunction("func1", 0x1234);

  // Given regions which start at the same location, emit a segment for the
  // last region.
  addCMR(Counter::getCounter(0), "file1", 1, 1, 2, 2);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 2, 2);
  addCMR(Counter::getCounter(2), "file1", 1, 1, 2, 2);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(2U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 6, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(2, 2, false), Segments[1]);
}

TEST_P(CoverageMappingTest, skipped_segments_have_no_count) {
  ProfileWriter.addRecord({"func1", 0x1234, {1}}, Err);
  startFunction("func1", 0x1234);

  addCMR(Counter::getCounter(0), "file1", 1, 1, 5, 5);
  addCMR(Counter::getCounter(0), "file1", 5, 1, 5, 5, /*Skipped=*/true);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(3U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 1, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(5, 1, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(5, 5, false), Segments[2]);
}

TEST_P(CoverageMappingTest, multiple_regions_end_after_parent_ends) {
  ProfileWriter.addRecord({"func1", 0x1234, {1, 0}}, Err);
  startFunction("func1", 0x1234);

  // 1| F{ a{
  // 2|
  // 3|    a} b{ c{
  // 4|
  // 5|    b}
  // 6|
  // 7| c} d{   e{
  // 8|
  // 9| d}      e} F}
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9); // < F
  addCMR(Counter::getCounter(0), "file1", 1, 1, 3, 5); // < a
  addCMR(Counter::getCounter(0), "file1", 3, 5, 5, 4); // < b
  addCMR(Counter::getCounter(1), "file1", 3, 5, 7, 3); // < c
  addCMR(Counter::getCounter(1), "file1", 7, 3, 9, 2); // < d
  addCMR(Counter::getCounter(1), "file1", 7, 7, 9, 7); // < e

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  // Old output (not sorted or unique):
  //   Segment at 1:1 with count 1
  //   Segment at 1:1 with count 1
  //   Segment at 3:5 with count 1
  //   Segment at 3:5 with count 0
  //   Segment at 3:5 with count 1
  //   Segment at 5:4 with count 0
  //   Segment at 7:3 with count 1
  //   Segment at 7:3 with count 0
  //   Segment at 7:7 with count 0
  //   Segment at 9:7 with count 0
  //   Segment at 9:2 with count 1
  //   Top level segment at 9:9

  // New output (sorted and unique):
  //   Segment at 1:1 (count = 1), RegionEntry
  //   Segment at 3:5 (count = 1), RegionEntry
  //   Segment at 5:4 (count = 0)
  //   Segment at 7:3 (count = 0), RegionEntry
  //   Segment at 7:7 (count = 0), RegionEntry
  //   Segment at 9:2 (count = 0)
  //   Segment at 9:7 (count = 1)
  //   Segment at 9:9 (count = 0), Skipped

  ASSERT_EQ(8U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 1, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(3, 5, 1, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(5, 4, 0, false), Segments[2]);
  EXPECT_EQ(CoverageSegment(7, 3, 0, true), Segments[3]);
  EXPECT_EQ(CoverageSegment(7, 7, 0, true), Segments[4]);
  EXPECT_EQ(CoverageSegment(9, 2, 0, false), Segments[5]);
  EXPECT_EQ(CoverageSegment(9, 7, 1, false), Segments[6]);
  EXPECT_EQ(CoverageSegment(9, 9, false), Segments[7]);
}

TEST_P(CoverageMappingTest, multiple_completed_segments_at_same_loc) {
  ProfileWriter.addRecord({"func1", 0x1234, {0, 1, 2}}, Err);
  startFunction("func1", 0x1234);

  // PR35495
  addCMR(Counter::getCounter(1), "file1", 2, 1, 18, 2);
  addCMR(Counter::getCounter(0), "file1", 8, 10, 14, 6);
  addCMR(Counter::getCounter(0), "file1", 8, 12, 14, 6);
  addCMR(Counter::getCounter(1), "file1", 9, 1, 14, 6);
  addCMR(Counter::getCounter(2), "file1", 11, 13, 11, 14);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(7U, Segments.size());
  EXPECT_EQ(CoverageSegment(2, 1, 1, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(8, 10, 0, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(8, 12, 0, true), Segments[2]);
  EXPECT_EQ(CoverageSegment(9, 1, 1, true), Segments[3]);
  EXPECT_EQ(CoverageSegment(11, 13, 2, true), Segments[4]);
  // Use count=1 (from 9:1 -> 14:6), not count=0 (from 8:12 -> 14:6).
  EXPECT_EQ(CoverageSegment(11, 14, 1, false), Segments[5]);
  EXPECT_EQ(CoverageSegment(18, 2, false), Segments[6]);
}

TEST_P(CoverageMappingTest, dont_emit_redundant_segments) {
  ProfileWriter.addRecord({"func1", 0x1234, {1, 1}}, Err);
  startFunction("func1", 0x1234);

  addCMR(Counter::getCounter(0), "file1", 1, 1, 4, 4);
  addCMR(Counter::getCounter(1), "file1", 2, 2, 5, 5);
  addCMR(Counter::getCounter(0), "file1", 3, 3, 6, 6);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(5U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 1, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(2, 2, 1, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(3, 3, 1, true), Segments[2]);
  EXPECT_EQ(CoverageSegment(4, 4, 1, false), Segments[3]);
  // A closing segment starting at 5:5 would be redundant: it would have the
  // same count as the segment starting at 4:4, and has all the same metadata.
  EXPECT_EQ(CoverageSegment(6, 6, false), Segments[4]);
}

TEST_P(CoverageMappingTest, dont_emit_closing_segment_at_new_region_start) {
  ProfileWriter.addRecord({"func1", 0x1234, {1}}, Err);
  startFunction("func1", 0x1234);

  addCMR(Counter::getCounter(0), "file1", 1, 1, 6, 5);
  addCMR(Counter::getCounter(0), "file1", 2, 2, 6, 5);
  addCMR(Counter::getCounter(0), "file1", 3, 3, 6, 5);
  addCMR(Counter::getCounter(0), "file1", 6, 5, 7, 7);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(5U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 1, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(2, 2, 1, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(3, 3, 1, true), Segments[2]);
  EXPECT_EQ(CoverageSegment(6, 5, 1, true), Segments[3]);
  // The old segment builder would get this wrong by emitting multiple segments
  // which start at 6:5 (a few of which were skipped segments). We should just
  // get a segment for the region entry.
  EXPECT_EQ(CoverageSegment(7, 7, false), Segments[4]);
}

TEST_P(CoverageMappingTest, handle_consecutive_regions_with_zero_length) {
  ProfileWriter.addRecord({"func1", 0x1234, {1, 2}}, Err);
  startFunction("func1", 0x1234);

  addCMR(Counter::getCounter(0), "file1", 1, 1, 1, 1);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 1, 1);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 1, 1);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 1, 1);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 1, 1);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(1U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, true), Segments[0]);
  // We need to get a skipped segment starting at 1:1. In this case there is
  // also a region entry at 1:1.
}

TEST_P(CoverageMappingTest, handle_sandwiched_zero_length_region) {
  ProfileWriter.addRecord({"func1", 0x1234, {2, 1}}, Err);
  startFunction("func1", 0x1234);

  addCMR(Counter::getCounter(0), "file1", 1, 5, 4, 4);
  addCMR(Counter::getCounter(1), "file1", 1, 9, 1, 50);
  addCMR(Counter::getCounter(1), "file1", 2, 7, 2, 34);
  addCMR(Counter::getCounter(1), "file1", 3, 5, 3, 21);
  addCMR(Counter::getCounter(1), "file1", 3, 21, 3, 21);
  addCMR(Counter::getCounter(1), "file1", 4, 12, 4, 17);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(10U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 5, 2, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(1, 9, 1, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(1, 50, 2, false), Segments[2]);
  EXPECT_EQ(CoverageSegment(2, 7, 1, true), Segments[3]);
  EXPECT_EQ(CoverageSegment(2, 34, 2, false), Segments[4]);
  EXPECT_EQ(CoverageSegment(3, 5, 1, true), Segments[5]);
  EXPECT_EQ(CoverageSegment(3, 21, 2, true), Segments[6]);
  // Handle the zero-length region by creating a segment with its predecessor's
  // count (i.e the count from 1:5 -> 4:4).
  EXPECT_EQ(CoverageSegment(4, 4, false), Segments[7]);
  // The area between 4:4 and 4:12 is skipped.
  EXPECT_EQ(CoverageSegment(4, 12, 1, true), Segments[8]);
  EXPECT_EQ(CoverageSegment(4, 17, false), Segments[9]);
}

TEST_P(CoverageMappingTest, handle_last_completed_region) {
  ProfileWriter.addRecord({"func1", 0x1234, {1, 2, 3, 4}}, Err);
  startFunction("func1", 0x1234);

  addCMR(Counter::getCounter(0), "file1", 1, 1, 8, 8);
  addCMR(Counter::getCounter(1), "file1", 2, 2, 5, 5);
  addCMR(Counter::getCounter(2), "file1", 3, 3, 4, 4);
  addCMR(Counter::getCounter(3), "file1", 6, 6, 7, 7);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());
  const auto FunctionRecords = LoadedCoverage->getCoveredFunctions();
  const auto &FunctionRecord = *FunctionRecords.begin();
  CoverageData Data = LoadedCoverage->getCoverageForFunction(FunctionRecord);
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());

  ASSERT_EQ(8U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 1, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(2, 2, 2, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(3, 3, 3, true), Segments[2]);
  EXPECT_EQ(CoverageSegment(4, 4, 2, false), Segments[3]);
  EXPECT_EQ(CoverageSegment(5, 5, 1, false), Segments[4]);
  EXPECT_EQ(CoverageSegment(6, 6, 4, true), Segments[5]);
  EXPECT_EQ(CoverageSegment(7, 7, 1, false), Segments[6]);
  EXPECT_EQ(CoverageSegment(8, 8, false), Segments[7]);
}

TEST_P(CoverageMappingTest, expansion_gets_first_counter) {
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(1), "foo", 10, 1, 10, 2);
  // This starts earlier in "foo", so the expansion should get its counter.
  addCMR(Counter::getCounter(2), "foo", 1, 1, 20, 1);
  addExpansionCMR("bar", "foo", 3, 3, 3, 3);

  writeAndReadCoverageRegions();
  ASSERT_EQ(1u, OutputFunctions.size());
  OutputFunctionCoverageData &Output = OutputFunctions.back();

  ASSERT_EQ(CounterMappingRegion::ExpansionRegion, Output.Regions[2].Kind);
  ASSERT_EQ(Counter::getCounter(2), Output.Regions[2].Count);
  ASSERT_EQ(3U, Output.Regions[2].LineStart);
}

TEST_P(CoverageMappingTest, basic_coverage_iteration) {
  ProfileWriter.addRecord({"func", 0x1234, {30, 20, 10, 0}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 4, 7);
  addCMR(Counter::getCounter(2), "file1", 5, 8, 9, 1);
  addCMR(Counter::getCounter(3), "file1", 10, 10, 11, 11);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

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

TEST_P(CoverageMappingTest, test_line_coverage_iterator) {
  ProfileWriter.addRecord({"func", 0x1234, {30, 20, 10, 0}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 4, 7);
  addCMR(Counter::getCounter(2), "file1", 5, 8, 9, 1);
  addCMR(Counter::getCounter(3), "file1", 10, 10, 11, 11);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");

  unsigned Line = 0;
  unsigned LineCounts[] = {20, 20, 20, 20, 30, 10, 10, 10, 10, 0, 0};
  for (const auto &LCS : getLineCoverageStats(Data)) {
    ASSERT_EQ(Line + 1, LCS.getLine());
    errs() << "Line: " << Line + 1 << ", count = " << LCS.getExecutionCount() << "\n";
    ASSERT_EQ(LineCounts[Line], LCS.getExecutionCount());
    ++Line;
  }
  ASSERT_EQ(11U, Line);
}

TEST_P(CoverageMappingTest, uncovered_function) {
  startFunction("func", 0x1234);
  addCMR(Counter::getZero(), "file1", 1, 2, 3, 4);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(2U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 2, 0, true), Segments[0]);
  ASSERT_EQ(CoverageSegment(3, 4, false),   Segments[1]);
}

TEST_P(CoverageMappingTest, uncovered_function_with_mapping) {
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 4, 7);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(3U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 0, true),  Segments[0]);
  ASSERT_EQ(CoverageSegment(4, 7, 0, false), Segments[1]);
  ASSERT_EQ(CoverageSegment(9, 9, false),    Segments[2]);
}

TEST_P(CoverageMappingTest, combine_regions) {
  ProfileWriter.addRecord({"func", 0x1234, {10, 20, 30}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 3, 3, 4, 4);
  addCMR(Counter::getCounter(2), "file1", 3, 3, 4, 4);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(4U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 10, true), Segments[0]);
  ASSERT_EQ(CoverageSegment(3, 3, 50, true), Segments[1]);
  ASSERT_EQ(CoverageSegment(4, 4, 10, false), Segments[2]);
  ASSERT_EQ(CoverageSegment(9, 9, false), Segments[3]);
}

TEST_P(CoverageMappingTest, restore_combined_counter_after_nested_region) {
  ProfileWriter.addRecord({"func", 0x1234, {10, 20, 40}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(2), "file1", 3, 3, 5, 5);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(4U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 30, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(3, 3, 40, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(5, 5, 30, false), Segments[2]);
  EXPECT_EQ(CoverageSegment(9, 9, false), Segments[3]);
}

// If CodeRegions and ExpansionRegions cover the same area,
// only counts of CodeRegions should be used.
TEST_P(CoverageMappingTest, dont_combine_expansions) {
  ProfileWriter.addRecord({"func", 0x1234, {10, 20}}, Err);
  ProfileWriter.addRecord({"func", 0x1234, {0, 0}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(1), "file1", 3, 3, 4, 4);
  addCMR(Counter::getCounter(1), "include1", 6, 6, 7, 7);
  addExpansionCMR("file1", "include1", 3, 3, 4, 4);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file1");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(4U, Segments.size());
  ASSERT_EQ(CoverageSegment(1, 1, 10, true), Segments[0]);
  ASSERT_EQ(CoverageSegment(3, 3, 20, true), Segments[1]);
  ASSERT_EQ(CoverageSegment(4, 4, 10, false), Segments[2]);
  ASSERT_EQ(CoverageSegment(9, 9, false), Segments[3]);
}

// If an area is covered only by ExpansionRegions, they should be combinated.
TEST_P(CoverageMappingTest, combine_expansions) {
  ProfileWriter.addRecord({"func", 0x1234, {2, 3, 7}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(1), "include1", 1, 1, 1, 10);
  addCMR(Counter::getCounter(2), "include2", 1, 1, 1, 10);
  addCMR(Counter::getCounter(0), "file", 1, 1, 5, 5);
  addExpansionCMR("file", "include1", 3, 1, 3, 5);
  addExpansionCMR("file", "include2", 3, 1, 3, 5);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("file");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(4U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 2, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(3, 1, 10, true), Segments[1]);
  EXPECT_EQ(CoverageSegment(3, 5, 2, false), Segments[2]);
  EXPECT_EQ(CoverageSegment(5, 5, false), Segments[3]);
}

TEST_P(CoverageMappingTest, strip_filename_prefix) {
  ProfileWriter.addRecord({"file1:func", 0x1234, {0}}, Err);

  startFunction("file1:func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  std::vector<std::string> Names;
  for (const auto &Func : LoadedCoverage->getCoveredFunctions())
    Names.push_back(Func.Name);
  ASSERT_EQ(1U, Names.size());
  ASSERT_EQ("func", Names[0]);
}

TEST_P(CoverageMappingTest, strip_unknown_filename_prefix) {
  ProfileWriter.addRecord({"<unknown>:func", 0x1234, {0}}, Err);

  startFunction("<unknown>:func", 0x1234);
  addCMR(Counter::getCounter(0), "", 1, 1, 9, 9);
  EXPECT_THAT_ERROR(loadCoverageMapping(/*EmitFilenames=*/false), Succeeded());

  std::vector<std::string> Names;
  for (const auto &Func : LoadedCoverage->getCoveredFunctions())
    Names.push_back(Func.Name);
  ASSERT_EQ(1U, Names.size());
  ASSERT_EQ("func", Names[0]);
}

TEST_P(CoverageMappingTest, dont_detect_false_instantiations) {
  ProfileWriter.addRecord({"foo", 0x1234, {10}}, Err);
  ProfileWriter.addRecord({"bar", 0x2345, {20}}, Err);

  startFunction("foo", 0x1234);
  addCMR(Counter::getCounter(0), "expanded", 1, 1, 1, 10);
  addExpansionCMR("main", "expanded", 4, 1, 4, 5);

  startFunction("bar", 0x2345);
  addCMR(Counter::getCounter(0), "expanded", 1, 1, 1, 10);
  addExpansionCMR("main", "expanded", 9, 1, 9, 5);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  std::vector<InstantiationGroup> InstantiationGroups =
      LoadedCoverage->getInstantiationGroups("expanded");
  for (const auto &Group : InstantiationGroups)
    ASSERT_EQ(Group.size(), 1U);
}

TEST_P(CoverageMappingTest, load_coverage_for_expanded_file) {
  ProfileWriter.addRecord({"func", 0x1234, {10}}, Err);

  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "expanded", 1, 1, 1, 10);
  addExpansionCMR("main", "expanded", 4, 1, 4, 5);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  CoverageData Data = LoadedCoverage->getCoverageForFile("expanded");
  std::vector<CoverageSegment> Segments(Data.begin(), Data.end());
  ASSERT_EQ(2U, Segments.size());
  EXPECT_EQ(CoverageSegment(1, 1, 10, true), Segments[0]);
  EXPECT_EQ(CoverageSegment(1, 10, false), Segments[1]);
}

TEST_P(CoverageMappingTest, skip_duplicate_function_record) {
  ProfileWriter.addRecord({"func", 0x1234, {1}}, Err);

  // This record should be loaded.
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);

  // This record should be loaded.
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(0), "file2", 1, 1, 9, 9);

  // This record should be skipped.
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);

  // This record should be loaded.
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file2", 1, 1, 9, 9);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);

  // This record should be skipped.
  startFunction("func", 0x1234);
  addCMR(Counter::getCounter(0), "file1", 1, 1, 9, 9);
  addCMR(Counter::getCounter(0), "file2", 1, 1, 9, 9);

  EXPECT_THAT_ERROR(loadCoverageMapping(), Succeeded());

  auto Funcs = LoadedCoverage->getCoveredFunctions();
  unsigned NumFuncs = std::distance(Funcs.begin(), Funcs.end());
  ASSERT_EQ(3U, NumFuncs);
}

// FIXME: Use ::testing::Combine() when llvm updates its copy of googletest.
INSTANTIATE_TEST_CASE_P(ParameterizedCovMapTest, CoverageMappingTest,
                        ::testing::Values(std::pair<bool, bool>({false, false}),
                                          std::pair<bool, bool>({false, true}),
                                          std::pair<bool, bool>({true, false}),
                                          std::pair<bool, bool>({true, true})),);

} // end anonymous namespace
