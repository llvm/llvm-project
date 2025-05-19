#include "DataAccessPerfReader.h"
#include "ErrorHandling.h"
#include "PerfReader.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <regex>

static llvm::Regex IPSampleRegex(": 0x[a-fA-F0-9]+ period:");
static llvm::Regex DataAddressRegex("addr: 0x[a-fA-F0-9]+");

namespace llvm {

void DataAccessPerfReader::parsePerfTraces() {
  parsePerfTrace(PerfTraceFilename);
}

static void testPerfSampleRecordRegex() {
  std::regex logRegex(
      R"(^.*?PERF_RECORD_SAMPLE\(.*?\):\s*(\d+)\/(\d+):\s*(0x[0-9a-fA-F]+)\s+period:\s*\d+\s+addr:\s*(0x[0-9a-fA-F]+)$)");

  std::smatch testMatch;
  const std::string testLine =
      "2193330181938979 0xa88 [0x48]: PERF_RECORD_SAMPLE(IP, 0x4002): "
      "1807344/1807344: 0x260b45 period: 100 addr: 0x200630";
  if (std::regex_search(testLine, testMatch, logRegex)) {
    if (testMatch.size() != 5) {
      exitWithError("Regex did not match expected number of groups.");
    }
    for (size_t i = 0; i < testMatch.size(); ++i) {
      errs() << "Group " << i << ": " << testMatch[i] << "\n";
    }
    // errs() << "Test line matched successfully.\n";
  } else {
    exitWithError("Test line did not match regex.");
  }
}

// Ignore mmap events.
void DataAccessPerfReader::parsePerfTrace(StringRef PerfTrace) {
  std::regex logRegex(
      R"(^.*?PERF_RECORD_SAMPLE\(.*?\):\s*(\d+)\/(\d+):\s*(0x[0-9a-fA-F]+)\s+period:\s*\d+\s+addr:\s*(0x[0-9a-fA-F]+)$)");
  uint64_t UnmatchedLine = 0, MatchedLine = 0;

  auto BufferOrErr = MemoryBuffer::getFile(PerfTrace);
  std::error_code EC = BufferOrErr.getError();
  if (EC)
    exitWithError("Failed to open perf trace file: " + PerfTrace);

  line_iterator LineIt(*BufferOrErr.get(), true);
  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef Line = *LineIt;

    // Parse MMAP event from perf trace.
    // Construct a binary from the binary file path.
    PerfScriptReader::MMapEvent MMap;
    if (Line.contains("PERF_RECORD_MMAP2")) {
      if (PerfScriptReader::extractMMapEventForBinary(Binary, Line, MMap)) {
        errs() << "MMap event found: "
               << "PID: " << MMap.PID
               << ", Address: " << format("0x%llx", MMap.Address)
               << ", Size: " << MMap.Size << ", Offset: " << MMap.Offset
               << ", Binary Path: " << MMap.BinaryPath << "\n";
        if (MMap.Offset == 0) {
          updateBinaryAddress(MMap);
        }
      }
      continue;
    }

    if (!Line.contains("PERF_RECORD_SAMPLE")) {
      // Skip lines that do not contain "PERF_RECORD_SAMPLE".
      continue;
    }
    // errs() << "Processing line: " << Line << "\n";

    // if (IPSampleRegex.match(Line, &Matches)) {
    //   errs() << "IP Captured: " << Matches.size() << "\n";
    // }
    // if (DataAddressRegex.match(Line, &Matches)) {
    //   errs() << "Data Address Captured: " << Matches.size() << "\n";
    // }

    std::smatch matches;
    const std::string LineStr = Line.str();

    if (std::regex_search(LineStr.begin(), LineStr.end(), matches, logRegex)) {
      if (matches.size() != 5)
        continue;

      uint64_t DataAddress = std::stoull(matches[4].str(), nullptr, 16);
      uint64_t IP = std::stoull(matches[3].str(), nullptr, 16);
      int32_t PID = std::stoi(matches[1].str());
      // if (DataAddress == 0x200630) {
      //   errs() << "Find data address at 0x200630, IP: " << format("0x%llx",
      //   IP)
      //          << " pid is " << PID << "\n";
      // }

      // errs() << matches.size() << " matches found in line: " << LineStr <<
      // "\n"; for (const auto &Match : matches) {
      //   errs() << "Match: " << Match.str() << "\n";
      // }
      //  Check if the PID matches the filter.

      if (PIDFilter && *PIDFilter != PID) {
        continue;
      }

      // Extract the address and count.

      uint64_t CanonicalDataAddress =
          Binary->canonicalizeVirtualAddress(DataAddress);
      // errs() << "Data address is " << format("0x" PRIx64 ":", DataAddress)
      //        << " Canonical data address is "
      //        << format("0x" PRIx64 ":", CanonicalDataAddress) << "\n";
      AddressToCount[CanonicalDataAddress] += 1;
      MatchedLine++;
    } else {
      // errs() << "\tNo match found for line: " << Line << "\n";
      UnmatchedLine++;
    }
  }

  errs() << "Total unmatched lines: " << UnmatchedLine << "\t"
         << "Matched lines: " << MatchedLine << "\n";
}

}  // namespace llvm
