#include "DataAccessPerfReader.h"
#include "ErrorHandling.h"
#include "PerfReader.h"
#include "ProfiledBinary.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

#include <regex>

static llvm::Regex IPSampleRegex(": 0x[a-fA-F0-9]+ period:");
static llvm::Regex DataAddressRegex("addr: 0x[a-fA-F0-9]+");

namespace llvm {

void DataAccessPerfReader::parsePerfTraces() {
  parsePerfTrace(PerfTraceFilename);
}

// Ignore mmap events.
void DataAccessPerfReader::parsePerfTrace(StringRef PerfTrace) {
  std::regex logRegex(
      R"(^.*?PERF_RECORD_SAMPLE\(.*?\):\s*(\d+)\/(\d+):\s*(0x[0-9a-fA-F]+)\s+period:\s*\d+\s+addr:\s*(0x[0-9a-fA-F]+)$)");

  std::set<std::pair<uint64_t, uint64_t>> Dedup;
  auto BufferOrErr = MemoryBuffer::getFile(PerfTrace);
  std::error_code EC = BufferOrErr.getError();
  if (EC)
    exitWithError("Failed to open perf trace file: " + PerfTrace);

  std::map<std::pair<uint64_t, uint64_t>, uint64_t> PieAddressMap;
  line_iterator LineIt(*BufferOrErr.get(), true);
  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef Line = *LineIt;

    // Parse MMAP event from perf trace.
    // Parse MMAP event from perf trace.
    // Construct a binary from the binary file path.
    MMapEvent MMap;
    if (Line.contains("PERF_RECORD_MMAP2")) {
      if (PerfScriptReader::extractMMapEventForBinary(Binary, Line, MMap)) {
        if (!MMap.MemProtectionFlag.contains("x")) {
          outs() << "PerfReader.cpp:469\tMMap: " << MMap.BinaryPath
                 << " loaded at " << format("0x%" PRIx64, MMap.Address)
                 << " with size " << format("0x%" PRIx64, MMap.Size)
                 << " and offset " << format("0x%" PRIx64, MMap.Offset) << "\t"
                 << "Protection: " << MMap.MemProtectionFlag << "\n";
          Binary->addMMapNonTextEvent(MMap);
        }
      }
      continue;
    }

    if (!Line.contains("PERF_RECORD_SAMPLE")) {
      // Skip lines that do not contain "PERF_RECORD_SAMPLE".
      continue;
    }

    std::smatch matches;
    const std::string LineStr = Line.str();

    if (std::regex_search(LineStr.begin(), LineStr.end(), matches, logRegex)) {
      if (matches.size() != 5)
        continue;

      uint64_t DataAddress = std::stoull(matches[4].str(), nullptr, 16);

      // Skip addresses out of the specified PT_LOAD section for data.
      if (!Binary->InRange(DataAddress))
        continue;

      int32_t PID = std::stoi(matches[1].str());
      if (PIDFilter && *PIDFilter != PID) {
        continue;
      }

      uint64_t IP = std::stoull(matches[3].str(), nullptr, 16);

      uint64_t DataAddressCanonicalized =
          Binary->CanonicalizeNonTextAddress(DataAddress);
      auto [Iter, Inserted] = Dedup.insert({IP, DataAddress});
      if (Inserted) {
        errs() << "Canonicalized data address: "
               << format("0x%llx", DataAddress) << " to "
               << format("0x%llx", DataAddressCanonicalized) << "\n";
      }
      PieAddressMap[{IP, DataAddress}] += 1;
      StringRef DataSymbol =
          Binary->symbolizeDataAddress(DataAddressCanonicalized);
      if (!DataSymbol.starts_with("_ZTV")) {
        // Skip non-vtable data addresses.
        continue;
      }

      AddressMap[IP][DataSymbol] += 1;
    }
  }

  for (const auto &[IPDataPair, Count] : PieAddressMap) {
    errs() << "IP " << format("0x%llx", IPDataPair.first)
           << " accesses data address " << format("0x%llx", IPDataPair.second)
           << " with count: " << Count << "\n";
  }
}

}  // namespace llvm
