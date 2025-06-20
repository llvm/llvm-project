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

// Ignore mmap events.
void DataAccessPerfReader::parsePerfTrace(StringRef PerfTrace) {
  std::regex logRegex(
      R"(^.*?PERF_RECORD_SAMPLE\(.*?\):\s*(\d+)\/(\d+):\s*(0x[0-9a-fA-F]+)\s+period:\s*\d+\s+addr:\s*(0x[0-9a-fA-F]+)$)");

  auto BufferOrErr = MemoryBuffer::getFile(PerfTrace);
  std::error_code EC = BufferOrErr.getError();
  if (EC)
    exitWithError("Failed to open perf trace file: " + PerfTrace);

  line_iterator LineIt(*BufferOrErr.get(), true);
  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef Line = *LineIt;

    // Parse MMAP event from perf trace.
    // Parse MMAP event from perf trace.
    // Construct a binary from the binary file path.
    PerfScriptReader::MMapEvent MMap;
    if (Line.contains("PERF_RECORD_MMAP2")) {
      if (PerfScriptReader::extractMMapEventForBinary(Binary, Line, MMap)) {
        // TODO: This is a hack to avoid mapping binary address for data section
        // mappings.
        if (MMap.Offset == 0) {
          updateBinaryAddress(MMap);
          errs() << "Binary base address is "
                 << format("0x%" PRIx64, Binary->getBaseAddress())
                 << " and preferred base address is "
                 << format("0x%" PRIx64, Binary->getPreferredBaseAddress())
                 << " and first loadable address is "
                 << format("0x%" PRIx64, Binary->getFirstLoadableAddress())
                 << "\n";
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
      if (DataAddress < DataMMap.Address ||
          DataAddress >= DataMMap.Address + DataMMap.Size)
        continue;

      int32_t PID = std::stoi(matches[1].str());
      //  Check if the PID matches the filter.

      if (PIDFilter && *PIDFilter != PID) {
        continue;
      }

      uint64_t IP = std::stoull(matches[3].str(), nullptr, 16);
      // Extract the address and count.
      uint64_t CanonicalDataAddress =
          canonicalizeDataAddress(DataAddress, *Binary, DataMMap, DataSegment);

      uint64_t CanonicalIPAddress = Binary->canonicalizeVirtualAddress(IP);

      AddressMap[CanonicalIPAddress][CanonicalDataAddress] += 1;
    }
  }
}

}  // namespace llvm
