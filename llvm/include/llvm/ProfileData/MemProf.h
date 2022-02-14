#ifndef LLVM_PROFILEDATA_MEMPROF_H_
#define LLVM_PROFILEDATA_MEMPROF_H_

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ProfileData/MemProfData.inc"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace memprof {

struct MemProfRecord {
  struct Frame {
    std::string Function;
    uint32_t LineOffset;
    uint32_t Column;
    bool IsInlineFrame;

    Frame(std::string Str, uint32_t Off, uint32_t Col, bool Inline)
        : Function(std::move(Str)), LineOffset(Off), Column(Col),
          IsInlineFrame(Inline) {}
  };

  std::vector<Frame> CallStack;
  // TODO: Replace this with the entry format described in the RFC so
  // that the InstrProfRecord reader and writer do not have to be concerned
  // about backwards compat.
  MemInfoBlock Info;

  void clear() {
    CallStack.clear();
    Info = MemInfoBlock();
  }

  // Prints out the contents of the memprof record in YAML.
  void print(llvm::raw_ostream &OS) const {
    OS << "    Callstack:\n";
    // TODO: Print out the frame on one line with to make it easier for deep
    // callstacks once we have a test to check valid YAML is generated.
    for (const auto &Frame : CallStack) {
      OS << "    -\n"
         << "      Function: " << Frame.Function << "\n"
         << "      LineOffset: " << Frame.LineOffset << "\n"
         << "      Column: " << Frame.Column << "\n"
         << "      Inline: " << Frame.IsInlineFrame << "\n";
    }

    OS << "    MemInfoBlock:\n";

    // TODO: Replace this once the format is updated to be version agnostic.
    OS << "      "
       << "AllocCount: " << Info.alloc_count << "\n";
    OS << "      "
       << "TotalAccessCount: " << Info.total_access_count << "\n";
    OS << "      "
       << "MinAccessCount: " << Info.min_access_count << "\n";
    OS << "      "
       << "MaxAccessCount: " << Info.max_access_count << "\n";
    OS << "      "
       << "TotalSize: " << Info.total_size << "\n";
    OS << "      "
       << "MinSize: " << Info.min_size << "\n";
    OS << "      "
       << "MaxSize: " << Info.max_size << "\n";
    OS << "      "
       << "AllocTimestamp: " << Info.alloc_timestamp << "\n";
    OS << "      "
       << "DeallocTimestamp: " << Info.dealloc_timestamp << "\n";
    OS << "      "
       << "TotalLifetime: " << Info.total_lifetime << "\n";
    OS << "      "
       << "MinLifetime: " << Info.min_lifetime << "\n";
    OS << "      "
       << "MaxLifetime: " << Info.max_lifetime << "\n";
    OS << "      "
       << "AllocCpuId: " << Info.alloc_cpu_id << "\n";
    OS << "      "
       << "DeallocCpuId: " << Info.dealloc_cpu_id << "\n";
    OS << "      "
       << "NumMigratedCpu: " << Info.num_migrated_cpu << "\n";
    OS << "      "
       << "NumLifetimeOverlaps: " << Info.num_lifetime_overlaps << "\n";
    OS << "      "
       << "NumSameAllocCpu: " << Info.num_same_alloc_cpu << "\n";
    OS << "      "
       << "NumSameDeallocCpu: " << Info.num_same_dealloc_cpu << "\n";
  }
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROF_H_
