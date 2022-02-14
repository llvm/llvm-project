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
       << "AllocCount: " << Info.AllocCount << "\n";
    OS << "      "
       << "TotalAccessCount: " << Info.TotalAccessCount << "\n";
    OS << "      "
       << "MinAccessCount: " << Info.MinAccessCount << "\n";
    OS << "      "
       << "MaxAccessCount: " << Info.MaxAccessCount << "\n";
    OS << "      "
       << "TotalSize: " << Info.TotalSize << "\n";
    OS << "      "
       << "MinSize: " << Info.MinSize << "\n";
    OS << "      "
       << "MaxSize: " << Info.MaxSize << "\n";
    OS << "      "
       << "AllocTimestamp: " << Info.AllocTimestamp << "\n";
    OS << "      "
       << "DeallocTimestamp: " << Info.DeallocTimestamp << "\n";
    OS << "      "
       << "TotalLifetime: " << Info.TotalLifetime << "\n";
    OS << "      "
       << "MinLifetime: " << Info.MinLifetime << "\n";
    OS << "      "
       << "MaxLifetime: " << Info.MaxLifetime << "\n";
    OS << "      "
       << "AllocCpuId: " << Info.AllocCpuId << "\n";
    OS << "      "
       << "DeallocCpuId: " << Info.DeallocCpuId << "\n";
    OS << "      "
       << "NumMigratedCpu: " << Info.NumMigratedCpu << "\n";
    OS << "      "
       << "NumLifetimeOverlaps: " << Info.NumLifetimeOverlaps << "\n";
    OS << "      "
       << "NumSameAllocCpu: " << Info.NumSameAllocCpu << "\n";
    OS << "      "
       << "NumSameDeallocCpu: " << Info.NumSameDeallocCpu << "\n";
  }
};

} // namespace memprof
} // namespace llvm

#endif // LLVM_PROFILEDATA_MEMPROF_H_
