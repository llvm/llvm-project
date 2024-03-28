#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Bitstream/BitCodeEnums.h"
#include "llvm/Bitstream/BitstreamWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <functional>
#include <map>
#include <system_error>
#include <utility>

using namespace llvm;

cl::opt<std::string> InputFile("raw-prof", cl::NotHidden, cl::Positional,
                               cl::desc("<Raw contextual profile>"));
cl::opt<std::string> OutputFile("output", cl::NotHidden, cl::Positional,
                                cl::desc("<Converted contextual profile>"));

enum Codes {
  Invalid,
  Guid,
  CalleeIndex,
  Counters,
};

struct ContextNode {
  uint64_t Guid = 0;
  uint64_t Next = 0;
  uint32_t NrCounters = 0;
  uint32_t NrCallsites = 0;
};

std::optional<StringRef>
getContext(uint64_t Addr, const std::map<uint64_t, StringRef> &Pages,
           std::function<std::optional<StringRef>()> Load) {
  while (true) {
    auto It = Pages.upper_bound(Addr);
    --It;
    if (It->first > Addr || Addr >= It->first + It->second.size()) {
      if (!Load())
        return std::nullopt;
      continue;
    }
    assert(It->first <= Addr);
    assert(Addr < It->first + It->second.size());
    uint64_t Offset = Addr - It->first;
    return It->second.substr(Offset);
  }
}

const ContextNode *
convertAddressToContext(uint64_t Addr,
                        const std::map<uint64_t, StringRef> &Pages,
                        std::function<std::optional<StringRef>()> Load) {
  if (Addr == 0)
    return nullptr;
  return reinterpret_cast<const ContextNode *>(
      getContext(Addr, Pages, Load).value().data());
}

void writeContext(StringRef FirstPage, BitstreamWriter &Writer,
                  std::map<uint64_t, StringRef> &Pages,
                  std::function<std::optional<StringRef>()> Load,
                  std::optional<uint32_t> Index = std::nullopt) {
  const auto *Root = reinterpret_cast<const ContextNode *>(FirstPage.data());
  for (auto *N = Root; N; N = convertAddressToContext(N->Next, Pages, Load)) {
    Writer.EnterSubblock(100, 2);
    Writer.EmitRecord(Codes::Guid, SmallVector<uint64_t, 1>{N->Guid});
    if (Index)
      p Writer.EmitRecord(Codes::CalleeIndex, SmallVector<uint32_t, 1>{*Index});
    //--- these go together to emit an array
    Writer.EmitCode(bitc::UNABBREV_RECORD);
    Writer.EmitVBR(Codes::Counters, 6);
    Writer.EmitVBR(N->NrCounters, 6);
    const uint64_t *CounterStart = reinterpret_cast<const uint64_t *>(&N[1]);
    for (auto I = 0U; I < N->NrCounters; ++I)
      Writer.EmitVBR64(CounterStart[I], 6);
    //---
    auto *CallsitesStart = reinterpret_cast<const uint64_t *>(
        reinterpret_cast<const char *>(&N[1]) +
        sizeof(uint64_t) * N->NrCounters);
    for (size_t I = 0; I < N->NrCallsites; ++I) {
      uint64_t Addr = CallsitesStart[I];
      if (!Addr)
        continue;
      if (auto S = getContext(Addr, Pages, Load))
        writeContext(*S, Writer, Pages, Load, I);
    }
    Writer.ExitBlock();
  }
}

int main(int argc, const char *argv[]) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv,
                              "LLVM Contextual Profile Converter\n");
  SmallVector<char, 1 << 20> Buff;
  std::error_code EC;
  raw_fd_stream Out(OutputFile, EC);
  if (EC) {
    errs() << "Could not open output file: " << EC.message() << "\n";
    return 1;
  }
  auto Input = MemoryBuffer::getFileOrSTDIN(InputFile);
  if (!Input)
    return 1;
  auto In = (*Input).get()->getBuffer();
  BitstreamWriter Writer(Buff, &Out, 0);
  std::map<uint64_t, StringRef> Pages;
  auto Load = [&]() -> std::optional<StringRef> {
    if (In.size() < 2 * sizeof(uint64_t))
      return std::nullopt;
    auto *Data = reinterpret_cast<const uint64_t *>(In.data());
    uint64_t Start = Data[0];
    uint64_t Len = Data[1];
    In = In.substr(2 * sizeof(uint64_t));
    auto It = Pages.insert({Start, In.substr(0, Len)});
    In = In.substr(Len);
    return It.first->second;
  };
  uint32_t Magic = 0xfafababa;
  Out.write(reinterpret_cast<const char *>(&Magic), sizeof(uint32_t));
  while (auto S = Load()) {
    writeContext(*S, Writer, Pages, Load);
  }
  Out.flush();
  Out.close();
  return 0;
}