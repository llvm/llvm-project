#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <iostream>
#include <iomanip>

static std::uint64_t NextEventID = 0;
static std::uint64_t CurrentEventID = 0;

static std::unordered_map<std::uint64_t, std::uint64_t> LastEvent;

extern "C" void __def_use_trace_inst(std::uint64_t InstID) {
  CurrentEventID = NextEventID++;

  std::fprintf(
      stderr,
      "EVENT %llu INST %llu\n",
      static_cast<unsigned long long>(CurrentEventID),
      static_cast<unsigned long long>(InstID));

  LastEvent[InstID] = CurrentEventID;
}

extern "C" void __def_use_trace_ssa_use(std::uint64_t DefID) {
  auto It = LastEvent.find(DefID);

  if (It == LastEvent.end()) {
    std::fprintf(
        stderr,
        "MISSING DEF INST %llu\n",
        static_cast<unsigned long long>(DefID));
    return;
  }

  std::fprintf(
      stderr,
      "EDGE %llu -> %llu\n",
      static_cast<unsigned long long>(It->second),
      static_cast<unsigned long long>(CurrentEventID));
}

extern "C" void __def_use_trace_store(uint64_t Address,
                                      uint64_t Size) {
  std::cerr << "STORE 0x" << std::hex <<  Address << " " <<  std::dec << Size << '\n';
}

extern "C" void __def_use_trace_load(uint64_t Address,
                                     uint64_t Size) {
  std::cerr << "LOAD 0x" << std::hex << Address << " " << std::dec <<  Size << '\n';
}
