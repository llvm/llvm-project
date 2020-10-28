#ifndef LLVM_EXAMPLES_THINLTOJIT_DISCOVERYLAYER_H
#define LLVM_EXAMPLES_THINLTOJIT_DISCOVERYLAYER_H

#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include "ThinLtoJIT.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace llvm {
namespace orc {

class ThinLtoInstrumentationLayer : public IRLayer {
public:
  ThinLtoInstrumentationLayer(ExecutionSession &ES, IRCompileLayer &BaseLayer,
                              ThinLtoJIT::ExplicitMemoryBarrier MemFence,
                              unsigned FlagsPerBucket)
      : IRLayer(ES, BaseLayer.getManglingOptions()), BaseLayer(BaseLayer),
        MemFence(MemFence) {
    // TODO: So far we only allocate one bucket.
    allocateDiscoveryFlags(FlagsPerBucket);
  }

  ~ThinLtoInstrumentationLayer() override;

  void emit(std::unique_ptr<MaterializationResponsibility> R,
            ThreadSafeModule TSM) override;

  unsigned reserveDiscoveryFlags(unsigned Count);
  void registerDiscoveryFlagOwners(std::vector<GlobalValue::GUID> Guids,
                                   unsigned FirstIdx);

  void nudgeIntoDiscovery(std::vector<GlobalValue::GUID> Functions);

  std::vector<unsigned> takeFlagsThatFired();
  std::vector<GlobalValue::GUID> takeFlagOwners(std::vector<unsigned> Indexes);

  void dump(raw_ostream &OS);

private:
  IRCompileLayer &BaseLayer;
  ThinLtoJIT::ExplicitMemoryBarrier MemFence;

  enum Flag : uint8_t { Clear = 0, Fired = 1 };

  // Lock-free read access.
  uint8_t *FlagsStorage;
  Flag *FlagsIncoming; // lock-free write by design
  Flag *FlagsHandled;
  unsigned NumFlagsAllocated;
  std::atomic<unsigned> NumFlagsUsed; // spin-lock

  // Acquire/release sync between writers and reader
  std::atomic<uint64_t> FlagsSync;

  // STL container requires locking for both, read and write access.
  mutable std::mutex DiscoveryFlagsInfoLock;
  std::map<unsigned, GlobalValue::GUID> FlagOwnersMap;

  void allocateDiscoveryFlags(unsigned MinFlags);
  void compileFunctionReachedFlagSetter(IRBuilder<> &B, Flag *F);
};

} // namespace orc
} // namespace llvm

#endif
