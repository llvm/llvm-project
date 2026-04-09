//===-- LX32TargetMachine.cpp - LX32 TargetMachine Implementation --------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file implements LX32TargetMachine, the entry point for the LX32 LLVM
// backend.
//
// It is organized into the following sections:
//
//   Section 0 — Data layout and relocation policy
//   Section 1 — Pass pipeline configuration (LX32PassConfig)
//   Section 2 — LX32TargetMachine methods
//   Section 3 — Target registration entry points
//
//===----------------------------------------------------------------------===//

#include "LX32TargetMachine.h"
#include "LX32ISelDAGToDAG.h"

#include "../TargetInfo/LX32TargetInfo.h"

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

// Forward declaration — see Section 3.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLX32TargetMC();
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeLX32AsmPrinter();

//===----------------------------------------------------------------------===//
// Section 0 — Data layout and relocation policy
//
// The DataLayout string is the authoritative description of LX32's memory
// model.  The LLVM optimizer and code generator query it to determine type
// sizes, alignments, and pointer widths.  Getting it wrong silently produces
// incorrect code, so every field is documented below:
//
//   e        — little-endian: the least significant byte has the lowest address.
//              LX32 is little-endian by design.
//
//   m:e      — ELF symbol mangling: LLVM uses ELF naming rules for symbols.
//              Required for the GNU linker and objdump to work correctly.
//
//   p:32:32  — pointers are 32 bits wide, ABI-aligned to 32-bit (4-byte)
//              boundaries.  This matches the ILP32 ABI: sizeof(void*) == 4.
//
//   i64:64   — 64-bit integers are aligned to 64-bit (8-byte) boundaries.
//              This is critical for the ILP32 long-long convention: a 64-bit
//              value passed in a register pair (a0+a1 or a1+a2) requires the
//              low-word register to be even-indexed, which the calling
//              convention enforces only when i64 has 8-byte alignment.
//
//   n32      — the native integer width is 32 bits.  The optimizer uses this
//              to decide whether widening or narrowing operations are free.
//              On LX32, 32-bit arithmetic is single-instruction; 64-bit
//              arithmetic requires two-instruction sequences.
//
//   S32      — the preferred stack alignment is 32 bits (4 bytes).  Call sites
//              use a stricter 16-byte alignment (enforced in FrameLowering) to
//              keep sp aligned for any ABI-compliant callee.
//
//===----------------------------------------------------------------------===//

static constexpr const char *LX32DataLayout = "e-m:e-p:32:32-i64:64-n32-S32";

// getEffectiveRelocModel — select the relocation model for LX32.
//
// LX32 v1 supports only static relocation.  Position-independent code (PIC)
// requires a Global Offset Table (GOT) and Procedure Linkage Table (PLT),
// neither of which is implemented yet.  If the user explicitly requests PIC
// (-fPIC), the backend falls back to static and the linker may warn.
//
// When PIC support is added:
//   return RM.value_or(Reloc::PIC_);
static Reloc::Model
getEffectiveRelocModel(std::optional<Reloc::Model> RM) {
  return RM.value_or(Reloc::Static);
}

//===----------------------------------------------------------------------===//
// Section 1 — Pass pipeline configuration
//
// LX32PassConfig installs the LX32-specific code generation passes in the
// order the LLVM backend framework expects.
//
// The minimum viable pipeline for text-assembly output (-filetype=asm):
//   addInstSelector() — instruction selection (added Day 10)
//   (all other passes are provided by TargetPassConfig defaults)
//
// For object-file output (-filetype=obj), additional passes are required:
//   AsmPrinter  — Day 12
//   MCCodeEmitter — Day 11
//
// The current implementation returns false from addInstSelector(), which
// tells the framework that no custom selector was added.  llc will report
// "unable to select instruction" when it reaches instruction selection.
// This is the expected failure mode for the skeleton backend — a diagnostic,
// not a crash.
//===----------------------------------------------------------------------===//

namespace {

class LX32PassConfig final : public TargetPassConfig {
public:
  LX32PassConfig(LX32TargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  LX32TargetMachine &getLX32TargetMachine() const {
    return getTM<LX32TargetMachine>();
  }

  // addInstSelector — register the LX32 DAG selector.
  bool addInstSelector() override {
    addPass(createLX32ISelDag(getLX32TargetMachine(), getOptLevel()));
    return false;
  }
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Section 2 — LX32TargetMachine methods
//===----------------------------------------------------------------------===//

namespace llvm {

LX32TargetMachine::LX32TargetMachine(
    const Target &T, const Triple &TT, StringRef CPU, StringRef FS,
    const TargetOptions &Options, std::optional<Reloc::Model> RM,
    std::optional<CodeModel::Model> CM, CodeGenOptLevel OL, bool JIT)
    : CodeGenTargetMachineImpl(T, LX32DataLayout, TT, CPU, FS, Options,
                               getEffectiveRelocModel(RM),
                               // CodeModel::Small: all code and data fit in
                               // a single 32-bit address range; no medium or
                               // large model is needed for LX32 v1.
                               getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(std::make_unique<TargetLoweringObjectFileELF>())
{
  (void)JIT; // JIT compilation is not supported in LX32 v1.

  // initAsmInfo — wire up MCAsmInfo, MCInstrInfo, MCRegisterInfo, and
  // MCSubtargetInfo for this target machine.  Must be called before
  // createPassConfig; otherwise TargetPassConfig will assert when it tries
  // to access the MC descriptors.
  initAsmInfo();
}

LX32TargetMachine::~LX32TargetMachine() = default;

// getSubtargetImpl — return the subtarget for the given function.
//
// The function's target-cpu and target-features attributes override the
// module-level CPU and feature strings.  This is how per-function
// __attribute__((target("..."))) is implemented in LLVM.
//
// The SubtargetMap cache prevents reconstructing the subtarget tables on
// every call.  In the common case (no per-function attributes) every call
// uses the same key and the map lookup returns the already-constructed entry.
const LX32Subtarget *
LX32TargetMachine::getSubtargetImpl(const Function &F) const {
  Attribute CPUAttr = F.getFnAttribute("target-cpu");
  Attribute FSAttr  = F.getFnAttribute("target-features");

  // Use the function attribute if present; fall back to the module-level value.
  std::string CPU =
      CPUAttr.isValid() ? CPUAttr.getValueAsString().str() : TargetCPU;
  std::string FS =
      FSAttr.isValid()  ? FSAttr.getValueAsString().str()  : TargetFS;

  // Normalise the CPU name here as a second line of defence.  The subtarget
  // constructor also normalises, but doing it here avoids constructing a
  // "generic" subtarget and then immediately discarding it from the cache.
  if (CPU.empty())
    CPU = "generic";

  std::string Key = CPU + FS;
  auto &Entry = SubtargetMap[Key];
  if (!Entry)
    Entry = std::make_unique<LX32Subtarget>(TargetTriple, CPU,
                                             /*TuneCPU=*/CPU, FS, *this);
  return Entry.get();
}

TargetPassConfig *
LX32TargetMachine::createPassConfig(PassManagerBase &PM) {
  return new LX32PassConfig(*this, PM);
}

//===----------------------------------------------------------------------===//
// Section 3 — Target registration entry points
//
// LLVM discovers targets through a pair of extern "C" init functions whose
// names are constructed as LLVMInitialize<Target><Layer>():
//
//   LLVMInitializeLX32TargetInfo  — registers the target name and triple
//                                    (defined in target/TargetInfo/)
//   LLVMInitializeLX32TargetMC    — registers MC-layer factories: MCAsmInfo,
//                                    MCInstrInfo, MCRegisterInfo, MCSubtargetInfo,
//                                    MCInstPrinter (defined in mc/)
//   LLVMInitializeLX32Target      — registers LX32TargetMachine (this function)
//
// The LLVM build system generates calls to these functions in the tool
// initialisation path.  The order matters: TargetInfo must be initialised
// before TargetMC, and TargetMC before Target.  We enforce this by calling
// LLVMInitializeLX32TargetMC explicitly from LLVMInitializeLX32Target.
//===----------------------------------------------------------------------===//

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeLX32Target() {
  // Ensure the MC-layer registrations are in place before TargetMachine
  // construction.  initAsmInfo() (called from our constructor) will assert
  // if MCAsmInfo is not yet registered.
  LLVMInitializeLX32TargetMC();
  LLVMInitializeLX32AsmPrinter();

  // Register LX32TargetMachine as the handler for the "lx32" architecture.
  // After this call, `llc -march=lx32` constructs an LX32TargetMachine.
  RegisterTargetMachine<LX32TargetMachine> X(getTheLX32TargetInfo());
}

} // namespace llvm
