//===-- HexagonTargetMachine.cpp - Define TargetMachine for Hexagon -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Hexagon target spec.
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "Hexagon.h"
#include "HexagonISelLowering.h"
#include "HexagonMachineScheduler.h"
#include "HexagonTargetObjectFile.h"
#include "HexagonTargetTransformInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;

static cl:: opt<bool> DisableHardwareLoops("disable-hexagon-hwloops",
  cl::Hidden, cl::desc("Disable Hardware Loops for Hexagon target"));

static cl::opt<bool> DisableHexagonCFGOpt("disable-hexagon-cfgopt",
  cl::Hidden, cl::ZeroOrMore, cl::init(false),
  cl::desc("Disable Hexagon CFG Optimization"));

static cl::opt<bool> DisableStoreWidening("disable-store-widen",
  cl::Hidden, cl::init(false), cl::desc("Disable store widening"));

static cl::opt<bool> EnableExpandCondsets("hexagon-expand-condsets",
  cl::init(true), cl::Hidden, cl::ZeroOrMore,
  cl::desc("Early expansion of MUX"));

static cl::opt<bool> EnableEarlyIf("hexagon-eif", cl::init(true), cl::Hidden,
  cl::ZeroOrMore, cl::desc("Enable early if-conversion"));

static cl::opt<bool> EnableGenInsert("hexagon-insert", cl::init(true),
  cl::Hidden, cl::desc("Generate \"insert\" instructions"));

static cl::opt<bool> EnableCommGEP("hexagon-commgep", cl::init(true),
  cl::Hidden, cl::ZeroOrMore, cl::desc("Enable commoning of GEP instructions"));

static cl::opt<bool> EnableGenExtract("hexagon-extract", cl::init(true),
  cl::Hidden, cl::desc("Generate \"extract\" instructions"));

static cl::opt<bool> EnableGenMux("hexagon-mux", cl::init(true), cl::Hidden,
  cl::desc("Enable converting conditional transfers into MUX instructions"));

static cl::opt<bool> EnableGenPred("hexagon-gen-pred", cl::init(true),
  cl::Hidden, cl::desc("Enable conversion of arithmetic operations to "
  "predicate instructions"));

static cl::opt<bool> DisableHSDR("disable-hsdr", cl::init(false), cl::Hidden,
  cl::desc("Disable splitting double registers"));

static cl::opt<bool> EnableBitSimplify("hexagon-bit", cl::init(true),
  cl::Hidden, cl::desc("Bit simplification"));

static cl::opt<bool> EnableLoopResched("hexagon-loop-resched", cl::init(true),
  cl::Hidden, cl::desc("Loop rescheduling"));

/// HexagonTargetMachineModule - Note that this is used on hosts that
/// cannot link in a library unless there are references into the
/// library.  In particular, it seems that it is not possible to get
/// things to work on Win32 without this.  Though it is unused, do not
/// remove it.
extern "C" int HexagonTargetMachineModule;
int HexagonTargetMachineModule = 0;

extern "C" void LLVMInitializeHexagonTarget() {
  // Register the target.
  RegisterTargetMachine<HexagonTargetMachine> X(TheHexagonTarget);
}

static ScheduleDAGInstrs *createVLIWMachineSched(MachineSchedContext *C) {
  return new VLIWMachineScheduler(C, make_unique<ConvergingVLIWScheduler>());
}

static MachineSchedRegistry
SchedCustomRegistry("hexagon", "Run Hexagon's custom scheduler",
                    createVLIWMachineSched);

namespace llvm {
  FunctionPass *createHexagonBitSimplify();
  FunctionPass *createHexagonCallFrameInformation();
  FunctionPass *createHexagonCFGOptimizer();
  FunctionPass *createHexagonCommonGEP();
  FunctionPass *createHexagonCopyToCombine();
  FunctionPass *createHexagonEarlyIfConversion();
  FunctionPass *createHexagonExpandCondsets();
  FunctionPass *createHexagonExpandPredSpillCode();
  FunctionPass *createHexagonFixupHwLoops();
  FunctionPass *createHexagonGenExtract();
  FunctionPass *createHexagonGenInsert();
  FunctionPass *createHexagonGenMux();
  FunctionPass *createHexagonGenPredicate();
  FunctionPass *createHexagonHardwareLoops();
  FunctionPass *createHexagonISelDag(HexagonTargetMachine &TM,
                                     CodeGenOpt::Level OptLevel);
  FunctionPass *createHexagonLoopRescheduling();
  FunctionPass *createHexagonNewValueJump();
  FunctionPass *createHexagonOptimizeSZextends();
  FunctionPass *createHexagonPacketizer();
  FunctionPass *createHexagonPeephole();
  FunctionPass *createHexagonSplitConst32AndConst64();
  FunctionPass *createHexagonSplitDoubleRegs();
  FunctionPass *createHexagonStoreWidening();
} // end namespace llvm;

/// HexagonTargetMachine ctor - Create an ILP32 architecture model.
///

/// Hexagon_TODO: Do I need an aggregate alignment?
///
HexagonTargetMachine::HexagonTargetMachine(const Target &T, const Triple &TT,
                                           StringRef CPU, StringRef FS,
                                           const TargetOptions &Options,
                                           Reloc::Model RM, CodeModel::Model CM,
                                           CodeGenOpt::Level OL)
    : LLVMTargetMachine(T, "e-m:e-p:32:32-i1:32-i64:64-a:0-n32", TT, CPU, FS,
                        Options, RM, CM, OL),
      TLOF(make_unique<HexagonTargetObjectFile>()) {
  initAsmInfo();
}

const HexagonSubtarget *
HexagonTargetMachine::getSubtargetImpl(const Function &F) const {
  AttributeSet FnAttrs = F.getAttributes();
  Attribute CPUAttr =
      FnAttrs.getAttribute(AttributeSet::FunctionIndex, "target-cpu");
  Attribute FSAttr =
      FnAttrs.getAttribute(AttributeSet::FunctionIndex, "target-features");

  std::string CPU = !CPUAttr.hasAttribute(Attribute::None)
                        ? CPUAttr.getValueAsString().str()
                        : TargetCPU;
  std::string FS = !FSAttr.hasAttribute(Attribute::None)
                       ? FSAttr.getValueAsString().str()
                       : TargetFS;

  auto &I = SubtargetMap[CPU + FS];
  if (!I) {
    // This needs to be done before we create a new subtarget since any
    // creation will depend on the TM and the code generation flags on the
    // function that reside in TargetOptions.
    resetTargetOptions(F);
    I = llvm::make_unique<HexagonSubtarget>(TargetTriple, CPU, FS, *this);
  }
  return I.get();
}

TargetIRAnalysis HexagonTargetMachine::getTargetIRAnalysis() {
  return TargetIRAnalysis([this](const Function &F) {
    return TargetTransformInfo(HexagonTTIImpl(this, F));
  });
}


HexagonTargetMachine::~HexagonTargetMachine() {}

namespace {
/// Hexagon Code Generator Pass Configuration Options.
class HexagonPassConfig : public TargetPassConfig {
public:
  HexagonPassConfig(HexagonTargetMachine *TM, PassManagerBase &PM)
    : TargetPassConfig(TM, PM) {
    bool NoOpt = (TM->getOptLevel() == CodeGenOpt::None);
    if (!NoOpt) {
      if (EnableExpandCondsets) {
        Pass *Exp = createHexagonExpandCondsets();
        insertPass(&RegisterCoalescerID, IdentifyingPassPtr(Exp));
      }
    }
  }

  HexagonTargetMachine &getHexagonTargetMachine() const {
    return getTM<HexagonTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    return createVLIWMachineSched(C);
  }

  void addIRPasses() override;
  bool addInstSelector() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addPreSched2() override;
  void addPreEmitPass() override;
};
} // namespace

TargetPassConfig *HexagonTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new HexagonPassConfig(this, PM);
}

void HexagonPassConfig::addIRPasses() {
  TargetPassConfig::addIRPasses();
  bool NoOpt = (getOptLevel() == CodeGenOpt::None);

  addPass(createAtomicExpandPass(TM));
  if (!NoOpt) {
    if (EnableCommGEP)
      addPass(createHexagonCommonGEP());
    // Replace certain combinations of shifts and ands with extracts.
    if (EnableGenExtract)
      addPass(createHexagonGenExtract());
  }
}

bool HexagonPassConfig::addInstSelector() {
  HexagonTargetMachine &TM = getHexagonTargetMachine();
  bool NoOpt = (getOptLevel() == CodeGenOpt::None);

  if (!NoOpt)
    addPass(createHexagonOptimizeSZextends());

  addPass(createHexagonISelDag(TM, getOptLevel()));

  if (!NoOpt) {
    // Create logical operations on predicate registers.
    if (EnableGenPred)
      addPass(createHexagonGenPredicate(), false);
    // Rotate loops to expose bit-simplification opportunities.
    if (EnableLoopResched)
      addPass(createHexagonLoopRescheduling(), false);
    // Split double registers.
    if (!DisableHSDR)
      addPass(createHexagonSplitDoubleRegs());
    // Bit simplification.
    if (EnableBitSimplify)
      addPass(createHexagonBitSimplify(), false);
    addPass(createHexagonPeephole());
    printAndVerify("After hexagon peephole pass");
    if (EnableGenInsert)
      addPass(createHexagonGenInsert(), false);
    if (EnableEarlyIf)
      addPass(createHexagonEarlyIfConversion(), false);
  }

  return false;
}

void HexagonPassConfig::addPreRegAlloc() {
  if (getOptLevel() != CodeGenOpt::None) {
    if (!DisableStoreWidening)
      addPass(createHexagonStoreWidening(), false);
    if (!DisableHardwareLoops)
      addPass(createHexagonHardwareLoops(), false);
  }
}

void HexagonPassConfig::addPostRegAlloc() {
  if (getOptLevel() != CodeGenOpt::None)
    if (!DisableHexagonCFGOpt)
      addPass(createHexagonCFGOptimizer(), false);
}

void HexagonPassConfig::addPreSched2() {
  addPass(createHexagonCopyToCombine(), false);
  if (getOptLevel() != CodeGenOpt::None)
    addPass(&IfConverterID, false);
  addPass(createHexagonSplitConst32AndConst64());
}

void HexagonPassConfig::addPreEmitPass() {
  bool NoOpt = (getOptLevel() == CodeGenOpt::None);

  if (!NoOpt)
    addPass(createHexagonNewValueJump(), false);

  // Expand Spill code for predicate registers.
  addPass(createHexagonExpandPredSpillCode(), false);

  // Create Packets.
  if (!NoOpt) {
    if (!DisableHardwareLoops)
      addPass(createHexagonFixupHwLoops(), false);
    // Generate MUX from pairs of conditional transfers.
    if (EnableGenMux)
      addPass(createHexagonGenMux(), false);

    addPass(createHexagonPacketizer(), false);
  }

  // Add CFI instructions if necessary.
  addPass(createHexagonCallFrameInformation(), false);
}
