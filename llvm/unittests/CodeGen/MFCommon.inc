// Add a few Bogus backend classes so we can create MachineInstrs without
// depending on a real target.
class BogusTargetLowering : public TargetLowering {
public:
  BogusTargetLowering(TargetMachine &TM) : TargetLowering(TM) {}
};

class BogusFrameLowering : public TargetFrameLowering {
public:
  BogusFrameLowering()
      : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(4), 4) {}

  void emitPrologue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  void emitEpilogue(MachineFunction &MF,
                    MachineBasicBlock &MBB) const override {}
  bool hasFP(const MachineFunction &MF) const override { return false; }
};

static TargetRegisterClass *const BogusRegisterClasses[] = {nullptr};

class BogusRegisterInfo : public TargetRegisterInfo {
public:
  BogusRegisterInfo()
      : TargetRegisterInfo(nullptr, BogusRegisterClasses, BogusRegisterClasses,
                           nullptr, nullptr, nullptr, LaneBitmask(~0u), nullptr,
                           nullptr) {
    InitMCRegisterInfo(nullptr, 0, 0, 0, nullptr, 0, nullptr, 0, nullptr,
                       nullptr, nullptr, nullptr, nullptr, 0, nullptr);
  }

  const MCPhysReg *
  getCalleeSavedRegs(const MachineFunction *MF) const override {
    return nullptr;
  }
  ArrayRef<const uint32_t *> getRegMasks() const override { return std::nullopt; }
  ArrayRef<const char *> getRegMaskNames() const override { return std::nullopt; }
  BitVector getReservedRegs(const MachineFunction &MF) const override {
    return BitVector();
  }
  const RegClassWeight &
  getRegClassWeight(const TargetRegisterClass *RC) const override {
    static RegClassWeight Bogus{1, 16};
    return Bogus;
  }
  unsigned getRegUnitWeight(unsigned RegUnit) const override { return 1; }
  unsigned getNumRegPressureSets() const override { return 0; }
  const char *getRegPressureSetName(unsigned Idx) const override {
    return "bogus";
  }
  unsigned getRegPressureSetLimit(const MachineFunction &MF,
                                  unsigned Idx) const override {
    return 0;
  }
  const int *
  getRegClassPressureSets(const TargetRegisterClass *RC) const override {
    static const int Bogus[] = {0, -1};
    return &Bogus[0];
  }
  const int *getRegUnitPressureSets(unsigned RegUnit) const override {
    static const int Bogus[] = {0, -1};
    return &Bogus[0];
  }

  Register getFrameRegister(const MachineFunction &MF) const override {
    return 0;
  }
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override {
    return false;

  }
};

class BogusSubtarget : public TargetSubtargetInfo {
public:
  BogusSubtarget(TargetMachine &TM)
      : TargetSubtargetInfo(Triple(""), "", "", "", {}, {}, nullptr, nullptr,
                            nullptr, nullptr, nullptr, nullptr),
        FL(), TL(TM) {}
  ~BogusSubtarget() override {}

  const TargetFrameLowering *getFrameLowering() const override { return &FL; }

  const TargetLowering *getTargetLowering() const override { return &TL; }

  const TargetInstrInfo *getInstrInfo() const override { return &TII; }

  const TargetRegisterInfo *getRegisterInfo() const override { return &TRI; }

private:
  BogusFrameLowering FL;
  BogusRegisterInfo TRI;
  BogusTargetLowering TL;
  TargetInstrInfo TII;
};

static TargetOptions getTargetOptionsForBogusMachine() {
  TargetOptions Opts;
  Opts.EmitCallSiteInfo = true;
  return Opts;
}

class BogusTargetMachine : public LLVMTargetMachine {
public:
  BogusTargetMachine()
      : LLVMTargetMachine(Target(), "", Triple(""), "", "",
                          getTargetOptionsForBogusMachine(), Reloc::Static,
                          CodeModel::Small, CodeGenOptLevel::Default),
        ST(*this) {}

  ~BogusTargetMachine() override {}

  const TargetSubtargetInfo *getSubtargetImpl(const Function &) const override {
    return &ST;
  }

private:
  BogusSubtarget ST;
};

BogusTargetMachine *createTargetMachine() {
  static BogusTargetMachine BogusTM;
  return &BogusTM;
}

std::unique_ptr<MachineFunction> createMachineFunction(LLVMContext &Ctx,
                                                       Module &M) {
  auto Type = FunctionType::get(Type::getVoidTy(Ctx), false);
  auto F = Function::Create(Type, GlobalValue::ExternalLinkage, "Test", &M);

  auto TM = createTargetMachine();
  unsigned FunctionNum = 42;
  MachineModuleInfo MMI(TM);
  const TargetSubtargetInfo &STI = *TM->getSubtargetImpl(*F);

  return std::make_unique<MachineFunction>(*F, *TM, STI, FunctionNum, MMI);
}

