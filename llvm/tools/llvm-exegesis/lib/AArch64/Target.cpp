//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "../Target.h"
#include "../Error.h"
#include "../MmapUtils.h"
#include "../SerialSnippetGenerator.h"
#include "../SnippetGenerator.h"
#include "../SubprocessMemory.h"
#include "AArch64.h"
#include "AArch64RegisterInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Process.h"
#include <vector>

#if defined(__aarch64__) && defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#ifdef HAVE_LIBPFM
#include <perfmon/perf_event.h>
#endif                 // HAVE_LIBPFM
#include <sys/prctl.h> // For PR_PAC_* constants
#ifndef PR_PAC_APIAKEY
#define PR_PAC_APIAKEY (1UL << 0)
#endif
#ifndef PR_PAC_APIBKEY
#define PR_PAC_APIBKEY (1UL << 1)
#endif
#ifndef PR_PAC_APDAKEY
#define PR_PAC_APDAKEY (1UL << 2)
#endif
#ifndef PR_PAC_APDBKEY
#define PR_PAC_APDBKEY (1UL << 3)
#endif
#endif

#define GET_AVAILABLE_OPCODE_CHECKER
#include "AArch64GenInstrInfo.inc"

namespace llvm {
namespace exegesis {

static unsigned getLoadImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 32:
    return AArch64::MOVi32imm;
  case 64:
    return AArch64::MOVi64imm;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an immediate value into a register.
static MCInst loadImmediate(MCRegister Reg, unsigned RegBitWidth,
                            const APInt &Value) {
  assert(Value.getBitWidth() <= RegBitWidth &&
         "Value must fit in the Register");
  return MCInstBuilder(getLoadImmediateOpcode(RegBitWidth))
      .addReg(Reg)
      .addImm(Value.getZExtValue());
}

static MCInst loadZPRImmediate(MCRegister Reg, unsigned RegBitWidth,
                               const APInt &Value) {
  assert(Value.getZExtValue() < (1 << 7) &&
         "Value must be in the range of the immediate opcode");
  return MCInstBuilder(AArch64::DUP_ZI_D)
      .addReg(Reg)
      .addImm(Value.getZExtValue())
      .addImm(0);
}

static MCInst loadPPRImmediate(MCRegister Reg, unsigned RegBitWidth,
                               const APInt &Value) {
  // For PPR, we typically use PTRUE instruction to set predicate registers
  return MCInstBuilder(AArch64::PTRUE_B)
      .addReg(Reg)
      .addImm(31); // All lanes true for 16 bits
}

// Generates instructions to load an immediate value into an FPCR register.
static std::vector<MCInst>
loadFPCRImmediate(MCRegister Reg, unsigned RegBitWidth, const APInt &Value) {
  MCRegister TempReg = AArch64::X16;
  MCInst LoadImm = MCInstBuilder(AArch64::MOVi64imm).addReg(TempReg).addImm(0);
  MCInst MoveToFPCR =
      MCInstBuilder(AArch64::MSR).addImm(AArch64SysReg::FPCR).addReg(TempReg);
  return {LoadImm, MoveToFPCR};
}

// Fetch base-instruction to load an FP immediate value into a register.
static unsigned getLoadFPImmediateOpcode(unsigned RegBitWidth) {
  switch (RegBitWidth) {
  case 16:
    return AArch64::FMOVH0; // FMOVHi;
  case 32:
    return AArch64::FMOVS0; // FMOVSi;
  case 64:
    return AArch64::MOVID; // FMOVDi;
  case 128:
    return AArch64::MOVIv2d_ns;
  }
  llvm_unreachable("Invalid Value Width");
}

// Generates instruction to load an FP immediate value into a register.
static MCInst loadFPImmediate(MCRegister Reg, unsigned RegBitWidth,
                              const APInt &Value) {
  assert(Value.getZExtValue() == 0 && "Expected initialisation value 0");
  MCInst Instructions =
      MCInstBuilder(getLoadFPImmediateOpcode(RegBitWidth)).addReg(Reg);
  if (RegBitWidth >= 64)
    Instructions.addOperand(MCOperand::createImm(Value.getZExtValue()));
  return Instructions;
}

static void generateRegisterStackPush(unsigned int RegToPush,
                                      std::vector<MCInst> &GeneratedCode,
                                      int imm = -16) {
  // STR [X|W]t, [SP, #simm]!: SP is decremented by default 16 bytes
  //                           before the store to maintain 16-bytes alignment.
  if (AArch64::GPR64RegClass.contains(RegToPush))
    GeneratedCode.push_back(MCInstBuilder(AArch64::STRXpre)
                                .addReg(AArch64::SP)
                                .addReg(RegToPush)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  else if (AArch64::GPR32RegClass.contains(RegToPush))
    GeneratedCode.push_back(MCInstBuilder(AArch64::STRWpre)
                                .addReg(AArch64::SP)
                                .addReg(RegToPush)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  else
    llvm_unreachable("Unsupported register class for stack push");
}

static void generateRegisterStackPop(unsigned int RegToPopTo,
                                     std::vector<MCInst> &GeneratedCode,
                                     int imm = 16) {
  // LDR Xt, [SP], #simm: SP is incremented by default 16 bytes after the load.
  if (AArch64::GPR64RegClass.contains(RegToPopTo))
    GeneratedCode.push_back(MCInstBuilder(AArch64::LDRXpost)
                                .addReg(AArch64::SP)
                                .addReg(RegToPopTo)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  else if (AArch64::GPR32RegClass.contains(RegToPopTo))
    GeneratedCode.push_back(MCInstBuilder(AArch64::LDRWpost)
                                .addReg(AArch64::SP)
                                .addReg(RegToPopTo)
                                .addReg(AArch64::SP)
                                .addImm(imm));
  else
    llvm_unreachable("Unsupported register class for stack pop");
}

void generateSysCall(long SyscallNumber, std::vector<MCInst> &GeneratedCode) {
  // AArch64 Linux follows the AAPCS (ARM Architecture Procedure Call Standard):
  // - X8 register contains the system call number
  // - X0-X5 registers contain the first 6 arguments (if any)
  // - SVC #0 instruction triggers the system call
  // - Return value is placed in X0 register
  GeneratedCode.push_back(
      loadImmediate(AArch64::X8, 64, APInt(64, SyscallNumber)));
  GeneratedCode.push_back(MCInstBuilder(AArch64::SVC).addImm(0));
}

/// Functions to save/restore system call registers
#if defined(__linux__) && defined(HAVE_LIBPFM)
constexpr std::array<unsigned, 8> SyscallArgumentRegisters{
    AArch64::X0, AArch64::X1, AArch64::X2, AArch64::X3,
    AArch64::X4, AArch64::X5, AArch64::X6, AArch64::X7,
};

static void saveSyscallRegisters(std::vector<MCInst> &GeneratedCode,
                                 unsigned ArgumentCount) {
  // AArch64 follows the AAPCS (ARM Architecture Procedure Call Standard):
  // X0-X7 registers contain the first 8 arguments.
  assert(ArgumentCount <= 8 &&
         "This implementation saves up to 8 argument registers (X0-X7)");
  // Preserve X8 (used for the syscall number/return value).
  generateRegisterStackPush(AArch64::X8, GeneratedCode);
  // Preserve the registers used to pass arguments to the system call.
  for (unsigned I = 0; I < ArgumentCount; ++I) {
    generateRegisterStackPush(SyscallArgumentRegisters[I], GeneratedCode);
  }
}

static void restoreSyscallRegisters(std::vector<MCInst> &GeneratedCode,
                                    unsigned ArgumentCount) {
  assert(ArgumentCount <= 8 &&
         "This implementation restores up to 8 argument registers (X0-X7)");
  // Restore registers in reverse order
  for (int I = ArgumentCount - 1; I >= 0; --I) {
    generateRegisterStackPop(SyscallArgumentRegisters[I], GeneratedCode);
  }
  generateRegisterStackPop(AArch64::X8, GeneratedCode);
}
#endif // __linux__ && HAVE_LIBPFM
#include "AArch64GenExegesis.inc"

namespace {

// Use X19 as the loop counter register since it's a callee-saved register
// that's available for temporary use.
constexpr MCPhysReg kDefaultLoopCounterReg = AArch64::X19;

class ExegesisAArch64Target : public ExegesisTarget {
public:
  ExegesisAArch64Target()
      : ExegesisTarget(AArch64CpuPfmCounters, AArch64_MC::isOpcodeAvailable) {}

  enum ArgumentRegisters {
    CodeSize = AArch64::X12,
    AuxiliaryMemoryFD = AArch64::X13,
    TempRegister = AArch64::X16,
  };

  std::vector<MCInst> _generateRegisterStackPop(MCRegister Reg,
                                                int imm = 0) const override {
    std::vector<MCInst> Insts;
    if (AArch64::GPR32RegClass.contains(Reg) ||
        AArch64::GPR64RegClass.contains(Reg)) {
      generateRegisterStackPop(Reg, Insts, imm);
      return Insts;
    }
    return {};
  }

  Error randomizeTargetMCOperand(const Instruction &Instr, const Variable &Var,
                                 MCOperand &AssignedValue,
                                 const BitVector &ForbiddenRegs) const override;

private:
#ifdef __linux__
  std::vector<MCInst> generateExitSyscall(unsigned ExitCode) const override;
  std::vector<MCInst>
  generateMmap(uintptr_t Address, size_t Length,
               uintptr_t FileDescriptorAddress) const override;
  void generateMmapAuxMem(std::vector<MCInst> &GeneratedCode) const override;
  std::vector<MCInst> generateMemoryInitialSetup() const override;
  std::vector<MCInst> setStackRegisterToAuxMem() const override;
  uintptr_t getAuxiliaryMemoryStartAddress() const override;
  std::vector<MCInst> configurePerfCounter(long Request,
                                           bool SaveRegisters) const override;
  std::vector<MCRegister> getArgumentRegisters() const override;
  std::vector<MCRegister> getRegistersNeedSaving() const override;
#endif // __linux__

  std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, MCRegister Reg,
                               const APInt &Value) const override {
    if (AArch64::GPR32RegClass.contains(Reg))
      return {loadImmediate(Reg, 32, Value)};
    if (AArch64::GPR64RegClass.contains(Reg))
      return {loadImmediate(Reg, 64, Value)};
    if (AArch64::PPRRegClass.contains(Reg))
      return {loadPPRImmediate(Reg, 16, Value)};
    if (AArch64::FPR8RegClass.contains(Reg))
      return {loadFPImmediate(Reg - AArch64::B0 + AArch64::D0, 64, Value)};
    if (AArch64::FPR16RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 16, Value)};
    if (AArch64::FPR32RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 32, Value)};
    if (AArch64::FPR64RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 64, Value)};
    if (AArch64::FPR128RegClass.contains(Reg))
      return {loadFPImmediate(Reg, 128, Value)};
    if (AArch64::ZPRRegClass.contains(Reg))
      return {loadZPRImmediate(Reg, 128, Value)};
    if (Reg == AArch64::FPCR)
      return {loadFPCRImmediate(Reg, 32, Value)};

    errs() << "setRegTo is not implemented, results will be unreliable\n";
    return {};
  }
  MCRegister getDefaultLoopCounterRegister(const Triple &) const override {
    return kDefaultLoopCounterReg;
  }

  void decrementLoopCounterAndJump(MachineBasicBlock &MBB,
                                   MachineBasicBlock &TargetMBB,
                                   const MCInstrInfo &MII,
                                   MCRegister LoopRegister) const override {
    // subs LoopRegister, LoopRegister, #1
    BuildMI(&MBB, DebugLoc(), MII.get(AArch64::SUBSXri))
        .addDef(LoopRegister)
        .addUse(LoopRegister)
        .addImm(1)  // Subtract 1
        .addImm(0); // No shift amount
    // b.ne TargetMBB
    BuildMI(&MBB, DebugLoc(), MII.get(AArch64::Bcc))
        .addImm(AArch64CC::NE)
        .addMBB(&TargetMBB);
  }

  // Registers that should not be selected for use in snippets.
  const MCPhysReg UnavailableRegisters[1] = {kDefaultLoopCounterReg};
  ArrayRef<MCPhysReg> getUnavailableRegisters() const override {
    return UnavailableRegisters;
  }

  bool matchesArch(Triple::ArchType Arch) const override {
    return Arch == Triple::aarch64 || Arch == Triple::aarch64_be;
  }

  void addTargetSpecificPasses(PassManagerBase &PM) const override {
    // Function return is a pseudo-instruction that needs to be expanded
    PM.add(createAArch64ExpandPseudoPass());
  }
};

Error ExegesisAArch64Target::randomizeTargetMCOperand(
    const Instruction &Instr, const Variable &Var, MCOperand &AssignedValue,
    const BitVector &ForbiddenRegs) const {
  const Operand &Op = Instr.getPrimaryOperand(Var);
  const auto OperandType = Op.getExplicitOperandInfo().OperandType;
  // NOTE: To resolve "Not all operands were initialized by snippet generator"
  // Requires OperandType to be defined for such opcode's operands in AArch64
  // tablegen files. And omit introduced OperandType(s).

  // Hacky Fix: Defaulting all OPERAND_UNKNOWN to immediate value 0 works with a
  // limitation that it introduces illegal instruction error for system
  // instructions. System instructions will need to be omitted with OperandType
  // or opcode specific values to avoid generating invalid encodings or
  // unreliable benchmark results for these system-level instructions.
  //  Implement opcode-specific immediate value handling for system instrs:
  //   - MRS/MSR: Use valid system register encodings (e.g., NZCV, FPCR, FPSR)
  //   - MSRpstatesvcrImm1: Use valid PSTATE field encodings (e.g., SPSel,
  //   DAIFSet)
  //   - SYSLxt/SYSxt: Use valid system instruction encodings with proper
  //   CRn/CRm/op values
  //   - UDF: Use valid undefined instruction immediate ranges (0-65535)

  switch (OperandType) {
  // MSL (Masking Shift Left) imm operand for 32-bit splatted SIMD constants
  // Correspond to AArch64InstructionSelector::tryAdvSIMDModImm321s()
  case llvm::AArch64::OPERAND_SHIFT_MSL: {
    // There are two valid encodings:
    //   - Type 7: imm at [15:8], [47:40], shift = 264 (0x108) → msl #8
    //   - Type 8: imm at [23:16], [55:48], shift = 272 (0x110) → msl #16
    //     Corresponds AArch64_AM::encodeAdvSIMDModImmType7()
    // But, v2s_msl and v4s_msl instructions accept either form,
    // Thus, Arbitrarily chosing 264 (msl #8) for simplicity.
    AssignedValue = MCOperand::createImm(264);
    return Error::success();
  }
  case llvm::AArch64::OPERAND_IMPLICIT_IMM_0:
    AssignedValue = MCOperand::createImm(0);
    return Error::success();
  case MCOI::OperandType::OPERAND_PCREL:
    AssignedValue = MCOperand::createImm(8);
    return Error::success();
  default:
    break;
  }

  return make_error<Failure>(
      Twine("Unimplemented operand type: MCOI::OperandType:")
          .concat(Twine(static_cast<int>(OperandType))));
}

} // namespace

#ifdef __linux__
static constexpr const uintptr_t VAddressSpaceCeiling = 0x0000800000000000;

std::vector<MCInst>
ExegesisAArch64Target::generateExitSyscall(unsigned ExitCode) const {
  std::vector<MCInst> ExitCallCode;
  ExitCallCode.push_back(loadImmediate(AArch64::X0, 64, APInt(64, ExitCode)));
  generateSysCall(SYS_exit, ExitCallCode); // SYS_exit is 93
  return ExitCallCode;
}

std::vector<MCInst>
ExegesisAArch64Target::generateMmap(uintptr_t Address, size_t Length,
                                    uintptr_t FileDescriptorAddress) const {
  // mmap(address, length, prot, flags, fd, offset=0)
  int flags = MAP_SHARED;
  int fd = -1;
  if (fd == -1) {
    dbgs() << "Warning: generateMmap using anonymous mapping\n";
    flags |= MAP_ANONYMOUS;
  }
  if (Address != 0)
    flags |= MAP_FIXED_NOREPLACE;
  std::vector<MCInst> MmapCode;
  MmapCode.push_back(
      loadImmediate(AArch64::X0, 64, APInt(64, Address))); // map adr
  MmapCode.push_back(
      loadImmediate(AArch64::X1, 64, APInt(64, Length))); // length
  MmapCode.push_back(loadImmediate(AArch64::X2, 64,
                                   APInt(64, PROT_READ | PROT_WRITE))); // prot
  MmapCode.push_back(loadImmediate(AArch64::X3, 64, APInt(64, flags))); // flags
  // FIXME: Loading [FileDescriptorAddress] as fd leds syscall to return error
  MmapCode.push_back(loadImmediate(AArch64::X4, 64, APInt(64, fd))); // fd
  MmapCode.push_back(loadImmediate(AArch64::X5, 64, APInt(64, 0)));  // offset
  generateSysCall(SYS_mmap, MmapCode); // SYS_mmap is 222
  return MmapCode;
}

void ExegesisAArch64Target::generateMmapAuxMem(
    std::vector<MCInst> &GeneratedCode) const {
  int fd = -1;
  int flags = MAP_SHARED;
  uintptr_t address = getAuxiliaryMemoryStartAddress();
  if (fd == -1) {
    dbgs() << "Warning: generateMmapAuxMem using anonymous mapping\n";
    flags |= MAP_ANONYMOUS;
  }
  if (address != 0)
    flags |= MAP_FIXED_NOREPLACE;
  int prot = PROT_READ | PROT_WRITE;

  GeneratedCode.push_back(
      loadImmediate(AArch64::X0, 64, APInt(64, address))); // map adr
  GeneratedCode.push_back(loadImmediate(
      AArch64::X1, 64,
      APInt(64, SubprocessMemory::AuxiliaryMemorySize))); // length
  GeneratedCode.push_back(
      loadImmediate(AArch64::X2, 64, APInt(64, prot))); // prot
  GeneratedCode.push_back(
      loadImmediate(AArch64::X3, 64, APInt(64, flags))); // flags
  GeneratedCode.push_back(loadImmediate(AArch64::X4, 64, APInt(64, fd))); // fd
  GeneratedCode.push_back(
      loadImmediate(AArch64::X5, 64, APInt(64, 0))); // offset
  generateSysCall(SYS_mmap, GeneratedCode);          // SYS_mmap is 222
}

std::vector<MCInst> ExegesisAArch64Target::generateMemoryInitialSetup() const {
  std::vector<MCInst> MemoryInitialSetupCode;
  generateMmapAuxMem(MemoryInitialSetupCode);

  // If using fixed address for auxiliary memory skip this step,
  // When using dynamic memory allocation (non-fixed address), we must preserve
  // the mmap return value (X0) which contains the allocated memory address.
  // This value is saved to the stack to ensure registers requiring memory
  // access can retrieve the correct address even if X0 is modified by
  // intermediate code.
  generateRegisterStackPush(AArch64::X0, MemoryInitialSetupCode);
  // FIXME: Ensure stack pointer remains stable to prevent loss of saved address
  return MemoryInitialSetupCode;
}

std::vector<MCInst> ExegesisAArch64Target::setStackRegisterToAuxMem() const {
  std::vector<MCInst> instructions; // NOP
  // Motivation unclear, found no need for this in AArch64.
  // TODO:  Implement this, if required.
  dbgs() << "Warning: setStackRegisterToAuxMem called but not required for "
            "AArch64\n";
  return instructions;
}

uintptr_t ExegesisAArch64Target::getAuxiliaryMemoryStartAddress() const {
  // Return the second to last page in the virtual address space to try and
  // prevent interference with memory annotations in the snippet
  // FIXME: Why 2 pages?
  return VAddressSpaceCeiling - (2 * llvm::sys::Process::getPageSizeEstimate());
}

std::vector<MCInst>
ExegesisAArch64Target::configurePerfCounter(long Request,
                                            bool SaveRegisters) const {
  std::vector<MCInst> ConfigurePerfCounterCode;
#ifdef HAVE_LIBPFM
  if (SaveRegisters)
    saveSyscallRegisters(ConfigurePerfCounterCode, 3);

  // Load actual file descriptor from auxiliary memory location [address + 0]
  // CounterFileDescriptor was stored at AuxiliaryMemoryMapping[0]
  dbgs() << "Warning: configurePerfCounter ioctl syscall failing\n";
  // FIXME: Ensure file descriptor is correctly populated at auxiliary memory
  // address before ioctl syscall to avoid unreliable benchmark results
  ConfigurePerfCounterCode.push_back(
      loadImmediate(ArgumentRegisters::TempRegister, 64,
                    APInt(64, getAuxiliaryMemoryStartAddress())));
  ConfigurePerfCounterCode.push_back(
      MCInstBuilder(AArch64::LDRWui)
          .addReg(AArch64::W0)
          .addReg(ArgumentRegisters::TempRegister)
          .addImm(0));
  ConfigurePerfCounterCode.push_back(
      loadImmediate(AArch64::X1, 64, APInt(64, Request))); // cmd
  ConfigurePerfCounterCode.push_back(
      loadImmediate(AArch64::X2, 64, APInt(64, PERF_IOC_FLAG_GROUP))); // arg
  generateSysCall(SYS_ioctl, ConfigurePerfCounterCode); // SYS_ioctl is 29

  if (SaveRegisters)
    restoreSyscallRegisters(ConfigurePerfCounterCode, 3);
#endif
  return ConfigurePerfCounterCode;
}

std::vector<MCRegister> ExegesisAArch64Target::getArgumentRegisters() const {
  return {AArch64::X0, AArch64::X1};
}

std::vector<MCRegister> ExegesisAArch64Target::getRegistersNeedSaving() const {
  return {
      AArch64::X0,
      AArch64::X1,
      AArch64::X2,
      AArch64::X3,
      AArch64::X4,
      AArch64::X5,
      AArch64::X8,
      ArgumentRegisters::TempRegister,
      ArgumentRegisters::CodeSize,
      ArgumentRegisters::AuxiliaryMemoryFD,
  };
}

#endif // __linux__

static ExegesisTarget *getTheExegesisAArch64Target() {
  static ExegesisAArch64Target Target;
  return &Target;
}

void InitializeAArch64ExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisAArch64Target());
}

} // namespace exegesis
} // namespace llvm
