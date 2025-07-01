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
#include <vector>
#define DEBUG_TYPE "exegesis-aarch64-target"

#if defined(__aarch64__) && defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h> // for getpagesize()
#ifdef HAVE_LIBPFM
#include <perfmon/perf_event.h>
#endif                   // HAVE_LIBPFM
#include <linux/prctl.h> // For PR_PAC_* constants
#include <sys/prctl.h>
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
#ifdef __linux__
constexpr std::array<unsigned, 8> SyscallArgumentRegisters{
    AArch64::X0, AArch64::X1, AArch64::X2, AArch64::X3,
    AArch64::X4, AArch64::X5, AArch64::X6, AArch64::X7,
};

static void saveSysCallRegisters(std::vector<MCInst> &GeneratedCode,
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

static void restoreSysCallRegisters(std::vector<MCInst> &GeneratedCode,
                                    unsigned ArgumentCount) {
  assert(ArgumentCount <= 8 &&
         "This implementation restores up to 8 argument registers (X0-X7)");
  // Restore registers in reverse order
  for (int I = ArgumentCount - 1; I >= 0; --I) {
    generateRegisterStackPop(SyscallArgumentRegisters[I], GeneratedCode);
  }
  generateRegisterStackPop(AArch64::X8, GeneratedCode);
}
#endif // __linux__
#include "AArch64GenExegesis.inc"

namespace {

class ExegesisAArch64Target : public ExegesisTarget {
public:
  ExegesisAArch64Target()
      : ExegesisTarget(AArch64CpuPfmCounters, AArch64_MC::isOpcodeAvailable) {}

  enum ReservedRegisters {
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

  bool matchesArch(Triple::ArchType Arch) const override {
    return Arch == Triple::aarch64 || Arch == Triple::aarch64_be;
  }

  void addTargetSpecificPasses(PassManagerBase &PM) const override {
    // Function return is a pseudo-instruction that needs to be expanded
    PM.add(createAArch64ExpandPseudoPass());
  }
};

} // namespace

#ifdef __linux__
// true : let use of fixed address to Virtual Address Space Ceiling
// false: let kernel choose the address of the auxiliary memory
bool UseFixedAddress = true;

static constexpr const uintptr_t VAddressSpaceCeiling = 0x0000800000000000;

static void generateRoundToNearestPage(unsigned int TargetRegister,
                                       std::vector<MCInst> &GeneratedCode) {
  int PageSizeShift = static_cast<int>(round(log2(getpagesize())));
  // Round down to the nearest page by getting rid of the least significant bits
  // representing location in the page.

  // Single instruction using AND with inverted mask (effectively BIC)
  uint64_t BitsToClearMask = (1ULL << PageSizeShift) - 1; // 0xFFF
  uint64_t AndMask = ~BitsToClearMask;                    // ...FFFFFFFFFFFF000
  GeneratedCode.push_back(MCInstBuilder(AArch64::ANDXri)
                              .addReg(TargetRegister) // Xd
                              .addReg(TargetRegister) // Xn
                              .addImm(AndMask)        // imm bitmask
  );
}

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
  if (Address != 0) {
    flags |= MAP_FIXED_NOREPLACE;
  }
  std::vector<MCInst> MmapCode;
  MmapCode.push_back(
      loadImmediate(AArch64::X0, 64, APInt(64, Address))); // map adr
  MmapCode.push_back(
      loadImmediate(AArch64::X1, 64, APInt(64, Length))); // length
  MmapCode.push_back(loadImmediate(AArch64::X2, 64,
                                   APInt(64, PROT_READ | PROT_WRITE))); // prot
  MmapCode.push_back(loadImmediate(AArch64::X3, 64, APInt(64, flags))); // flags
  // FIXME: File descriptor address is not initialized.
  // Copy file descriptor location from aux memory into X4
  MmapCode.push_back(
      loadImmediate(AArch64::X4, 64, APInt(64, FileDescriptorAddress))); // fd
  MmapCode.push_back(loadImmediate(AArch64::X5, 64, APInt(64, 0))); // offset
  generateSysCall(SYS_mmap, MmapCode); // SYS_mmap is 222
  return MmapCode;
}

void ExegesisAArch64Target::generateMmapAuxMem(
    std::vector<MCInst> &GeneratedCode) const {
  int fd = -1;
  int flags = MAP_SHARED;
  uintptr_t address = getAuxiliaryMemoryStartAddress();
  if (fd == -1)
    flags |= MAP_ANONYMOUS;
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
  generateMmapAuxMem(MemoryInitialSetupCode); // FIXME: Uninit file descriptor

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
  // TODO: Implement this, Found no need for this in AArch64.
  return instructions;
}

uintptr_t ExegesisAArch64Target::getAuxiliaryMemoryStartAddress() const {
  if (!UseFixedAddress)
    // Allow kernel to select an appropriate memory address
    return 0;
  // Return the second to last page in the virtual address space
  // to try and prevent interference with memory annotations in the snippet
  // VAddressSpaceCeiling = 0x0000800000000000
  // FIXME: Why 2 pages?
  return VAddressSpaceCeiling - (2 * getpagesize());
}

std::vector<MCInst>
ExegesisAArch64Target::configurePerfCounter(long Request,
                                            bool SaveRegisters) const {
  std::vector<MCInst> ConfigurePerfCounterCode; // NOP
  // FIXME: SYSCALL exits with EBADF error - file descriptor is invalid
  // No file is opened previosly to add as file descriptor
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
      ReservedRegisters::TempRegister,
      ReservedRegisters::CodeSize,
      ReservedRegisters::AuxiliaryMemoryFD,
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
