//===-- X86Subtarget.h - Define Subtarget for the X86 ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the X86 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_X86_X86SUBTARGET_H
#define LLVM_LIB_TARGET_X86_X86SUBTARGET_H

#include "X86FrameLowering.h"
#include "X86ISelLowering.h"
#include "X86InstrInfo.h"
#include "X86SelectionDAGInfo.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/GlobalISel/GISelAccessor.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/MC/MCInstrItineraries.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <memory>

#define GET_SUBTARGETINFO_HEADER
#include "X86GenSubtargetInfo.inc"

namespace llvm {

class GlobalValue;

/// The X86 backend supports a number of different styles of PIC.
///
namespace PICStyles {

enum Style {
  StubPIC,          // Used on i386-darwin in pic mode.
  GOT,              // Used on 32 bit elf on when in pic mode.
  RIPRel,           // Used on X86-64 when in pic mode.
  None              // Set when not in pic mode.
};

} // end namespace PICStyles

class X86Subtarget final : public X86GenSubtargetInfo {
protected:
  enum X86SSEEnum {
    NoSSE, SSE1, SSE2, SSE3, SSSE3, SSE41, SSE42, AVX, AVX2, AVX512F
  };

  enum X863DNowEnum {
    NoThreeDNow, MMX, ThreeDNow, ThreeDNowA
  };

  enum X86ProcFamilyEnum {
    Others, IntelAtom, IntelSLM
  };

  /// X86 processor family: Intel Atom, and others
  X86ProcFamilyEnum X86ProcFamily;

  /// Which PIC style to use
  PICStyles::Style PICStyle;

  const TargetMachine &TM;

  /// SSE1, SSE2, SSE3, SSSE3, SSE41, SSE42, or none supported.
  X86SSEEnum X86SSELevel;

  /// MMX, 3DNow, 3DNow Athlon, or none supported.
  X863DNowEnum X863DNowLevel;

  /// True if the processor supports X87 instructions.
  bool HasX87;

  /// True if this processor has conditional move instructions
  /// (generally pentium pro+).
  bool HasCMov;

  /// True if the processor supports X86-64 instructions.
  bool HasX86_64;

  /// True if the processor supports POPCNT.
  bool HasPOPCNT;

  /// True if the processor supports SSE4A instructions.
  bool HasSSE4A;

  /// Target has AES instructions
  bool HasAES;

  /// Target has FXSAVE/FXRESTOR instructions
  bool HasFXSR;

  /// Target has XSAVE instructions
  bool HasXSAVE;

  /// Target has XSAVEOPT instructions
  bool HasXSAVEOPT;

  /// Target has XSAVEC instructions
  bool HasXSAVEC;

  /// Target has XSAVES instructions
  bool HasXSAVES;

  /// Target has carry-less multiplication
  bool HasPCLMUL;

  /// Target has 3-operand fused multiply-add
  bool HasFMA;

  /// Target has 4-operand fused multiply-add
  bool HasFMA4;

  /// Target has XOP instructions
  bool HasXOP;

  /// Target has TBM instructions.
  bool HasTBM;

  /// True if the processor has the MOVBE instruction.
  bool HasMOVBE;

  /// True if the processor has the RDRAND instruction.
  bool HasRDRAND;

  /// Processor has 16-bit floating point conversion instructions.
  bool HasF16C;

  /// Processor has FS/GS base insturctions.
  bool HasFSGSBase;

  /// Processor has LZCNT instruction.
  bool HasLZCNT;

  /// Processor has BMI1 instructions.
  bool HasBMI;

  /// Processor has BMI2 instructions.
  bool HasBMI2;

  /// Processor has VBMI instructions.
  bool HasVBMI;

  /// Processor has Integer Fused Multiply Add
  bool HasIFMA;

  /// Processor has RTM instructions.
  bool HasRTM;

  /// Processor has ADX instructions.
  bool HasADX;

  /// Processor has SHA instructions.
  bool HasSHA;

  /// Processor has PRFCHW instructions.
  bool HasPRFCHW;

  /// Processor has RDSEED instructions.
  bool HasRDSEED;

  /// Processor has LAHF/SAHF instructions.
  bool HasLAHFSAHF;

  /// Processor has MONITORX/MWAITX instructions.
  bool HasMWAITX;

  /// Processor has Cache Line Zero instruction
  bool HasCLZERO;

  /// Processor has Prefetch with intent to Write instruction
  bool HasPFPREFETCHWT1;

  /// True if BT (bit test) of memory instructions are slow.
  bool IsBTMemSlow;

  /// True if SHLD instructions are slow.
  bool IsSHLDSlow;

  /// True if the PMULLD instruction is slow compared to PMULLW/PMULHW and
  //  PMULUDQ.
  bool IsPMULLDSlow;

  /// True if unaligned memory accesses of 16-bytes are slow.
  bool IsUAMem16Slow;

  /// True if unaligned memory accesses of 32-bytes are slow.
  bool IsUAMem32Slow;

  /// True if SSE operations can have unaligned memory operands.
  /// This may require setting a configuration bit in the processor.
  bool HasSSEUnalignedMem;

  /// True if this processor has the CMPXCHG16B instruction;
  /// this is true for most x86-64 chips, but not the first AMD chips.
  bool HasCmpxchg16b;

  /// True if the LEA instruction should be used for adjusting
  /// the stack pointer. This is an optimization for Intel Atom processors.
  bool UseLeaForSP;

  /// True if there is no performance penalty to writing only the lower parts
  /// of a YMM or ZMM register without clearing the upper part.
  bool HasFastPartialYMMorZMMWrite;

  /// True if hardware SQRTSS instruction is at least as fast (latency) as
  /// RSQRTSS followed by a Newton-Raphson iteration.
  bool HasFastScalarFSQRT;

  /// True if hardware SQRTPS/VSQRTPS instructions are at least as fast
  /// (throughput) as RSQRTPS/VRSQRTPS followed by a Newton-Raphson iteration.
  bool HasFastVectorFSQRT;

  /// True if 8-bit divisions are significantly faster than
  /// 32-bit divisions and should be used when possible.
  bool HasSlowDivide32;

  /// True if 32-bit divides are significantly faster than
  /// 64-bit divisions and should be used when possible.
  bool HasSlowDivide64;

  /// True if LZCNT instruction is fast.
  bool HasFastLZCNT;

  /// True if SHLD based rotate is fast.
  bool HasFastSHLDRotate;

  /// True if the short functions should be padded to prevent
  /// a stall when returning too early.
  bool PadShortFunctions;

  /// True if the Calls with memory reference should be converted
  /// to a register-based indirect call.
  bool CallRegIndirect;

  /// True if the LEA instruction inputs have to be ready at address generation
  /// (AG) time.
  bool LEAUsesAG;

  /// True if the LEA instruction with certain arguments is slow
  bool SlowLEA;

  /// True if INC and DEC instructions are slow when writing to flags
  bool SlowIncDec;

  /// Processor has AVX-512 PreFetch Instructions
  bool HasPFI;

  /// Processor has AVX-512 Exponential and Reciprocal Instructions
  bool HasERI;

  /// Processor has AVX-512 Conflict Detection Instructions
  bool HasCDI;

  /// Processor has AVX-512 Doubleword and Quadword instructions
  bool HasDQI;

  /// Processor has AVX-512 Byte and Word instructions
  bool HasBWI;

  /// Processor has AVX-512 Vector Length eXtenstions
  bool HasVLX;

  /// Processor has PKU extenstions
  bool HasPKU;

  /// Processor supports MPX - Memory Protection Extensions
  bool HasMPX;

  /// Processor has Software Guard Extensions
  bool HasSGX;

  /// Processor supports Flush Cache Line instruction
  bool HasCLFLUSHOPT;

  /// Processor supports Cache Line Write Back instruction
  bool HasCLWB;

  /// Use software floating point for code generation.
  bool UseSoftFloat;

  /// The minimum alignment known to hold of the stack frame on
  /// entry to the function and which must be maintained by every function.
  unsigned stackAlignment;

  /// Max. memset / memcpy size that is turned into rep/movs, rep/stos ops.
  ///
  unsigned MaxInlineSizeThreshold;

  /// What processor and OS we're targeting.
  Triple TargetTriple;

  /// Instruction itineraries for scheduling
  InstrItineraryData InstrItins;

  /// Gather the accessor points to GlobalISel-related APIs.
  /// This is used to avoid ifndefs spreading around while GISel is
  /// an optional library.
  std::unique_ptr<GISelAccessor> GISel;

private:
  /// Override the stack alignment.
  unsigned StackAlignOverride;

  /// True if compiling for 64-bit, false for 16-bit or 32-bit.
  bool In64BitMode;

  /// True if compiling for 32-bit, false for 16-bit or 64-bit.
  bool In32BitMode;

  /// True if compiling for 16-bit, false for 32-bit or 64-bit.
  bool In16BitMode;

  X86SelectionDAGInfo TSInfo;
  // Ordering here is important. X86InstrInfo initializes X86RegisterInfo which
  // X86TargetLowering needs.
  X86InstrInfo InstrInfo;
  X86TargetLowering TLInfo;
  X86FrameLowering FrameLowering;

public:
  /// This constructor initializes the data members to match that
  /// of the specified triple.
  ///
  X86Subtarget(const Triple &TT, StringRef CPU, StringRef FS,
               const X86TargetMachine &TM, unsigned StackAlignOverride);

  /// This object will take onwership of \p GISelAccessor.
  void setGISelAccessor(GISelAccessor &GISel) { this->GISel.reset(&GISel); }

  const X86TargetLowering *getTargetLowering() const override {
    return &TLInfo;
  }

  const X86InstrInfo *getInstrInfo() const override { return &InstrInfo; }

  const X86FrameLowering *getFrameLowering() const override {
    return &FrameLowering;
  }

  const X86SelectionDAGInfo *getSelectionDAGInfo() const override {
    return &TSInfo;
  }

  const X86RegisterInfo *getRegisterInfo() const override {
    return &getInstrInfo()->getRegisterInfo();
  }

  /// Returns the minimum alignment known to hold of the
  /// stack frame on entry to the function and which must be maintained by every
  /// function for this subtarget.
  unsigned getStackAlignment() const { return stackAlignment; }

  /// Returns the maximum memset / memcpy size
  /// that still makes it profitable to inline the call.
  unsigned getMaxInlineSizeThreshold() const { return MaxInlineSizeThreshold; }

  /// ParseSubtargetFeatures - Parses features string setting specified
  /// subtarget options.  Definition of function is auto generated by tblgen.
  void ParseSubtargetFeatures(StringRef CPU, StringRef FS);

  /// Methods used by Global ISel
  const CallLowering *getCallLowering() const override;
  const InstructionSelector *getInstructionSelector() const override;
  const LegalizerInfo *getLegalizerInfo() const override;
  const RegisterBankInfo *getRegBankInfo() const override;

private:
  /// Initialize the full set of dependencies so we can use an initializer
  /// list for X86Subtarget.
  X86Subtarget &initializeSubtargetDependencies(StringRef CPU, StringRef FS);
  void initializeEnvironment();
  void initSubtargetFeatures(StringRef CPU, StringRef FS);

public:
  /// Is this x86_64? (disregarding specific ABI / programming model)
  bool is64Bit() const {
    return In64BitMode;
  }

  bool is32Bit() const {
    return In32BitMode;
  }

  bool is16Bit() const {
    return In16BitMode;
  }

  /// Is this x86_64 with the ILP32 programming model (x32 ABI)?
  bool isTarget64BitILP32() const {
    return In64BitMode && (TargetTriple.getEnvironment() == Triple::GNUX32 ||
                           TargetTriple.isOSNaCl());
  }

  /// Is this x86_64 with the LP64 programming model (standard AMD64, no x32)?
  bool isTarget64BitLP64() const {
    return In64BitMode && (TargetTriple.getEnvironment() != Triple::GNUX32 &&
                           !TargetTriple.isOSNaCl());
  }

  PICStyles::Style getPICStyle() const { return PICStyle; }
  void setPICStyle(PICStyles::Style Style)  { PICStyle = Style; }

  bool hasX87() const { return HasX87; }
  bool hasCMov() const { return HasCMov; }
  bool hasSSE1() const { return X86SSELevel >= SSE1; }
  bool hasSSE2() const { return X86SSELevel >= SSE2; }
  bool hasSSE3() const { return X86SSELevel >= SSE3; }
  bool hasSSSE3() const { return X86SSELevel >= SSSE3; }
  bool hasSSE41() const { return X86SSELevel >= SSE41; }
  bool hasSSE42() const { return X86SSELevel >= SSE42; }
  bool hasAVX() const { return X86SSELevel >= AVX; }
  bool hasAVX2() const { return X86SSELevel >= AVX2; }
  bool hasAVX512() const { return X86SSELevel >= AVX512F; }
  bool hasFp256() const { return hasAVX(); }
  bool hasInt256() const { return hasAVX2(); }
  bool hasSSE4A() const { return HasSSE4A; }
  bool hasMMX() const { return X863DNowLevel >= MMX; }
  bool has3DNow() const { return X863DNowLevel >= ThreeDNow; }
  bool has3DNowA() const { return X863DNowLevel >= ThreeDNowA; }
  bool hasPOPCNT() const { return HasPOPCNT; }
  bool hasAES() const { return HasAES; }
  bool hasFXSR() const { return HasFXSR; }
  bool hasXSAVE() const { return HasXSAVE; }
  bool hasXSAVEOPT() const { return HasXSAVEOPT; }
  bool hasXSAVEC() const { return HasXSAVEC; }
  bool hasXSAVES() const { return HasXSAVES; }
  bool hasPCLMUL() const { return HasPCLMUL; }
  // Prefer FMA4 to FMA - its better for commutation/memory folding and
  // has equal or better performance on all supported targets.
  bool hasFMA() const { return (HasFMA || hasAVX512()) && !HasFMA4; }
  bool hasFMA4() const { return HasFMA4; }
  bool hasAnyFMA() const { return hasFMA() || hasFMA4(); }
  bool hasXOP() const { return HasXOP; }
  bool hasTBM() const { return HasTBM; }
  bool hasMOVBE() const { return HasMOVBE; }
  bool hasRDRAND() const { return HasRDRAND; }
  bool hasF16C() const { return HasF16C; }
  bool hasFSGSBase() const { return HasFSGSBase; }
  bool hasLZCNT() const { return HasLZCNT; }
  bool hasBMI() const { return HasBMI; }
  bool hasBMI2() const { return HasBMI2; }
  bool hasVBMI() const { return HasVBMI; }
  bool hasIFMA() const { return HasIFMA; }
  bool hasRTM() const { return HasRTM; }
  bool hasADX() const { return HasADX; }
  bool hasSHA() const { return HasSHA; }
  bool hasPRFCHW() const { return HasPRFCHW; }
  bool hasRDSEED() const { return HasRDSEED; }
  bool hasLAHFSAHF() const { return HasLAHFSAHF; }
  bool hasMWAITX() const { return HasMWAITX; }
  bool hasCLZERO() const { return HasCLZERO; }
  bool isBTMemSlow() const { return IsBTMemSlow; }
  bool isSHLDSlow() const { return IsSHLDSlow; }
  bool isPMULLDSlow() const { return IsPMULLDSlow; }
  bool isUnalignedMem16Slow() const { return IsUAMem16Slow; }
  bool isUnalignedMem32Slow() const { return IsUAMem32Slow; }
  bool hasSSEUnalignedMem() const { return HasSSEUnalignedMem; }
  bool hasCmpxchg16b() const { return HasCmpxchg16b; }
  bool useLeaForSP() const { return UseLeaForSP; }
  bool hasFastPartialYMMorZMMWrite() const {
    return HasFastPartialYMMorZMMWrite;
  }
  bool hasFastScalarFSQRT() const { return HasFastScalarFSQRT; }
  bool hasFastVectorFSQRT() const { return HasFastVectorFSQRT; }
  bool hasFastLZCNT() const { return HasFastLZCNT; }
  bool hasFastSHLDRotate() const { return HasFastSHLDRotate; }
  bool hasSlowDivide32() const { return HasSlowDivide32; }
  bool hasSlowDivide64() const { return HasSlowDivide64; }
  bool padShortFunctions() const { return PadShortFunctions; }
  bool callRegIndirect() const { return CallRegIndirect; }
  bool LEAusesAG() const { return LEAUsesAG; }
  bool slowLEA() const { return SlowLEA; }
  bool slowIncDec() const { return SlowIncDec; }
  bool hasCDI() const { return HasCDI; }
  bool hasPFI() const { return HasPFI; }
  bool hasERI() const { return HasERI; }
  bool hasDQI() const { return HasDQI; }
  bool hasBWI() const { return HasBWI; }
  bool hasVLX() const { return HasVLX; }
  bool hasPKU() const { return HasPKU; }
  bool hasMPX() const { return HasMPX; }
  bool hasCLFLUSHOPT() const { return HasCLFLUSHOPT; }

  bool isXRaySupported() const override { return is64Bit(); }

  bool isAtom() const { return X86ProcFamily == IntelAtom; }
  bool isSLM() const { return X86ProcFamily == IntelSLM; }
  bool useSoftFloat() const { return UseSoftFloat; }

  /// Use mfence if we have SSE2 or we're on x86-64 (even if we asked for
  /// no-sse2). There isn't any reason to disable it if the target processor
  /// supports it.
  bool hasMFence() const { return hasSSE2() || is64Bit(); }

  const Triple &getTargetTriple() const { return TargetTriple; }

  bool isTargetDarwin() const { return TargetTriple.isOSDarwin(); }
  bool isTargetFreeBSD() const { return TargetTriple.isOSFreeBSD(); }
  bool isTargetDragonFly() const { return TargetTriple.isOSDragonFly(); }
  bool isTargetSolaris() const { return TargetTriple.isOSSolaris(); }
  bool isTargetPS4() const { return TargetTriple.isPS4CPU(); }

  bool isTargetELF() const { return TargetTriple.isOSBinFormatELF(); }
  bool isTargetCOFF() const { return TargetTriple.isOSBinFormatCOFF(); }
  bool isTargetMachO() const { return TargetTriple.isOSBinFormatMachO(); }

  bool isTargetLinux() const { return TargetTriple.isOSLinux(); }
  bool isTargetKFreeBSD() const { return TargetTriple.isOSKFreeBSD(); }
  bool isTargetGlibc() const { return TargetTriple.isOSGlibc(); }
  bool isTargetAndroid() const { return TargetTriple.isAndroid(); }
  bool isTargetNaCl() const { return TargetTriple.isOSNaCl(); }
  bool isTargetNaCl32() const { return isTargetNaCl() && !is64Bit(); }
  bool isTargetNaCl64() const { return isTargetNaCl() && is64Bit(); }
  bool isTargetMCU() const { return TargetTriple.isOSIAMCU(); }
  bool isTargetFuchsia() const { return TargetTriple.isOSFuchsia(); }

  bool isTargetWindowsMSVC() const {
    return TargetTriple.isWindowsMSVCEnvironment();
  }

  bool isTargetKnownWindowsMSVC() const {
    return TargetTriple.isKnownWindowsMSVCEnvironment();
  }

  bool isTargetWindowsCoreCLR() const {
    return TargetTriple.isWindowsCoreCLREnvironment();
  }

  bool isTargetWindowsCygwin() const {
    return TargetTriple.isWindowsCygwinEnvironment();
  }

  bool isTargetWindowsGNU() const {
    return TargetTriple.isWindowsGNUEnvironment();
  }

  bool isTargetWindowsItanium() const {
    return TargetTriple.isWindowsItaniumEnvironment();
  }

  bool isTargetCygMing() const { return TargetTriple.isOSCygMing(); }

  bool isOSWindows() const { return TargetTriple.isOSWindows(); }

  bool isTargetWin64() const {
    return In64BitMode && TargetTriple.isOSWindows();
  }

  bool isTargetWin32() const {
    return !In64BitMode && (isTargetCygMing() || isTargetKnownWindowsMSVC());
  }

  bool isPICStyleGOT() const { return PICStyle == PICStyles::GOT; }
  bool isPICStyleRIPRel() const { return PICStyle == PICStyles::RIPRel; }

  bool isPICStyleStubPIC() const {
    return PICStyle == PICStyles::StubPIC;
  }

  bool isPositionIndependent() const { return TM.isPositionIndependent(); }

  bool isCallingConvWin64(CallingConv::ID CC) const {
    switch (CC) {
    // On Win64, all these conventions just use the default convention.
    case CallingConv::C:
    case CallingConv::Fast:
    case CallingConv::X86_FastCall:
    case CallingConv::X86_StdCall:
    case CallingConv::X86_ThisCall:
    case CallingConv::X86_VectorCall:
    case CallingConv::Intel_OCL_BI:
      return isTargetWin64();
    // This convention allows using the Win64 convention on other targets.
    case CallingConv::X86_64_Win64:
      return true;
    // This convention allows using the SysV convention on Windows targets.
    case CallingConv::X86_64_SysV:
      return false;
    // Otherwise, who knows what this is.
    default:
      return false;
    }
  }

  /// Classify a global variable reference for the current subtarget according
  /// to how we should reference it in a non-pcrel context.
  unsigned char classifyLocalReference(const GlobalValue *GV) const;

  unsigned char classifyGlobalReference(const GlobalValue *GV,
                                        const Module &M) const;
  unsigned char classifyGlobalReference(const GlobalValue *GV) const;

  /// Classify a global function reference for the current subtarget.
  unsigned char classifyGlobalFunctionReference(const GlobalValue *GV,
                                                const Module &M) const;
  unsigned char classifyGlobalFunctionReference(const GlobalValue *GV) const;

  /// Classify a blockaddress reference for the current subtarget according to
  /// how we should reference it in a non-pcrel context.
  unsigned char classifyBlockAddressReference() const;

  /// Return true if the subtarget allows calls to immediate address.
  bool isLegalToCallImmediateAddr() const;

  /// This function returns the name of a function which has an interface
  /// like the non-standard bzero function, if such a function exists on
  /// the current subtarget and it is considered prefereable over
  /// memset with zero passed as the second argument. Otherwise it
  /// returns null.
  const char *getBZeroEntry() const;

  /// This function returns true if the target has sincos() routine in its
  /// compiler runtime or math libraries.
  bool hasSinCos() const;

  /// Enable the MachineScheduler pass for all X86 subtargets.
  bool enableMachineScheduler() const override { return true; }

  bool enableEarlyIfConversion() const override;

  /// Return the instruction itineraries based on the subtarget selection.
  const InstrItineraryData *getInstrItineraryData() const override {
    return &InstrItins;
  }

  AntiDepBreakMode getAntiDepBreakMode() const override {
    return TargetSubtargetInfo::ANTIDEP_CRITICAL;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_X86_X86SUBTARGET_H
