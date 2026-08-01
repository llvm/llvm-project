//===-------------------- HexagonXQFloatGenerator.cpp --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass enables generation of XQFloat instructions. XQF instructions
// are more efficient, but can be less precise in comparison to IEEE ones.
// Based on the accuracy preservation of the generated code, we enabled four
// modes - Strict IEEE-754 compliant, IEEE-754 compliant, Lossy subnormals and
// legacy mode.
//
// Strict IEEE mode adheres to similar accuracy and precision as of IEEE-754.
//
// IEEE-754 compliant mode excludes IEEE-754 overflows and lower precision
// subnormals due to larger dynamic range than IEEE-754.
// All subnormals have extra precision.
//
// Lossy subnormals mode without normalization result in a loss of accuracy.
// This provides greater precision than a clamp of subnormals to 0.
// If dataset excludes subnormals, it behavas as IEEE-754 compliant mode.
//
// The direct mode has a loss of 1 bit of accuracy compared to IEEE-754.
//
// V79 replaces the prior internal HVX floating point format for floating-point
// arithmetic. The new internal HVX floating-point format yields results
// identical to IEEE-754 round-to-even mode. The new format contains more bits
// than IEEE-754, which optionally produces results with greater range and
// accuracy. Only the HVX vector registers use the HVX floating-point format.
// Memory maintains all floating-point data in IEEE-754 format,
// and all loads/stores use the IEEE-754 format. A subset of HVX floating-point
// operations transform IEEE-754 floating-point data to HVX floating-point data.
// Subsequent HVX floating-point instructions may consume operands in the HVX
// floating-point without conversion to IEEE-754, which allows for performant
// & energy efficient code. The program does not need to switch between formats
// continuously. The program must convert the HVX floating-point results to
// IEEE-754 prior to storing to memory.

// HVX floating-point achieves IEEE-754 compliance through normalization.
// The program may skip normalization when faster calculation is desired, and
// IEEE-754 compliance isn’t required. HVX floating-point contains two input
// types: qf32, single precision floating point, and qf16, half precision
// floating point. In Hexagon, IEEE-754 contains two input types: sf, single
// precision floating point, and hf, half precision floating point.
//
// Only HVX floating-point source and destination instructions use HVX
// floating-point values. Instructions specify the HVX floating-point format
// with the qf16 and qf32 identifier. A source vector register will drop the
// extended state of a HVX floating-point value when an instruction reads the
// source vector register without the qf16 or qf32 identifier. A destination
// vector register will reset its extended state when an instruction writes to
// a vector register without the qf16 or qf32 identifier. When dropping the
// extended state, the floating-point value loses accuracy. The program may
// preserve the floating-point value by converting HVX floating-point values
// to IEEE-754 values. Compiler must convert HVX floating-point values to
// IEEE-754 values before using as an input to stores, permutes, shifts, and
// any other operations that do not source the HVX floating-point format.
//
// Depending on the desired results, HVX floating-point operations may have
// some requirements on the input sources. The HVX floating-point values
// require normalization to achieve IEEE-754 compliance, while faster operations
// may skip normalization. The program normalizes HVX floating-point values
// before subsequent HVX floating-point operations, so the floating-point value
// does not lose precision. The program also obtains results identical to
// IEEE-754 by converting all HVX floating-point results to IEEE-754 format
// before consumed in any subsequent operation. There are however cases where
// this conversion is redundant, or the differences between IEEE-754 and HVX
// floating-point may not be a concern.
//
// The conversion logic can be understood by the table below:
//
// ================================================================================================================================================
//            |                              | |                               |
//            |    Inputs to add/subtarct    |                  Inputs to
//            multiplication instuctions              |     Non-HVX floating
//            point    | |    instructions              | |          instruction
//            | |                              | | |
// ===============================================================================================================================================|
// Sources    | IEEE-  | HVX      | HVX      | sf        | qf32      | qf32 | hf
// | qf16     | qf16     | IEE-754 | HVX      | HVX      |
//            |  754   | floating | floating |           | from      | from | |
//            from     | from     |         | floating | floating | |        |
//            point    | point    |           | mult      | adder     | | mult
//            | adder    |         | point    | point    | |        | from     |
//            from     |           |           |           |          | | | |
//            from     | from     | |        | multi    | adder    |           |
//            |           |          |          |          |         | mult |
//            adder    | |        |          |          |           | | | | | |
//            |          |          |
// ===============================================================================================================================================|
// Strict     | Direct | Convert  | Convert  | Normalize | Convert   | Convert
// | widening | Convert  | Convert  | Direct  | Convert  | Convert  | IEEE-754
// | Use    | to       | to       |           | to IEEE   | to IEEE   | multiply
// | to IEEE, | to IEEE, | use     | to       | to       | compliance |        |
// IEEE     | IEEE     |           | then      | then      | then     | widening
// | widening |         | IEEE     | IEEE     |
//            |        |          |          |           | normalize | normalize
//            | convert  | multiply,| multiply,|         |          |          |
//            |        |          |          |           |           | | to IEEE
//            | convert  | convert  |         |          |          | |        |
//            |          |           |           |           |          | to
//            IEEE  | to IEEE  |         |          |          |
// -----------------------------------------------------------------------------------------------------------------------------------------------|
// IEEE-754   | Direct | Direct   | Direct   | Normalize | Direct    | Normalize
// | Widening | Direct   | Widening | Direct  | Convert  | Convert  | compliance
// | Use    | Use      | Use      |           | use       |           | multiply
// | use      | multiply | use     | to IEEE  | to IEEE  |
// -----------------------------------------------------------------------------------------------------------------------------------------------|
// Lossy      | Direct | Direct   | Direct   | Direct    | Direct    | Normalize
// | Direct   | Direct   | Widening | Direct  | Convert  | Convert  | Subnormals
// | Use    | Use      | Use      | Use       | use       |           | use |
// use      | multiply | use     | to IEEE  | to IEEE  |
// -----------------------------------------------------------------------------------------------------------------------------------------------|
// Direct     | Direct | Direct   | Direct   | Direct    | Direct    | Direct |
// Direct   | Direct   | Direct   | Direct  | Direct   | Direct   | Lossy      |
// Use    | Use      | Use      | Use       | use       | use       | use      |
// use      | use      | use     | use      | use      |
// -----------------------------------------------------------------------------------------------------------------------------------------------|
//
// For v81, the normalization sequence changes. Instead of multiplying 0
// and -0, a simple copy operation normalizes the unnormal value. Both
// qf and IEEE-754 value can be unnormal.
// Additionally for v81, we have two new vsub instructions which are handled.

#define HEXAGON_XQFLOAT_GENERATOR "XQFloat Generator pass"

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "HexagonTargetMachine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

#define DEBUG_TYPE "hexagon-xqf-gen"

using namespace llvm;

extern cl::opt<QFloatMode> QFloatModeValue;

// Master flag to enable XQF generations
cl::opt<bool> EnableHVXXQFloat("enable-xqf-gen", cl::init(false),
                               cl::desc("Enable XQFloat generations"));
// Master flag to remove extraneous qf to sf/hf conversions
cl::opt<bool>
    EnableConversionsRemoval("enable-rem-conv", cl::init(false),
                             cl::desc("Enable extraneous conversions removal"));

// Diagnostic flags
cl::opt<bool> PrintDebug("debug-print", cl::init(false),
                         cl::desc("Print function mir after transformation"));
// This vector contains the opcodes which generate qf32 from add/subtract
static constexpr unsigned XQFPAdd32[] = {
    // vector add instructions
    Hexagon::V6_vadd_sf, Hexagon::V6_vadd_qf32, Hexagon::V6_vadd_qf32_mix,

    // vector subtract instructions
    Hexagon::V6_vsub_qf32, Hexagon::V6_vsub_qf32_mix, Hexagon::V6_vsub_sf,
    Hexagon::V6_vsub_sf_mix};

// This vector contains the opcodes which generate qf16 from add/subtract
static constexpr unsigned XQFPAdd16[] = {
    // vector add instructions
    Hexagon::V6_vadd_hf, Hexagon::V6_vadd_qf16, Hexagon::V6_vadd_qf16_mix,

    // vector subtract intrutions
    Hexagon::V6_vsub_hf, Hexagon::V6_vsub_qf16, Hexagon::V6_vsub_qf16_mix,
    Hexagon::V6_vsub_hf_mix};

// This vector contains the opcodes which generate qf32 from multiplication
static constexpr unsigned XQFPMult32[] = {
    Hexagon::V6_vmpy_qf32, Hexagon::V6_vmpy_qf32_qf16, Hexagon::V6_vmpy_qf32_hf,
    Hexagon::V6_vmpy_qf32_sf, Hexagon::V6_vmpy_qf32_mix_hf};
// This vector contains the opcodes which generate qf16 from multiplication
static constexpr unsigned XQFPMult16[] = {Hexagon::V6_vmpy_qf16,
                                          Hexagon::V6_vmpy_qf16_hf,
                                          Hexagon::V6_vmpy_qf16_mix_hf};

namespace llvm {
FunctionPass *createHexagonXQFloatGenerator();
void initializeHexagonXQFloatGeneratorPass(PassRegistry &);
} // namespace llvm

namespace {

struct HexagonXQFloatGenerator : public MachineFunctionPass {
public:
  static char ID;
  HexagonXQFloatGenerator() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return HEXAGON_XQFLOAT_GENERATOR; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // Handle each XQF optimization level
  bool HandleStrictIEEE(MachineFunction &);
  bool HandleCompliantIEEE(MachineFunction &);
  bool HandleLossySubnormals(MachineFunction &);
  bool HandleLossyLegacy(MachineFunction &);

  // Checkers functions for input operands
  bool checkIfInputFromAdder32(Register Reg);
  bool checkIfInputFromAdder16(Register Reg);
  bool checkIfInputFromMult32(Register Reg);
  bool checkIfInputFromMult16(Register Reg);
  bool deleteList();

  // Helper functions for conversion/normalization/widening
  bool widenMultiplicationInputF16(MachineInstr &, Register &, Register &,
                                   Register &, bool);
  bool widenMultiplicationInputF16Rt(MachineInstr &, Register &, Register &,
                                     Register &);
  void widenMultiplyInputHF(MachineInstr &, Register &, Register &, Register &);
  bool normalizeMultiplicationInputF32(MachineInstr &, Register &, Register &,
                                       Register &, Register &, bool &);
  void normalizeMultiplicationInputSF(MachineInstr &, Register &, Register &,
                                      Register &, Register &, bool &);
  bool convertNormalizeMultOp32(MachineInstr &, Register &, Register &,
                                Register &, Register &, bool &);
  bool convertWidenMultOp16(MachineInstr &, Register &, Register &, Register &,
                            bool);
  bool convertWidenMultOp32(MachineInstr &, Register &, Register &, Register &,
                            bool);
  void createPrologInstructions(MachineInstr &, Register &);
  bool convertAddOpToIEEE16(MachineInstr &, Register &, Register &, Register &,
                            bool, bool, bool);
  bool convertAddOpToIEEE32(MachineInstr &, Register &, Register &, Register &,
                            bool, bool, bool);
  void generateQF16FromQF32(MachineInstr &, Register &, Register &);
  bool convertIfInputToNonHVX(MachineInstr &, bool);
  void createConvertInstr(MachineInstr *, Register &, Register &, bool);

  // V81 specific normalization function
  bool V81normalizeMultF32(MachineInstr &, Register &, Register &, Register &,
                           bool, bool, bool);

  const HexagonSubtarget *HST = nullptr;
  const HexagonInstrInfo *HII = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  SmallVector<MachineInstr *, 16>
      OriginalMI; // Hold the instructions to be deleted
};

// Print machine function
static void debug_print([[maybe_unused]] MachineFunction &MF) {
  dbgs() << "\n=== Printing function ===\n";
#ifndef NDEBUG
  for (MachineBasicBlock &MBB : MF)
    MBB.dump();
#endif // NDEBUG
}

// This class removes redundant vector convert instructions from qf to hf/sf.
// Additionally, it relaces use of sf/hf registers with qf types.
// The resulting code is complete without dangling instructions.
// FIXME: Liveness is not preserved.
class VectorConvertRemove {

public:
  VectorConvertRemove(MachineFunction &_MF, MachineRegisterInfo *_MRI,
                      const HexagonSubtarget *_HST)
      : MF(_MF), MRI(_MRI), HST(_HST) {
    HII = HST->getInstrInfo();
  }

  void run();

private:
  MachineFunction &MF;
  MachineRegisterInfo *MRI;
  const HexagonSubtarget *HST;
  const HexagonInstrInfo *HII;

  enum Operation { Add16, Add32, Sub16, Sub32, Mul16, Mul32 };
  // Helper functions
  void handle_addsub_sf_sf(MachineInstr &, Register &, Register &, Register &,
                           bool);
  void handle_addsub_qf_sf(MachineInstr &, Register &, Register &, Register &,
                           bool);
  void handle_addsubmul_hf_hf(MachineInstr &, Register &, Register &,
                              Register &, Operation);
  void handle_addsubmul_qf_hf(MachineInstr &, Register &, Register &,
                              Register &, Operation);
  void handle_qf32_mul_sf_sf(MachineInstr &, Register &, Register &,
                             Register &);
  bool checkHVXUses32(MachineInstr *, MachineInstr *);
  bool checkHVXUses16(MachineInstr *, MachineInstr *);
  unsigned getOperation(Operation, bool, bool);

  // List which holds conversion instructions
  SmallPtrSet<MachineInstr *, 16> ConvInstrList;
  // List which holds qf handling instructions
  std::vector<MachineInstr *> SfHfInstrList;
};

// both : both operands are replaced
unsigned VectorConvertRemove::getOperation(Operation Op, bool firstOpQf,
                                           bool secOpQf) {
  if (firstOpQf && secOpQf) {
    switch (Op) {
    case Add16:
      return Hexagon::V6_vadd_qf16;
    case Add32:
      return Hexagon::V6_vadd_qf32;
    case Sub16:
      return Hexagon::V6_vsub_qf16;
    case Sub32:
      return Hexagon::V6_vsub_qf32;
    case Mul16:
      return Hexagon::V6_vmpy_qf16;
    case Mul32:
      return Hexagon::V6_vmpy_qf32_qf16;
    }
  } else if (firstOpQf) {
    switch (Op) {
    case Add16:
      return Hexagon::V6_vadd_qf16_mix;
    case Add32:
      return Hexagon::V6_vadd_qf32_mix;
    case Sub16:
      return Hexagon::V6_vsub_qf16_mix;
    case Sub32:
      return Hexagon::V6_vsub_qf32_mix;
    case Mul16:
      return Hexagon::V6_vmpy_qf16_mix_hf;
    case Mul32:
      return Hexagon::V6_vmpy_qf32_mix_hf;
    }
  } else if (secOpQf) {
    switch (Op) {
    case Sub16:
      return Hexagon::V6_vsub_hf_mix;
    case Sub32:
      return Hexagon::V6_vsub_sf_mix;
    default:
      break;
    }
  } else {
  }
  llvm_unreachable("Unknown opcode and operand combination!");
}

// Return false if there are multiple instructions where the qf32 is used
// other than the instruction for which it is called
bool VectorConvertRemove::checkHVXUses32(MachineInstr *MI,
                                         MachineInstr *UseMI) {
  Register convReg = MI->getOperand(0).getReg();
  // Iterate over all uses of the Def we are analyzing
  for (auto &MO : make_range(MRI->use_begin(convReg), MRI->use_end())) {
    MachineInstr *UMI = MO.getParent();
    if (UMI == UseMI)
      continue;
    // Since the convert cannot be deleted, we set the operand as NOT kill
    MI->getOperand(1).setIsKill(false);
    return false;
  }
  return true;
}

// Return false if there are multiple instructions where the qf16 is used
// other than the instruction for which it is called
bool VectorConvertRemove::checkHVXUses16(MachineInstr *MI,
                                         MachineInstr *UseMI) {
  Register convReg = MI->getOperand(0).getReg();
  // Iterate over all uses of the Def we are analyzing
  for (auto &MO : make_range(MRI->use_begin(convReg), MRI->use_end())) {
    MachineInstr *UMI = MO.getParent();
    if (UMI == UseMI)
      continue;
    // Since the convert cannot be deleted, we set the operand as NOT kill
    MI->getOperand(1).setIsKill(false);
    return false;
  }
  return true;
}

// Removes converts feeding to op(sf,sf), and replaces its sf operands with qf
void VectorConvertRemove::handle_addsub_sf_sf(MachineInstr &MI, Register &Reg1,
                                              Register &Reg2, Register &Dest,
                                              bool isAdd) {

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  bool firstConv = false, secConv = false;
  bool DefOp1_del = false, DefOp2_del = false;
  Register Src1, Src2;

  MachineInstr *DefOp1 = MRI->getVRegDef(Reg1);
  MachineInstr *DefOp2 = MRI->getVRegDef(Reg2);
  // check if the first operand is from a convert operation
  if (DefOp1->getOpcode() == Hexagon::V6_vconv_sf_qf32) {
    if (checkHVXUses32(DefOp1, &MI))
      DefOp1_del = true;
    Src1 = DefOp1->getOperand(1).getReg();
    firstConv = true;
  }

  // check if the second operand is from a convert operation
  if (DefOp2->getOpcode() == Hexagon::V6_vconv_sf_qf32) {
    if (checkHVXUses32(DefOp2, &MI))
      DefOp2_del = true;
    Src2 = DefOp2->getOperand(1).getReg();
    secConv = true;
  }

  if (firstConv && secConv) {
    BuildMI(MBB, MI, DL,
            HII->get(getOperation(isAdd ? Operation::Add32 : Operation::Sub32,
                                  true, true)),
            Dest)
        .addReg(Src1)
        .addReg(Src2);
    SfHfInstrList.push_back(&MI);
  } else if (firstConv) {
    BuildMI(MBB, MI, DL,
            HII->get(getOperation(isAdd ? Operation::Add32 : Operation::Sub32,
                                  true, false)),
            Dest)
        .addReg(Src1)
        .addReg(Reg2);
    SfHfInstrList.push_back(&MI);
  } else if (secConv) {
    // For v79, there is no provision for 2nd op being qf for add/sub
    if (HST->useHVXV81Ops()) {
      if (isAdd)
        BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), Dest)
            .addReg(Src2)
            .addReg(Reg1);
      else
        BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vsub_sf_mix), Dest)
            .addReg(Reg1)
            .addReg(Src2);
      SfHfInstrList.push_back(&MI);
      // For v79, there is no provision for 2nd op being qf for add/sub. Since
      // add is commutative, the ops can be rotated.
    } else if (HST->useHVXV79Ops()) {
      // for vadd we interchange the ops, for vsub we ignore
      if (isAdd) {
        BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), Dest)
            .addReg(Src2)
            .addReg(Reg1);
        SfHfInstrList.push_back(&MI);
      } else // don't delete the convert instruction for vsub
        DefOp2_del = false;
    }
  }

  if (DefOp1_del)
    ConvInstrList.insert(DefOp1);
  if (DefOp2_del)
    ConvInstrList.insert(DefOp2);
}

// Removes converts feeding to op(hf,hf), and replaces its hf operands with qf
void VectorConvertRemove::handle_addsubmul_hf_hf(MachineInstr &MI,
                                                 Register &Reg1, Register &Reg2,
                                                 Register &Dest, Operation Op) {

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  bool firstConv = false, secConv = false;
  bool DefOp1_del = false, DefOp2_del = false;
  bool isSub = Op == Operation::Sub16;
  Register Src1, Src2;

  MachineInstr *DefOp1 = MRI->getVRegDef(Reg1);
  MachineInstr *DefOp2 = MRI->getVRegDef(Reg2);
  // check if the first operand is from a convert operation
  if (DefOp1->getOpcode() == Hexagon::V6_vconv_hf_qf16) {
    if (checkHVXUses16(DefOp1, &MI))
      DefOp1_del = true;
    Src1 = DefOp1->getOperand(1).getReg();
    firstConv = true;
  }

  // check if the second operand is from a convert operation
  if (DefOp2->getOpcode() == Hexagon::V6_vconv_hf_qf16) {
    if (checkHVXUses16(DefOp2, &MI))
      DefOp2_del = true;
    Src2 = DefOp2->getOperand(1).getReg();
    secConv = true;
  }

  if (firstConv && secConv) {
    BuildMI(MBB, MI, DL, HII->get(getOperation(Op, true, true)), Dest)
        .addReg(Src1)
        .addReg(Src2);
    SfHfInstrList.push_back(&MI);
  } else if (firstConv) {
    BuildMI(MBB, MI, DL, HII->get(getOperation(Op, true, false)), Dest)
        .addReg(Src1)
        .addReg(Reg2);
    SfHfInstrList.push_back(&MI);
  } else if (secConv) {
    // For v81, we interchange the ops for vadd/vmul
    // for vsub we use qf as second operand
    if (HST->useHVXV81Ops()) {
      if (!isSub)
        BuildMI(MBB, MI, DL, HII->get(getOperation(Op, true, false)), Dest)
            .addReg(Src2)
            .addReg(Reg1);
      else
        BuildMI(MBB, MI, DL, HII->get(getOperation(Op, false, true)), Dest)
            .addReg(Reg1)
            .addReg(Src2);
      SfHfInstrList.push_back(&MI);
    } else if (HST->useHVXV79Ops()) {
      // for vadd/vmul we interchange the ops, for vsub we ignore
      if (!isSub) {
        BuildMI(MBB, MI, DL, HII->get(getOperation(Op, true, false)), Dest)
            .addReg(Src2)
            .addReg(Reg1);
        SfHfInstrList.push_back(&MI);
      } else // don't delete the convert instruction for vsub
        DefOp2_del = false;
    }
  }

  if (DefOp1_del)
    ConvInstrList.insert(DefOp1);
  if (DefOp2_del)
    ConvInstrList.insert(DefOp2);
}

// Removes converts feeding to op(qf,sf), and replaces its sf operands with qf
void VectorConvertRemove::handle_addsub_qf_sf(MachineInstr &MI, Register &Reg1,
                                              Register &Reg2, Register &Dest,
                                              bool isAdd) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  Register Src;
  bool conv = false;

  MachineInstr *DefOp = MRI->getVRegDef(Reg2);
  // check if the second operand is from a convert operation
  if (DefOp->getOpcode() == Hexagon::V6_vconv_sf_qf32) {
    if (checkHVXUses32(DefOp, &MI))
      ConvInstrList.insert(DefOp);
    Src = DefOp->getOperand(1).getReg();
    conv = true;
  }

  if (conv) {
    BuildMI(MBB, MI, DL,
            HII->get(isAdd ? Hexagon::V6_vadd_qf32 : Hexagon::V6_vsub_qf32),
            Dest)
        .addReg(Reg1)
        .addReg(Src);
    SfHfInstrList.push_back(&MI);
  }
}

// Removes converts feeding to op(qf,hf), and replaces its hf operands with qf
void VectorConvertRemove::handle_addsubmul_qf_hf(MachineInstr &MI,
                                                 Register &Reg1, Register &Reg2,
                                                 Register &Dest, Operation Op) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  Register Src;
  bool conv = false;

  MachineInstr *DefOp = MRI->getVRegDef(Reg2);
  // check if the second operand is from a convert operation
  if (DefOp->getOpcode() == Hexagon::V6_vconv_hf_qf16) {
    if (checkHVXUses16(DefOp, &MI))
      ConvInstrList.insert(DefOp);
    Src = DefOp->getOperand(1).getReg();
    conv = true;
  }

  if (conv) {
    BuildMI(MBB, MI, DL, HII->get(getOperation(Op, true, true)), Dest)
        .addReg(Reg1)
        .addReg(Src);
    SfHfInstrList.push_back(&MI);
  }
}

// Removes converts feeding to op(sf,sf), and replaces its sf operands with qf
void VectorConvertRemove::handle_qf32_mul_sf_sf(MachineInstr &MI,
                                                Register &Reg1, Register &Reg2,
                                                Register &Dest) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  Register Src1, Src2;
  bool firstConv = false, secConv = false;

  MachineInstr *DefOp1 = MRI->getVRegDef(Reg1);
  MachineInstr *DefOp2 = MRI->getVRegDef(Reg2);

  if (DefOp1->getOpcode() == Hexagon::V6_vconv_sf_qf32 &&
      DefOp2->getOpcode() == Hexagon::V6_vconv_sf_qf32) {
    // If yes, we can remove the convert
    if (checkHVXUses32(DefOp1, &MI) && checkHVXUses32(DefOp2, &MI)) {
      ConvInstrList.insert(DefOp1);
      ConvInstrList.insert(DefOp2);
    }
    Src1 = DefOp1->getOperand(1).getReg();
    Src2 = DefOp2->getOperand(1).getReg();
    firstConv = true;
    secConv = true;
  }

  // If both are true, then only replace with qf32 = vmpy(qf32, qf32)
  if (firstConv && secConv) {
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(Src1)
        .addReg(Src2);
    SfHfInstrList.push_back(&MI);
  }
}

void VectorConvertRemove::run() {
  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      // Skip if the instruction does not have two operands,
      // or is a bundle instruction
      // or is a debug instruction
      if (MI.getNumOperands() != 3 || MI.isDebugInstr())
        continue;

      auto Op1 = MI.getOperand(1);
      if (!Op1.isReg())
        continue;
      auto Op2 = MI.getOperand(2);
      if (!Op2.isReg())
        continue;
      auto Op0 = MI.getOperand(0);
      if (!Op0.isReg())
        continue;
      Register Reg1 = Op1.getReg();
      Register Reg2 = Op2.getReg();
      Register Dest = Op0.getReg();

      switch (MI.getOpcode()) {
      // TODO Handle the new vsub instructions
      // qf32 = vadd(sf, sf)
      case Hexagon::V6_vadd_sf:
        handle_addsub_sf_sf(MI, Reg1, Reg2, Dest, true);
        break;
      // qf32 = vsub(sf, sf)
      case Hexagon::V6_vsub_sf:
        handle_addsub_sf_sf(MI, Reg1, Reg2, Dest, false);
        break;
      // qf32 = vadd(qf32, sf)
      case Hexagon::V6_vadd_qf32_mix:
        handle_addsub_qf_sf(MI, Reg1, Reg2, Dest, true);
        break;
      // qf32 = vsub(qf32, sf)
      case Hexagon::V6_vsub_qf32_mix:
        handle_addsub_qf_sf(MI, Reg1, Reg2, Dest, false);
        break;
      // qf16 = vadd(hf, hf)
      case Hexagon::V6_vadd_hf:
        handle_addsubmul_hf_hf(MI, Reg1, Reg2, Dest, Operation::Add16);
        break;
      // qf16 = vsub(hf, hf)
      case Hexagon::V6_vsub_hf:
        handle_addsubmul_hf_hf(MI, Reg1, Reg2, Dest, Operation::Sub16);
        break;
      // qf16 = vadd(qf16, hf)
      case Hexagon::V6_vadd_qf16_mix:
        handle_addsubmul_qf_hf(MI, Reg1, Reg2, Dest, Operation::Add16);
        break;
      // qf16 = vsub(qf16, hf)
      case Hexagon::V6_vsub_qf16_mix:
        handle_addsubmul_qf_hf(MI, Reg1, Reg2, Dest, Operation::Sub16);
        break;
      // qf32 = vmpy(sf, sf)
      case Hexagon::V6_vmpy_qf32_sf:
        handle_qf32_mul_sf_sf(MI, Reg1, Reg2, Dest);
        break;
      // qf32 = vmpy(hf, hf)
      case Hexagon::V6_vmpy_qf32_hf:
        handle_addsubmul_hf_hf(MI, Reg1, Reg2, Dest, Operation::Mul32);
        break;
      // qf32 = vmpy(qf16, hf)
      case Hexagon::V6_vmpy_qf32_mix_hf:
        handle_addsubmul_qf_hf(MI, Reg1, Reg2, Dest, Operation::Mul32);
        break;
      // qf16 = vmpy(hf, hf)
      case Hexagon::V6_vmpy_qf16_hf:
        handle_addsubmul_hf_hf(MI, Reg1, Reg2, Dest, Operation::Mul16);
        break;
      // qf16 = vmpy(qf16, hf)
      case Hexagon::V6_vmpy_qf16_mix_hf:
        handle_addsubmul_qf_hf(MI, Reg1, Reg2, Dest, Operation::Mul16);
        break;
      default:
        break;
      }
    }
  }

  // Delete the vadd/vsub/vmpy instructions
  for (MachineInstr *sfhfMI : SfHfInstrList) {
    LLVM_DEBUG(dbgs() << "deleting sf/hf instruction ");
    LLVM_DEBUG(sfhfMI->dump());
    sfhfMI->eraseFromParent();
  }
  // Delete conversion instructions
  for (MachineInstr *convMI : ConvInstrList) {
    LLVM_DEBUG(dbgs() << "deleting conversion instruction");
    LLVM_DEBUG(convMI->dump());
    convMI->eraseFromParent();
  }
}

char HexagonXQFloatGenerator::ID = 0;

} // namespace

INITIALIZE_PASS(HexagonXQFloatGenerator, "hexagon-xqfloat-generator",
                HEXAGON_XQFLOAT_GENERATOR, false, false)

FunctionPass *llvm::createHexagonXQFloatGenerator() {
  return new HexagonXQFloatGenerator();
}

// Returns true if qf32 input is from an adder/subtract unit
bool HexagonXQFloatGenerator::checkIfInputFromAdder32(Register Reg) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def)
    return false;

  // If the definition is a copy, we need to analyze its def again
  if (Def->getOpcode() == TargetOpcode::COPY) {
    Register SrcReg = Def->getOperand(1).getReg();
    if (SrcReg.isValid())
      return checkIfInputFromAdder32(SrcReg);
    return false;
  } else if (Def->getOpcode() == TargetOpcode::REG_SEQUENCE) {
    Register SrcReg1 = Def->getOperand(1).getReg();
    Register SrcReg2 = Def->getOperand(2).getReg();
    bool isTrue = false;
    if (SrcReg1.isValid())
      isTrue = checkIfInputFromAdder32(SrcReg1);
    if (SrcReg2.isValid())
      isTrue |= checkIfInputFromAdder32(SrcReg2);
    return isTrue;
  } else
    return llvm::is_contained(XQFPAdd32, Def->getOpcode());
}

// Returns true if qf16 input is from an adder/subtract unit
bool HexagonXQFloatGenerator::checkIfInputFromAdder16(Register Reg) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def)
    return false;

  // if the definition is a copy, we need to analyze its def again
  if (Def->getOpcode() == TargetOpcode::COPY) {
    Register SrcReg = Def->getOperand(1).getReg();
    if (SrcReg.isValid())
      return checkIfInputFromAdder16(SrcReg);
    return false;
  } else
    return llvm::is_contained(XQFPAdd16, Def->getOpcode());
}

// Returns true if qf32 input is from a multiplier unit
bool HexagonXQFloatGenerator::checkIfInputFromMult32(Register Reg) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def)
    return false;

  // if the definition is a copy, we need to analyze its def again
  if (Def->getOpcode() == TargetOpcode::COPY) {
    Register SrcReg = Def->getOperand(1).getReg();
    if (SrcReg.isValid())
      return checkIfInputFromMult32(SrcReg);
    return false;
  } else if (Def->getOpcode() == TargetOpcode::REG_SEQUENCE) {
    Register SrcReg1 = Def->getOperand(1).getReg();
    Register SrcReg2 = Def->getOperand(2).getReg();
    bool isTrue = false;
    if (SrcReg1.isValid())
      isTrue |= checkIfInputFromMult32(SrcReg1);
    if (SrcReg2.isValid())
      isTrue |= checkIfInputFromMult32(SrcReg2);
    return isTrue;
  } else
    return llvm::is_contained(XQFPMult32, Def->getOpcode());
}

// Returns true if qf16 input is from a multiplier unit
bool HexagonXQFloatGenerator::checkIfInputFromMult16(Register Reg) {
  MachineInstr *Def = MRI->getVRegDef(Reg);
  if (!Def)
    return false;

  // if the definition is a copy, we need to analyze its def again
  if (Def->getOpcode() == TargetOpcode::COPY) {
    Register SrcReg = Def->getOperand(1).getReg();
    if (SrcReg.isValid())
      return checkIfInputFromMult16(SrcReg);
    return false;
  } else
    return llvm::is_contained(XQFPMult16, Def->getOpcode());
}

// Generates sf = qf32 instruction or hf = qf16 intruction
void HexagonXQFloatGenerator::createConvertInstr(MachineInstr *UseMI,
                                                 Register &NewR, Register &OldR,
                                                 bool is32bit) {
  const DebugLoc &DL = UseMI->getDebugLoc();
  MachineBasicBlock *MBB = UseMI->getParent();
  NewR = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  if (is32bit)
    BuildMI(*MBB, *UseMI, DL, HII->get(Hexagon::V6_vconv_sf_qf32), NewR)
        .addReg(OldR);
  else
    BuildMI(*MBB, *UseMI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), NewR)
        .addReg(OldR);
}

// Generate HVX to IEEE conversion instruction for all non-HVX uses
bool HexagonXQFloatGenerator::convertIfInputToNonHVX(MachineInstr &MI,
                                                     bool is32bit) {
  Register NewR;
  bool Changed = false;
  ;
  Register Dest = MI.getOperand(0).getReg();

  // Iterate over all uses of the Def we are analyzing
  for (auto &MO : make_range(MRI->use_begin(Dest), MRI->use_end())) {
    MachineInstr *UseMI = MO.getParent();
    // Omit if the use is a REG_SEQUENCE instruction, since the only
    // use of REG_SEQUENCE in qf context is transforming to IEEE.
    // Omit for use in DBG instructions.
    // Omit for use in PHI instructions since PHI result can be used as a qf
    // operand.
    if (UseMI->getOpcode() == TargetOpcode::REG_SEQUENCE ||
        UseMI->getOpcode() == TargetOpcode::DBG_VALUE ||
        UseMI->getOpcode() == TargetOpcode::DBG_LABEL ||
        UseMI->getOpcode() == TargetOpcode::PHI)
      continue;

    // If 32-bit operand
    if (is32bit) {
      // If it is a copy instruction, we need to analyze it uses
      if (UseMI->getOpcode() == TargetOpcode::COPY)
        return convertIfInputToNonHVX(*UseMI, /* 32 bit */ true);
      if (!HII->usesQFOperand(UseMI)) {
        createConvertInstr(UseMI, NewR, Dest, /*32 bit*/ true);
        MO.setReg(NewR);
        Changed = true;
      }
      // If 16-bit operand
    } else {
      // If it is a copy instruction, we need to analyze it uses
      if (UseMI->getOpcode() == TargetOpcode::COPY)
        return convertIfInputToNonHVX(*UseMI, /* 16 bit */ false);
      if (!HII->usesQFOperand(UseMI)) {
        createConvertInstr(UseMI, NewR, Dest, /*16 bit*/ false);
        MO.setReg(NewR);
        Changed = true;
      }
    }
  }
  return Changed;
}

// generate qf16 = qf32 via:
// hf = qf32
// V0 = #0
// qf16 = vsub(hf,V0)
void HexagonXQFloatGenerator::generateQF16FromQF32(MachineInstr &MI,
                                                   Register &Dest,
                                                   Register &SrcReg) {

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  Register convertReg = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf32), convertReg)
      .addReg(SrcReg);
  Register VR0 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vd0), VR0);

  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vsub_hf), Dest)
      .addReg(convertReg)
      .addReg(VR0);
}

// Widen qf16 = vmpy(hf, hf) result unconditionally
void HexagonXQFloatGenerator::widenMultiplyInputHF(MachineInstr &MI,
                                                   Register &Reg1,
                                                   Register &Reg2,
                                                   Register &Dest) {
  Register output_mpy = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_hf), output_mpy)
      .addReg(Reg1)
      .addReg(Reg2);
  generateQF16FromQF32(MI, Dest, output_mpy);
}

// Widen vmpy(qf16, qf16/hf) result conditionally
bool HexagonXQFloatGenerator::widenMultiplicationInputF16(MachineInstr &MI,
                                                          Register &Reg1,
                                                          Register &Reg2,
                                                          Register &Dest,
                                                          bool twoOps) {
  bool firstconvert = false, secondconvert = false;
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  // We widen only that operand which comes from add/subtract unit.
  if (checkIfInputFromAdder16(Reg1))
    firstconvert = true;
  // twoOps == true suggest 2nd operand is qf16, else it is hf
  if (twoOps && checkIfInputFromAdder16(Reg2))
    secondconvert = true;

  Register widenReg;
  // if either operands from add/subtract unit, we widen
  if (twoOps) {
    if (firstconvert || secondconvert) {
      widenReg = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_qf16), widenReg)
          .addReg(Reg1)
          .addReg(Reg2);
    } else {
      return false;
    }
  } else {
    if (firstconvert) {
      widenReg = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_mix_hf), widenReg)
          .addReg(Reg1)
          .addReg(Reg2);
    } else {
      return false;
    }
  }

  // generate qf16 = qf32
  generateQF16FromQF32(MI, Dest, widenReg);

  return true;
}

// Handle qf16 = vmpy(qf16, Rt)
// For strict IEEE mode, convert the qf16 to IEEE before widening
bool HexagonXQFloatGenerator::widenMultiplicationInputF16Rt(MachineInstr &MI,
                                                            Register &Reg1,
                                                            Register &Reg2,
                                                            Register &Dest) {
  // If the first input is not from an adder, for strict-ieee check if
  // input from mult, else return false.
  if (!checkIfInputFromAdder16(Reg1)) {
    if (QFloatModeValue == QFloatMode::StrictIEEE) {
      if (!checkIfInputFromMult16(Reg1))
        return false;
    } else
      return false;
  }

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  Register VSplatReg = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_lvsplatw), VSplatReg).addReg(Reg2);

  Register widenReg = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
  if (QFloatModeValue == QFloatMode::StrictIEEE) {
    Register VHf = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VHf).addReg(Reg1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_hf), widenReg)
        .addReg(VHf)
        .addReg(VSplatReg);
  } else {
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_mix_hf), widenReg)
        .addReg(Reg1)
        .addReg(VSplatReg);
  }

  // generate qf16 = qf32
  generateQF16FromQF32(MI, Dest, widenReg);
  return true;
}

// Handle qf32 = vadd/vsub(qf32/sf, qf32/sf)
// Handle vadd/vsub instructions with qf32 operands conditionally
// isAdd:  true if an add instruction is analyzed, false for subtract
// isFirstOpQf: true if 1st operand is qf32 type, false if sf type
// isSecOpQf: true if 2nd operand is qf32 type, false if sf type
bool HexagonXQFloatGenerator::convertAddOpToIEEE32(
    MachineInstr &MI, Register &Reg1, Register &Reg2, Register &Dest,
    bool isAdd, bool isFirstOpQf, bool isSecOpQf) {

  Register VR1;
  Register VR2;
  bool firstconvert = false, secondconvert = false;
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  // If the first operand is qf32 type
  if (isFirstOpQf) {
    // If the first operand is from add/sub/mul unit,
    // generate IEEE conversion instruction sf = qf32
    if (checkIfInputFromAdder32(Reg1) || checkIfInputFromMult32(Reg1)) {
      VR1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32), VR1)
          .addReg(Reg1);
      firstconvert = true;
    }
  }

  // If 2nd operand is of qf32 type
  if (isSecOpQf) {
    // If the second operand is from add/sub/mul unit,
    // generate IEEE conversion instruction
    if (checkIfInputFromAdder32(Reg2) || checkIfInputFromMult32(Reg2)) {
      VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32), VR2)
          .addReg(Reg2);
      secondconvert = true;
    }
  }

  // If both operands are qf32 type, use V6_v[add/sub]_sf instruction
  // If one of them is of sf type, use V6_v[add/sub]_qf32_mix instruction
  // Output is qf32
  if (isFirstOpQf && isSecOpQf) {
    if (firstconvert && secondconvert) {
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_sf : Hexagon::V6_vsub_sf), Dest)
          .addReg(VR1)
          .addReg(VR2);
    } else if (firstconvert) {
      if (isAdd)
        BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), Dest)
            .addReg(Reg2)
            .addReg(VR1);
      // For vsub type, for v81 we use a different opcode,
      // for v79, we convert the 2nd op to IEEE too.
      else {
        if (HST->useHVXV81Ops())
          BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vsub_sf_mix), Dest)
              .addReg(VR1)
              .addReg(Reg2);
        else {
          VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
          BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32), VR2)
              .addReg(Reg2);
          BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vsub_sf), Dest)
              .addReg(VR1)
              .addReg(VR2);
        }
      }
    } else if (secondconvert) {
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_qf32_mix
                             : Hexagon::V6_vsub_qf32_mix),
              Dest)
          .addReg(Reg1)
          .addReg(VR2);
    } else { // none of the inputs is from an add/sub/mul unit
      return false;
    }
    // handle vadd/vsub when the 1st op of original instruction is qf type
  } else if (isFirstOpQf) {
    if (firstconvert)
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_sf : Hexagon::V6_vsub_sf), Dest)
          .addReg(VR1)
          .addReg(Reg2);
    else
      return false;
    // handle vadd/vsub when the 2nd op of original instruction is qf type
  } else if (isSecOpQf) {
    if (secondconvert)
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_sf : Hexagon::V6_vsub_sf), Dest)
          .addReg(Reg1)
          .addReg(VR2);
    else
      return false;
  } else
    return false;
  return true;
}

// Handle qf16 = vadd/vsub(qf16, qf16/hf)
// Handle vadd/vsub instructions with qf16 operands conditionally
// isAdd:  true if an add instruction is analyzed, false for subtract
// isFirstOpQf: true if 1st operand is qf16 type, false if hf type
// isSecOpQf: true if 2nd operand is qf16 type, false if hf type
bool HexagonXQFloatGenerator::convertAddOpToIEEE16(
    MachineInstr &MI, Register &Reg1, Register &Reg2, Register &Dest,
    bool isAdd, bool isFirstOpQf, bool isSecOpQf) {

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  Register VR1;
  Register VR2;
  bool firstconvert = false, secondconvert = false;

  // If the first qf16 operand is from add/sub/mul unit,
  // generate IEEE conversion instruction
  if (isFirstOpQf) {
    if (checkIfInputFromAdder16(Reg1) || checkIfInputFromMult16(Reg1)) {
      VR1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR1)
          .addReg(Reg1);
      firstconvert = true;
    }
  }
  if (isSecOpQf) {
    // If the second operand is from add/sub/mul unit,
    // generate IEEE conversion instruction
    if (checkIfInputFromAdder16(Reg2) || checkIfInputFromMult16(Reg2)) {
      VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR2)
          .addReg(Reg2);
      secondconvert = true;
    }
  }

  // If both operands are qf16 type, use V6_v[add/sub]_hf instruction
  // If one of them is of hf type, use V6_v[add/sub]_qf16_mix instruction
  // Output is qf16
  if (isFirstOpQf && isSecOpQf) {
    if (firstconvert && secondconvert) {
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_hf : Hexagon::V6_vsub_hf), Dest)
          .addReg(VR1)
          .addReg(VR2);
    } else if (firstconvert) {
      if (isAdd)
        BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf16_mix), Dest)
            .addReg(Reg2)
            .addReg(VR1);
      // For vsub type, for v81 we use a different opcode,
      // for v79, we convert the 2nd op to IEEE too.
      else {
        if (HST->useHVXV81Ops())
          BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vsub_hf_mix), Dest)
              .addReg(VR1)
              .addReg(Reg2);
        else {
          VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
          BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR2)
              .addReg(Reg2);
          BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vsub_hf), Dest)
              .addReg(VR1)
              .addReg(VR2);
        }
      }
    } else if (secondconvert) {
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_qf16_mix
                             : Hexagon::V6_vsub_qf16_mix),
              Dest)
          .addReg(Reg1)
          .addReg(VR2);
    } else { // none of the inputs is from an add/sub/mul unit
      return false;
    }
    // handle vadd/vsub when the 1st op of original instruction is qf type
  } else if (isFirstOpQf) {
    if (firstconvert)
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_hf : Hexagon::V6_vsub_hf), Dest)
          .addReg(VR1)
          .addReg(Reg2);
    else
      return false;
    // handle vadd/vsub when the 2nd op of original instruction is qf type
  } else if (isSecOpQf) {
    if (secondconvert)
      BuildMI(MBB, MI, DL,
              HII->get(isAdd ? Hexagon::V6_vadd_hf : Hexagon::V6_vsub_hf), Dest)
          .addReg(Reg1)
          .addReg(VR2);
    else
      return false;
  } else
    return false;
  return true;
}

// Create the prolog
// v0 = #0
// R1 = #0x80000000
// v1.sf = vsplat(R1)
// v2.sf = vmpy(v0.sf, v1.sf)
void HexagonXQFloatGenerator::createPrologInstructions(MachineInstr &MI,
                                                       Register &R_mpy) {

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  Register VR0 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vd0), VR0);

  Register R_0 = MRI->createVirtualRegister(&Hexagon::IntRegsRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::A2_tfrsi), R_0).addImm(0x80000000);

  Register VR_0 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_lvsplatw), VR_0).addReg(R_0);

  R_mpy = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_sf), R_mpy)
      .addReg(VR0)
      .addReg(VR_0);
}

bool HexagonXQFloatGenerator::V81normalizeMultF32(
    MachineInstr &MI, Register &Reg1, Register &Reg2, Register &Dest,
    bool firstconvert, bool secondconvert, bool strictieee) {
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  Register input_mpy1, input_mpy2;

  auto Op =
      strictieee ? Hexagon::V6_vconv_qf32_sf : Hexagon::V6_vconv_qf32_qf32;

  // Normalize both input operands
  if (firstconvert && secondconvert) {
    input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Op), input_mpy1).addReg(Reg1);
    BuildMI(MBB, MI, DL, HII->get(Op), input_mpy2).addReg(Reg2);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(input_mpy2);
  }
  // Normalize only first operand
  else if (firstconvert) {
    input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Op), input_mpy1).addReg(Reg1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(Reg2);
  }
  // Normalize only second operand
  else if (secondconvert) {
    input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Op), input_mpy2).addReg(Reg2);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(Reg1)
        .addReg(input_mpy2);
  } else
    // we do nothing if the inputs are not from adder/sub/mult unit
    return false;

  return true;
}

// Normalize qf32 = vmpy(sf, sf) instruction unconditionally
void HexagonXQFloatGenerator::normalizeMultiplicationInputSF(
    MachineInstr &MI, Register &Src1, Register &Src2, Register &Dest,
    Register &R_mpy, bool &PrologCreated) {

  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  if (HST->useHVXV81Ops()) {
    Register input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    Register input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    // Normalize both inputs
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_qf32_sf), input_mpy1)
        .addReg(Src1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_qf32_sf), input_mpy2)
        .addReg(Src2);
    // Add the new vmpy
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(input_mpy2);
    return;
  }

  if (!PrologCreated) {
    createPrologInstructions(MI, R_mpy);
    PrologCreated = true;
  }

  Register input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  Register input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
  // Normalize both inputs
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), input_mpy1)
      .addReg(R_mpy)
      .addReg(Src1);
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), input_mpy2)
      .addReg(R_mpy)
      .addReg(Src2);
  // Add the new vmpy
  BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
      .addReg(input_mpy1)
      .addReg(input_mpy2);
}

// Convert and normalize qf32 = vmpy(qf32, qf32) instructions conditionally
bool HexagonXQFloatGenerator::convertNormalizeMultOp32(
    MachineInstr &MI, Register &Reg1, Register &Reg2, Register &Dest,
    Register &R_mpy, bool &PrologCreated) {

  Register VR1, VR2;
  bool firstconvert = false, secondconvert = false;
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  // If the first operand is from add/subtract/multiply unit, generate IEEE
  // conversion instruction
  if (checkIfInputFromAdder32(Reg1) || checkIfInputFromMult32(Reg1)) {
    VR1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32), VR1).addReg(Reg1);
    firstconvert = true;
  }

  if (checkIfInputFromAdder32(Reg2) || checkIfInputFromMult32(Reg2)) {
    VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32), VR2).addReg(Reg2);
    secondconvert = true;
  }

  if (HST->useHVXV81Ops()) {
    if (firstconvert && secondconvert)
      return V81normalizeMultF32(MI, VR1, VR2, Dest, true, true, true);
    else if (firstconvert)
      return V81normalizeMultF32(MI, VR1, Reg2, Dest, true, false, true);
    else if (secondconvert)
      return V81normalizeMultF32(MI, Reg1, VR2, Dest, false, true, true);
    else
      return false;
  }

  // create prolog if not already created
  if (!PrologCreated && (firstconvert || secondconvert)) {
    createPrologInstructions(MI, R_mpy);
    PrologCreated = true;
  }

  Register input_mpy1, input_mpy2;

  // Normalize both IEEE converts
  if (firstconvert && secondconvert) {
    input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), input_mpy1)
        .addReg(R_mpy)
        .addReg(VR1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), input_mpy2)
        .addReg(R_mpy)
        .addReg(VR2);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(input_mpy2);
    // Normalize only first operand
  } else if (firstconvert) {
    input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), input_mpy1)
        .addReg(R_mpy)
        .addReg(VR1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(Reg2);
    // Normalize only second operand
  } else if (secondconvert) {
    input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32_mix), input_mpy2)
        .addReg(R_mpy)
        .addReg(VR2);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(Reg1)
        .addReg(input_mpy2);
  } else {
    // we do nothing if the inputs are not fromadder/subtracter/multiplier unit
    return false;
  }
  return true;
}

// Convert to IEEE and widen qf16 = vmpy(qf16/hf, qf16) conditionally
// Then convert qf32 to qf16
// twoOps: true if the first operand is qf type, false if hf type
bool HexagonXQFloatGenerator::convertWidenMultOp16(MachineInstr &MI,
                                                   Register &Reg1,
                                                   Register &Reg2,
                                                   Register &Dest,
                                                   bool twoOps) {

  Register VR1, VR2, output_mpy;
  bool firstconvert = false,
       secondconvert = false; // normalize with hf or qf16 operands
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  // If the first operand is from add/sub/mul unit,
  // generate IEEE conversion instruction
  if (checkIfInputFromAdder16(Reg1) || checkIfInputFromMult16(Reg1)) {
    VR1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR1).addReg(Reg1);
    firstconvert = true;
  }

  if (twoOps) {
    if (checkIfInputFromAdder16(Reg2) || checkIfInputFromMult16(Reg2)) {
      VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR2)
          .addReg(Reg2);
      secondconvert = true;
    }
  }

  if (twoOps) {
    // Both operands have been converted to IEEE
    if (firstconvert && secondconvert) {
      output_mpy = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_hf), output_mpy)
          .addReg(VR1)
          .addReg(VR2);
      // Only one operand has been converted to IEEE
    } else if (firstconvert) {
      output_mpy = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_mix_hf), output_mpy)
          .addReg(Reg2)
          .addReg(VR1);
    } else if (secondconvert) {
      output_mpy = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_mix_hf), output_mpy)
          .addReg(Reg1)
          .addReg(VR2);
    } else {
      // Neither have to be transformed
      return false;
    }
  } else {
    if (firstconvert) {
      output_mpy = MRI->createVirtualRegister(&Hexagon::HvxWRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_hf), output_mpy)
          .addReg(VR1)
          .addReg(Reg2);
    } else
      return false;
  }

  // convert qf32 to qf16
  generateQF16FromQF32(MI, Dest, output_mpy);

  return true;
}

// Convert to IEEE and perform qf32 = vmpy(qf16/hf, qf16) conditionally
// Final output is qf32 type
bool HexagonXQFloatGenerator::convertWidenMultOp32(MachineInstr &MI,
                                                   Register &Reg1,
                                                   Register &Reg2,
                                                   Register &Dest,
                                                   bool twoOps) {
  Register VR1, VR2;
  bool firstconvert = false,
       secondconvert = false; // normalize with hf or qf16 operands
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  // If the first operand is from add/subtract/multiply unit, generate IEEE
  // conversion instruction
  if (checkIfInputFromAdder16(Reg1) || checkIfInputFromMult16(Reg1)) {
    VR1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR1).addReg(Reg1);
    firstconvert = true;
  }

  if (twoOps) {
    if (checkIfInputFromAdder16(Reg2) || checkIfInputFromMult16(Reg2)) {
      VR2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vconv_hf_qf16), VR2)
          .addReg(Reg2);
      secondconvert = true;
    }
  }

  if (twoOps) {
    // Both operands have been converted to IEEE
    if (firstconvert && secondconvert) {
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_hf), Dest)
          .addReg(VR1)
          .addReg(VR2);
      // Only one operand has been converted to IEEE
    } else if (firstconvert) {
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_mix_hf), Dest)
          .addReg(Reg2)
          .addReg(VR1);
    } else if (secondconvert) {
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_mix_hf), Dest)
          .addReg(Reg1)
          .addReg(VR2);
    } else
      // Neither have to be transformed
      return false;
  } else {
    if (firstconvert)
      BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32_hf), Dest)
          .addReg(VR1)
          .addReg(Reg2);
    else
      return false;
  }

  return true;
}

// Normalize instructions of type qf32 = vmpy(qf32, qf32)
bool HexagonXQFloatGenerator::normalizeMultiplicationInputF32(
    MachineInstr &MI, Register &Reg1, Register &Reg2, Register &Dest,
    Register &R_mpy, bool &PrologCreated) {
  bool firstconvert = false, secondconvert = false;
  MachineBasicBlock &MBB = *MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();

  // We normalize only that operand which comes from add/subtract unit.
  if (checkIfInputFromAdder32(Reg1))
    firstconvert = true;
  if (checkIfInputFromAdder32(Reg2))
    secondconvert = true;

  // v81 normalization
  if (HST->useHVXV81Ops())
    return V81normalizeMultF32(MI, Reg1, Reg2, Dest, firstconvert,
                               secondconvert, false);

  // create normalization operand conditionally for v79
  if ((!PrologCreated && (firstconvert || secondconvert))) {
    createPrologInstructions(MI, R_mpy);
    PrologCreated = true;
  }

  Register input_mpy1, input_mpy2;

  // Normalize both input operands
  if (firstconvert && secondconvert) {
    input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
    input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32), input_mpy1)
        .addReg(R_mpy)
        .addReg(Reg1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32), input_mpy2)
        .addReg(R_mpy)
        .addReg(Reg2);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(input_mpy2);
    // Normalize only first operand
  } else if (firstconvert) {
    input_mpy1 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32), input_mpy1)
        .addReg(R_mpy)
        .addReg(Reg1);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy1)
        .addReg(Reg2);
    // Normalize only second operand
  } else if (secondconvert) {
    input_mpy2 = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);

    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vadd_qf32), input_mpy2)
        .addReg(R_mpy)
        .addReg(Reg2);
    BuildMI(MBB, MI, DL, HII->get(Hexagon::V6_vmpy_qf32), Dest)
        .addReg(input_mpy2)
        .addReg(Reg1);
  } else {
    // we do nothing if the inputs are not from adder/subtracter/multiplier unit
    return false;
  }

  return true;
}

bool HexagonXQFloatGenerator::deleteList() {
  if (OriginalMI.empty())
    return false;
  bool Changed = false;
  for (MachineInstr *origMI : OriginalMI) {
    LLVM_DEBUG(dbgs() << "deleting redundant instruction");
    LLVM_DEBUG(origMI->dump());
    origMI->eraseFromParent();
    Changed = true;
  }
  OriginalMI.clear();
  return Changed;
}

// Parent function to handle Loosy subnormal transformations
bool HexagonXQFloatGenerator::HandleLossySubnormals(MachineFunction &MF) {
  bool Changed = false;
  Register R_mpy;
  for (auto &MBB : MF) {
    bool PrologCreated = false;
    for (auto &MI : MBB) {
      Changed |= deleteList();
      // Skip if the instruction does not have two operands,
      // or is a bundle instruction
      // or is a debug instruction
      if (MI.getNumOperands() != 3 || MI.isDebugInstr())
        continue;
      auto Op1 = MI.getOperand(1);
      if (!Op1.isReg())
        continue;
      auto Op2 = MI.getOperand(2);
      if (!Op2.isReg())
        continue;
      auto Op0 = MI.getOperand(0);
      if (!Op0.isReg())
        continue;
      Register Reg1 = Op1.getReg();
      Register Reg2 = Op2.getReg();
      Register Dest = Op0.getReg();

      // FIXME Do not process physical registers as operands
      if (!Reg1.isVirtual() || !Reg2.isVirtual() || !Dest.isVirtual())
        continue;

      switch (MI.getOpcode()) {
      // qf32 = vmpy(qf32, qf32)
      // Normalize one or both input operands
      // if from add/sub unit
      case Hexagon::V6_vmpy_qf32:
        if (normalizeMultiplicationInputF32(MI, Reg1, Reg2, Dest, R_mpy,
                                            PrologCreated))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // qf16 = vmpy(qf16, qf16)
      // Widening multiply to qf32 and convert back to qf16
      // if any of the operands are from add/sub unit
      case Hexagon::V6_vmpy_qf16:
        if (widenMultiplicationInputF16(MI, Reg1, Reg2, Dest, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, Rt.hf)
      // Splat Rt to vector and then widening multiply
      // and then convert back to qf16
      // if first operand is from add/sub unit
      case Hexagon::V6_vmpy_rt_qf16:
        if (widenMultiplicationInputF16Rt(MI, Reg1, Reg2, Dest))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, hf)
      // Widening multiply to qf32 and convert back to qf16
      // if first operand is from add/sub unit
      case Hexagon::V6_vmpy_qf16_mix_hf:
        if (widenMultiplicationInputF16(MI, Reg1, Reg2, Dest, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      // Check if use of qf32 generating add/sub/mul instructions
      // are used as non-HVX operands.
      // If yes, convert the use to IEEE
      case Hexagon::V6_vadd_sf:
      case Hexagon::V6_vadd_qf32:
      case Hexagon::V6_vadd_qf32_mix:
      case Hexagon::V6_vsub_sf:
      case Hexagon::V6_vsub_qf32:
      case Hexagon::V6_vsub_qf32_mix:
      case Hexagon::V6_vsub_sf_mix:
      case Hexagon::V6_vmpy_qf32_qf16:
      case Hexagon::V6_vmpy_qf32_hf:
      case Hexagon::V6_vmpy_qf32_mix_hf:
      case Hexagon::V6_vmpy_rt_sf:
      case Hexagon::V6_vmpy_qf32_sf:
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      // Check if use of qf16 generating add/sub/mul instructions
      // are used as non-HVX operands.
      // If yes, convert the use to IEEE
      case Hexagon::V6_vadd_hf:
      case Hexagon::V6_vsub_hf:
      case Hexagon::V6_vadd_qf16:
      case Hexagon::V6_vsub_qf16:
      case Hexagon::V6_vadd_qf16_mix:
      case Hexagon::V6_vsub_qf16_mix:
      case Hexagon::V6_vsub_hf_mix:
      case Hexagon::V6_vmpy_qf16_hf:
      case Hexagon::V6_vmpy_rt_hf:
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      default:
        break;
      }
    }
  }
  if (OriginalMI.empty() || !Changed)
    return false;
  return true;
}

// Parent function to handle all IEEE-754 compliant transformations
bool HexagonXQFloatGenerator::HandleCompliantIEEE(MachineFunction &MF) {
  bool Changed = false;
  Register R_mpy;
  for (auto &MBB : MF) {
    bool PrologCreated = false;
    for (auto &MI : MBB) {
      Changed |= deleteList();
      // Skip if the instruction does not have two operands,
      // or is a bundle instruction
      // or is a debug instruction
      if (MI.getNumOperands() != 3 || MI.isDebugInstr())
        continue;

      auto Op1 = MI.getOperand(1);
      if (!Op1.isReg())
        continue;
      auto Op2 = MI.getOperand(2);
      if (!Op2.isReg())
        continue;
      auto Op0 = MI.getOperand(0);
      if (!Op0.isReg())
        continue;
      Register Reg1 = Op1.getReg();
      Register Reg2 = Op2.getReg();
      Register Dest = Op0.getReg();
      Register VRtSplat;

      // FIXME Do not process physical registers as operands
      if (!Reg1.isVirtual() || !Reg2.isVirtual() || !Dest.isVirtual())
        continue;

      switch (MI.getOpcode()) {

      // ==== Handle multiplication instructions ====

      // qf32 = vmpy(sf, Rt.sf)
      // Splat Rt to a vector
      // Normalize both input operands unconditionally
      case Hexagon::V6_vmpy_rt_sf:
        VRtSplat = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
        BuildMI(MBB, MI, MI.getDebugLoc(), HII->get(Hexagon::V6_lvsplatw),
                VRtSplat)
            .addReg(Reg2);
        normalizeMultiplicationInputSF(MI, Reg1, VRtSplat, Dest, R_mpy,
                                       PrologCreated);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // qf32 = vmpy(sf, sf)
      // Normalize both operands unconditionally
      case Hexagon::V6_vmpy_qf32_sf:
        normalizeMultiplicationInputSF(MI, Reg1, Reg2, Dest, R_mpy,
                                       PrologCreated);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // qf32 = vmpy(qf32, qf32)
      // Normalize one or both input operands
      // if from add/sub unit
      case Hexagon::V6_vmpy_qf32:
        if (normalizeMultiplicationInputF32(MI, Reg1, Reg2, Dest, R_mpy,
                                            PrologCreated))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // qf16 = vmpy(hf, rt)
      // Splat Rt to vector and then widening multiply
      case Hexagon::V6_vmpy_rt_hf:
        VRtSplat = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
        BuildMI(MBB, MI, MI.getDebugLoc(), HII->get(Hexagon::V6_lvsplatw),
                VRtSplat)
            .addReg(Reg2);
        widenMultiplyInputHF(MI, Reg1, VRtSplat, Dest);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // Widening multiply
      // qf16 = vmpy(hf, hf)
      case Hexagon::V6_vmpy_qf16_hf:
        widenMultiplyInputHF(MI, Reg1, Reg2, Dest);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, qf16)
      // Widening multiply to qf32 and convert back to qf16
      // if any of the operands are from add/sub unit
      case Hexagon::V6_vmpy_qf16:
        if (widenMultiplicationInputF16(MI, Reg1, Reg2, Dest, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, Rt.hf)
      // Splat Rt to vector and then widening multiply
      // and then convert back to qf16
      // if first operand is from add/sub unit
      case Hexagon::V6_vmpy_rt_qf16:
        if (widenMultiplicationInputF16Rt(MI, Reg1, Reg2, Dest))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, hf)
      // Widening multiply to qf32 and convert back to qf16
      // if first operand is from add/sub unit
      case Hexagon::V6_vmpy_qf16_mix_hf:
        if (widenMultiplicationInputF16(MI, Reg1, Reg2, Dest, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // Check if use of qf32/qf16 generating add/sub/mul
      // instructions are used as non-HVX operands.
      // If yes, convert the use to IEEE
      case Hexagon::V6_vadd_sf:
      case Hexagon::V6_vadd_qf32:
      case Hexagon::V6_vadd_qf32_mix:
      case Hexagon::V6_vsub_sf:
      case Hexagon::V6_vsub_qf32:
      case Hexagon::V6_vsub_qf32_mix:
      case Hexagon::V6_vsub_sf_mix:
      case Hexagon::V6_vmpy_qf32_qf16:
      case Hexagon::V6_vmpy_qf32_hf:
      case Hexagon::V6_vmpy_qf32_mix_hf:
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      case Hexagon::V6_vadd_hf:
      case Hexagon::V6_vsub_hf:
      case Hexagon::V6_vadd_qf16:
      case Hexagon::V6_vsub_qf16:
      case Hexagon::V6_vadd_qf16_mix:
      case Hexagon::V6_vsub_qf16_mix:
      case Hexagon::V6_vsub_hf_mix:
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      default:
        break;
      }
    }
  }
  if (OriginalMI.empty() || !Changed)
    return false;
  return true;
}

// Parent function to do strict IEEE transformations
bool HexagonXQFloatGenerator::HandleStrictIEEE(MachineFunction &MF) {

  bool Changed = false;
  Register R_mpy;
  for (auto &MBB : MF) {
    bool PrologCreated = false;
    for (auto &MI : MBB) {
      Changed |= deleteList();
      // Skip if the instruction does not have two operands,
      // or is a bundle instruction
      // or is a debug instruction
      if (MI.getNumOperands() != 3 || MI.isDebugInstr())
        continue;

      auto Op1 = MI.getOperand(1);
      if (!Op1.isReg())
        continue;
      auto Op2 = MI.getOperand(2);
      if (!Op2.isReg())
        continue;
      auto Op0 = MI.getOperand(0);
      if (!Op0.isReg())
        continue;
      Register Reg1 = Op1.getReg();
      Register Reg2 = Op2.getReg();
      Register Dest = Op0.getReg();
      Register VRtSplat;

      // FIXME Do not process physical registers as operands
      if (!Reg1.isVirtual() || !Reg2.isVirtual() || !Dest.isVirtual())
        continue;

      switch (MI.getOpcode()) {
      // ==== Handle add/subtract instructions ====
      // Convert one or both the input operands to IEEE 32-bit
      // if from add/sub/mult unit(s)
      // qf32 = vadd(qf32, qf32)
      case Hexagon::V6_vadd_qf32:
        if (convertAddOpToIEEE32(MI, Reg1, Reg2, Dest, true, true, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      // qf32 = vsub(qf32, qf32)
      case Hexagon::V6_vsub_qf32:
        if (convertAddOpToIEEE32(MI, Reg1, Reg2, Dest, false, true, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      // Convert only the first input operand to IEEE 32-bit
      // if it is from add/sub/mult unit
      // qf32 = vadd(qf32, sf)
      case Hexagon::V6_vadd_qf32_mix:
        if (convertAddOpToIEEE32(MI, Reg1, Reg2, Dest, true, true, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      // qf32 = vsub(qf32, sf)
      case Hexagon::V6_vsub_qf32_mix:
        if (convertAddOpToIEEE32(MI, Reg1, Reg2, Dest, false, true, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      // qf32 = vsub(sf, qf32)
      case Hexagon::V6_vsub_sf_mix:
        if (convertAddOpToIEEE32(MI, Reg1, Reg2, Dest, false, false, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
        break;

      // Convert one or both the input operands to IEEE 16-bit
      // if from add/sub/mult unit(s)
      // qf16 = vadd(qf16, qf16)
      case Hexagon::V6_vadd_qf16:
        if (convertAddOpToIEEE16(MI, Reg1, Reg2, Dest, true, true, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      // qf16 = vsub(qf16, qf16)
      case Hexagon::V6_vsub_qf16:
        if (convertAddOpToIEEE16(MI, Reg1, Reg2, Dest, false, true, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      // Convert only the first input operand IEEE 16-bit
      // if it is from add/sub/mul unit
      // qf16 = vadd(qf16, hf)
      case Hexagon::V6_vadd_qf16_mix:
        if (convertAddOpToIEEE16(MI, Reg1, Reg2, Dest, true, true, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      // qf16 = vsub(qf16, hf)
      case Hexagon::V6_vsub_qf16_mix:
        if (convertAddOpToIEEE16(MI, Reg1, Reg2, Dest, false, true, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      // qf16 = vsub(hf, qf16)
      case Hexagon::V6_vsub_hf_mix:
        if (convertAddOpToIEEE16(MI, Reg1, Reg2, Dest, false, false, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // ==== Handle multiplication instructions ====

      // qf32 = vmpy(sf, Rt.sf)
      // Splat Rt to a vector
      // Normalize both input operands unconditionally
      case Hexagon::V6_vmpy_rt_sf:
        VRtSplat = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
        BuildMI(MBB, MI, MI.getDebugLoc(), HII->get(Hexagon::V6_lvsplatw),
                VRtSplat)
            .addReg(Reg2);
        normalizeMultiplicationInputSF(MI, Reg1, VRtSplat, Dest, R_mpy,
                                       PrologCreated);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // Normalize both operands unconditionally
      // qf32 = vmpy(sf, sf)
      case Hexagon::V6_vmpy_qf32_sf:
        normalizeMultiplicationInputSF(MI, Reg1, Reg2, Dest, R_mpy,
                                       PrologCreated);
        Changed |= convertIfInputToNonHVX(MI, true);
        OriginalMI.push_back(&MI);
        break;

      // Convert one or both input operands to IEEE 32-bit
      // if from add/sub/mult unit and normalize
      // qf32 = vmpy(qf32, qf32)
      case Hexagon::V6_vmpy_qf32:
        if (convertNormalizeMultOp32(MI, Reg1, Reg2, Dest, R_mpy,
                                     PrologCreated))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // Convert one or both input operands to IEEE 16-bit
      // if from mul/add/sub unit;
      // then widening multiply to generate qf32
      // then convert to qf16
      // qf16 = vmpy(qf16, qf16)
      case Hexagon::V6_vmpy_qf16:
        if (convertWidenMultOp16(MI, Reg1, Reg2, Dest, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // Convert one or both input operands to IEEE 16-bit
      // if from mul/add/sub unit;
      // then widening multiply to generate qf32
      // qf32 = vmpy(qf16, qf16)
      case Hexagon::V6_vmpy_qf32_qf16:
        if (convertWidenMultOp32(MI, Reg1, Reg2, Dest, true))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;

      // qf16 = vmpy(hf, rt)
      // Splat Rt to vector and then widening multiply
      case Hexagon::V6_vmpy_rt_hf:
        VRtSplat = MRI->createVirtualRegister(&Hexagon::HvxVRRegClass);
        BuildMI(MBB, MI, MI.getDebugLoc(), HII->get(Hexagon::V6_lvsplatw),
                VRtSplat)
            .addReg(Reg2);
        widenMultiplyInputHF(MI, Reg1, VRtSplat, Dest);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // Widening multiply, then convert to IEEE
      // qf16 = vmpy(hf, hf)
      case Hexagon::V6_vmpy_qf16_hf:
        widenMultiplyInputHF(MI, Reg1, Reg2, Dest);
        OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, Rt.hf)
      // Splat Rt to vector and then widening multiply
      // and then convert back to qf16
      // if first operand is from add/sub unit
      case Hexagon::V6_vmpy_rt_qf16:
        if (widenMultiplicationInputF16Rt(MI, Reg1, Reg2, Dest))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf16 = vmpy(qf16, hf)
      // Convert only the first input operans to IEEE 16-bit
      // if from mul/add/sub unit;
      // then widening multiply to generate qf32
      // then convert back to qf16
      case Hexagon::V6_vmpy_qf16_mix_hf:
        if (convertWidenMultOp16(MI, Reg1, Reg2, Dest, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, false);
        break;

      // qf32 = vmpy(qf16, hf)
      // Convert only the first input operans to IEEE 16-bit
      // if from mul/add/sub unit;
      // then widening multiply to generate qf32
      case Hexagon::V6_vmpy_qf32_mix_hf:
        if (convertWidenMultOp32(MI, Reg1, Reg2, Dest, false))
          OriginalMI.push_back(&MI);
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      // Check if use of qf32/qf16 generating add/sub/mul
      // instructions are used as non-HVX operands.
      // If yes, convert the use to IEEE
      case Hexagon::V6_vadd_sf:
      case Hexagon::V6_vsub_sf:
        Changed |= convertIfInputToNonHVX(MI, true);
        break;
      case Hexagon::V6_vadd_hf:
      case Hexagon::V6_vsub_hf:
        Changed |= convertIfInputToNonHVX(MI, false);
        break;
      default:
        break;
      }
    }
  }
  if (OriginalMI.empty() || !Changed)
    return false;
  return true;
}

// There is no conversions in lossy mode
bool HexagonXQFloatGenerator::HandleLossyLegacy(MachineFunction &MF) {
  return false;
}

bool HexagonXQFloatGenerator::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableHVXXQFloat || (QFloatModeValue == QFloatMode::Legacy))
    return false;

  bool Changed = false;
  HST = &MF.getSubtarget<HexagonSubtarget>();
  HII = HST->getInstrInfo();
  MRI = &MF.getRegInfo();

  if (EnableConversionsRemoval &&
      !(QFloatModeValue == QFloatMode::StrictIEEE)) {
    VectorConvertRemove VCR(MF, MRI, HST);
    VCR.run();
    LLVM_DEBUG(dbgs() << "\nExtraneous conversion instructions removed for "
                      << MF.getName());
    if (PrintDebug)
      debug_print(MF);
  }

  switch (QFloatModeValue) {
  case QFloatMode::StrictIEEE:
    LLVM_DEBUG(dbgs() << "\nGenerating code for STRICT-IEEE mode.\n");
    Changed = HandleStrictIEEE(MF);
    break;
  case QFloatMode::IEEE:
    LLVM_DEBUG(dbgs() << "\nGenerating code for IEEE mode.\n");
    Changed = HandleCompliantIEEE(MF);
    break;
  case QFloatMode::Lossy:
    LLVM_DEBUG(dbgs() << "\nGenerating code for LOSSY mode.\n");
    Changed = HandleLossySubnormals(MF);
    break;
  case QFloatMode::Legacy:
    LLVM_DEBUG(dbgs() << "\nGenerating code for LEGACY mode.\n");
    Changed = HandleLossyLegacy(MF);
    break;
  }
  LLVM_DEBUG(dbgs() << "...fine");

  // Delete the original instructions
  for (MachineInstr *origMI : OriginalMI) {
    LLVM_DEBUG(origMI->dump());
    origMI->eraseFromParent();
  }
  OriginalMI.clear();

  return Changed;
}
