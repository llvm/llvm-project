// WebAssemblyMachineFunctionInfo.h-WebAssembly machine function info-*- C++ -*-
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file declares WebAssembly-specific per-machine-function
/// information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYMACHINEFUNCTIONINFO_H

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

namespace llvm {

/// This class is derived from MachineFunctionInfo and contains private
/// WebAssembly-specific information for each MachineFunction.
class WebAssemblyFunctionInfo final : public MachineFunctionInfo {
  MachineFunction &MF;

  std::vector<MVT> Params;

  /// A mapping from CodeGen vreg index to WebAssembly register number.
  std::vector<unsigned> WARegs;

  /// A mapping from CodeGen vreg index to a boolean value indicating whether
  /// the given register is considered to be "stackified", meaning it has been
  /// determined or made to meet the stack requirements:
  ///   - single use (per path)
  ///   - single def (per path)
  ///   - defined and used in LIFO order with other stack registers
  BitVector VRegStackified;

  // One entry for each possible target reg. we expect it to be small.
  std::vector<unsigned> PhysRegs;

public:
  explicit WebAssemblyFunctionInfo(MachineFunction &MF) : MF(MF) {
    PhysRegs.resize(WebAssembly::NUM_TARGET_REGS, -1U);
  }
  ~WebAssemblyFunctionInfo() override;

  void addParam(MVT VT) { Params.push_back(VT); }
  const std::vector<MVT> &getParams() const { return Params; }

  static const unsigned UnusedReg = -1u;

  void stackifyVReg(unsigned VReg) {
    if (TargetRegisterInfo::virtReg2Index(VReg) >= VRegStackified.size())
      VRegStackified.resize(TargetRegisterInfo::virtReg2Index(VReg) + 1);
    VRegStackified.set(TargetRegisterInfo::virtReg2Index(VReg));
  }
  void unstackifyVReg(unsigned VReg) {
    if (TargetRegisterInfo::virtReg2Index(VReg) >= VRegStackified.size())
      return;
    VRegStackified.reset(TargetRegisterInfo::virtReg2Index(VReg));
  }
  bool isVRegStackified(unsigned VReg) const {
    if (TargetRegisterInfo::virtReg2Index(VReg) >= VRegStackified.size())
      return false;
    return VRegStackified.test(TargetRegisterInfo::virtReg2Index(VReg));
  }

  void initWARegs();
  void setWAReg(unsigned VReg, unsigned WAReg) {
    assert(WAReg != UnusedReg);
    assert(TargetRegisterInfo::virtReg2Index(VReg) < WARegs.size());
    WARegs[TargetRegisterInfo::virtReg2Index(VReg)] = WAReg;
  }
  unsigned getWAReg(unsigned Reg) const {
    if (TargetRegisterInfo::isVirtualRegister(Reg)) {
      assert(TargetRegisterInfo::virtReg2Index(Reg) < WARegs.size());
      return WARegs[TargetRegisterInfo::virtReg2Index(Reg)];
    }
    return PhysRegs[Reg];
  }
  // If new virtual registers are created after initWARegs has been called,
  // this function can be used to add WebAssembly register mappings for them.
  void addWAReg(unsigned VReg, unsigned WAReg) {
    assert(VReg = WARegs.size());
    WARegs.push_back(WAReg);
  }

  void addPReg(unsigned PReg, unsigned WAReg) {
    assert(PReg < WebAssembly::NUM_TARGET_REGS);
    assert(WAReg < -1U);
    PhysRegs[PReg] = WAReg;
  }
  const std::vector<unsigned> &getPhysRegs() const { return PhysRegs; }
};

} // end namespace llvm

#endif
