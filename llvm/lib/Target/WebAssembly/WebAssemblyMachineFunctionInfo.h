// WebAssemblyMachineFunctionInfo.h-WebAssembly machine function info-*- C++ -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares WebAssembly-specific per-machine-function
/// information.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYMACHINEFUNCTIONINFO_H

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/MC/MCSymbolWasm.h"

namespace llvm {

namespace yaml {
struct WebAssemblyFunctionInfo;
}

/// This class is derived from MachineFunctionInfo and contains private
/// WebAssembly-specific information for each MachineFunction.
class WebAssemblyFunctionInfo final : public MachineFunctionInfo {
  MachineFunction &MF;

  std::vector<MVT> Params;
  std::vector<MVT> Results;
  std::vector<MVT> Locals;

  /// A mapping from CodeGen vreg index to WebAssembly register number.
  std::vector<unsigned> WARegs;

  /// A mapping from CodeGen vreg index to a boolean value indicating whether
  /// the given register is considered to be "stackified", meaning it has been
  /// determined or made to meet the stack requirements:
  ///   - single use (per path)
  ///   - single def (per path)
  ///   - defined and used in LIFO order with other stack registers
  BitVector VRegStackified;

  // A virtual register holding the pointer to the vararg buffer for vararg
  // functions. It is created and set in TLI::LowerFormalArguments and read by
  // TLI::LowerVASTART
  unsigned VarargVreg = -1U;

  // A virtual register holding the base pointer for functions that have
  // overaligned values on the user stack.
  unsigned BasePtrVreg = -1U;

  // Function properties.
  bool CFGStackified = false;

public:
  explicit WebAssemblyFunctionInfo(MachineFunction &MF) : MF(MF) {}
  ~WebAssemblyFunctionInfo() override;
  void initializeBaseYamlFields(const yaml::WebAssemblyFunctionInfo &YamlMFI);

  void addParam(MVT VT) { Params.push_back(VT); }
  const std::vector<MVT> &getParams() const { return Params; }

  void addResult(MVT VT) { Results.push_back(VT); }
  const std::vector<MVT> &getResults() const { return Results; }

  void clearParamsAndResults() {
    Params.clear();
    Results.clear();
  }

  void setNumLocals(size_t NumLocals) { Locals.resize(NumLocals, MVT::i32); }
  void setLocal(size_t i, MVT VT) { Locals[i] = VT; }
  void addLocal(MVT VT) { Locals.push_back(VT); }
  const std::vector<MVT> &getLocals() const { return Locals; }

  unsigned getVarargBufferVreg() const {
    assert(VarargVreg != -1U && "Vararg vreg hasn't been set");
    return VarargVreg;
  }
  void setVarargBufferVreg(unsigned Reg) { VarargVreg = Reg; }

  unsigned getBasePointerVreg() const {
    assert(BasePtrVreg != -1U && "Base ptr vreg hasn't been set");
    return BasePtrVreg;
  }
  void setBasePointerVreg(unsigned Reg) { BasePtrVreg = Reg; }

  static const unsigned UnusedReg = -1u;

  void stackifyVReg(unsigned VReg) {
    assert(MF.getRegInfo().getUniqueVRegDef(VReg));
    auto I = TargetRegisterInfo::virtReg2Index(VReg);
    if (I >= VRegStackified.size())
      VRegStackified.resize(I + 1);
    VRegStackified.set(I);
  }
  bool isVRegStackified(unsigned VReg) const {
    auto I = TargetRegisterInfo::virtReg2Index(VReg);
    if (I >= VRegStackified.size())
      return false;
    return VRegStackified.test(I);
  }

  void initWARegs();
  void setWAReg(unsigned VReg, unsigned WAReg) {
    assert(WAReg != UnusedReg);
    auto I = TargetRegisterInfo::virtReg2Index(VReg);
    assert(I < WARegs.size());
    WARegs[I] = WAReg;
  }
  unsigned getWAReg(unsigned VReg) const {
    auto I = TargetRegisterInfo::virtReg2Index(VReg);
    assert(I < WARegs.size());
    return WARegs[I];
  }

  // For a given stackified WAReg, return the id number to print with push/pop.
  static unsigned getWARegStackId(unsigned Reg) {
    assert(Reg & INT32_MIN);
    return Reg & INT32_MAX;
  }

  bool isCFGStackified() const { return CFGStackified; }
  void setCFGStackified(bool Value = true) { CFGStackified = Value; }
};

void computeLegalValueVTs(const Function &F, const TargetMachine &TM, Type *Ty,
                          SmallVectorImpl<MVT> &ValueVTs);

// Compute the signature for a given FunctionType (Ty). Note that it's not the
// signature for F (F is just used to get varous context)
void computeSignatureVTs(const FunctionType *Ty, const Function &F,
                         const TargetMachine &TM, SmallVectorImpl<MVT> &Params,
                         SmallVectorImpl<MVT> &Results);

void valTypesFromMVTs(const ArrayRef<MVT> &In,
                      SmallVectorImpl<wasm::ValType> &Out);

std::unique_ptr<wasm::WasmSignature>
signatureFromMVTs(const SmallVectorImpl<MVT> &Results,
                  const SmallVectorImpl<MVT> &Params);

namespace yaml {

struct WebAssemblyFunctionInfo final : public yaml::MachineFunctionInfo {
  bool CFGStackified = false;

  WebAssemblyFunctionInfo() = default;
  WebAssemblyFunctionInfo(const llvm::WebAssemblyFunctionInfo &MFI);

  void mappingImpl(yaml::IO &YamlIO) override;
  ~WebAssemblyFunctionInfo() = default;
};

template <> struct MappingTraits<WebAssemblyFunctionInfo> {
  static void mapping(IO &YamlIO, WebAssemblyFunctionInfo &MFI) {
    YamlIO.mapOptional("isCFGStackified", MFI.CFGStackified, false);
  }
};

} // end namespace yaml

} // end namespace llvm

#endif
