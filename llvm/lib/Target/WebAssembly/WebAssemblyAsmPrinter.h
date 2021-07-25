// WebAssemblyAsmPrinter.h - WebAssembly implementation of AsmPrinter-*- C++ -*-
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYASMPRINTER_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYASMPRINTER_H

#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class WebAssemblyTargetStreamer;

class LLVM_LIBRARY_VISIBILITY WebAssemblyAsmPrinter final : public AsmPrinter {
  const WebAssemblySubtarget *Subtarget;
  const MachineRegisterInfo *MRI;
  WebAssemblyFunctionInfo *MFI;
  // TODO: Do the uniquing of Signatures here instead of ObjectFileWriter?
  std::vector<std::unique_ptr<wasm::WasmSignature>> Signatures;
  std::vector<std::unique_ptr<std::string>> Names;
  bool signaturesEmitted = false;

  StringRef storeName(StringRef Name) {
    std::unique_ptr<std::string> N = std::make_unique<std::string>(Name);
    Names.push_back(std::move(N));
    return *Names.back();
  }

public:
  explicit WebAssemblyAsmPrinter(TargetMachine &TM,
                                 std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), Subtarget(nullptr), MRI(nullptr),
        MFI(nullptr) {}

  StringRef getPassName() const override {
    return "WebAssembly Assembly Printer";
  }

  const WebAssemblySubtarget &getSubtarget() const { return *Subtarget; }
  void addSignature(std::unique_ptr<wasm::WasmSignature> &&Sig) {
    Signatures.push_back(std::move(Sig));
  }

  //===------------------------------------------------------------------===//
  // MachineFunctionPass Implementation.
  //===------------------------------------------------------------------===//

  bool runOnMachineFunction(MachineFunction &MF) override {
    Subtarget = &MF.getSubtarget<WebAssemblySubtarget>();
    MRI = &MF.getRegInfo();
    MFI = MF.getInfo<WebAssemblyFunctionInfo>();
    return AsmPrinter::runOnMachineFunction(MF);
  }

  //===------------------------------------------------------------------===//
  // AsmPrinter Implementation.
  //===------------------------------------------------------------------===//

  void emitEndOfAsmFile(Module &M) override;
  void EmitProducerInfo(Module &M);
  void EmitTargetFeatures(Module &M);
  void emitGlobalVariable(const GlobalVariable *GV) override;
  void emitJumpTableInfo() override;
  void emitConstantPool() override;
  void emitLinkage(const GlobalValue *, MCSymbol *) const override;
  void emitFunctionBodyStart() override;
  void emitInstruction(const MachineInstr *MI) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;

  MVT getRegType(unsigned RegNo) const;
  std::string regToString(const MachineOperand &MO);
  WebAssemblyTargetStreamer *getTargetStreamer();
  MCSymbolWasm *getMCSymbolForFunction(const Function *F, bool EnableEmEH,
                                       wasm::WasmSignature *Sig,
                                       bool &InvokeDetected);
  MCSymbol *getOrCreateWasmSymbol(StringRef Name);
  void emitExternalDecls(const Module &M);
};

} // end namespace llvm

#endif
