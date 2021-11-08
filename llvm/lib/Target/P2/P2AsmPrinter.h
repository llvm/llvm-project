//===-- P2AsmPrinter.h - P2 LLVM Assembly Printer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// P2 Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2ASMPRINTER_H
#define LLVM_LIB_TARGET_P2_P2ASMPRINTER_H

#include "P2MachineFunctionInfo.h"
#include "P2MCInstLower.h"

#include "P2TargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
    class MCStreamer;
    class MachineInstr;
    class MachineBasicBlock;
    class Module;
    class raw_ostream;
    class StringRef;

    class LLVM_LIBRARY_VISIBILITY P2AsmPrinter : public AsmPrinter {

    public:
        const P2FunctionInfo *P2FI;
        P2MCInstLower MCInstLowering;

        explicit P2AsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
                            : AsmPrinter(TM, std::move(Streamer)), MCInstLowering(*this) {}

        virtual StringRef getPassName() const override {
            return "P2 Assembly Printer";
        }

        virtual bool runOnMachineFunction(MachineFunction &MF) override;

        void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);
        bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum, const char *ExtraCode, raw_ostream &O) override;
        bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum, const char *ExtraCode, raw_ostream &O) override;

        void emitInstruction(const MachineInstr *MI) override;

        void emitFunctionEntryLabel() override;
        void emitFunctionBodyStart() override;
        void emitFunctionBodyEnd() override;
        void emitStartOfAsmFile(Module &M) override;

        void emitInlineAsmStart() const override;
        void emitInlineAsmEnd(const MCSubtargetInfo &StartInfo, const MCSubtargetInfo *EndInfo) const override;
    };
}

#endif