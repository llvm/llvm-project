//===-- P2MCInstLower.h - Lower MachineInstr to MCInst -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2MCINSTLOWER_H
#define LLVM_LIB_TARGET_P2_P2MCINSTLOWER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/Support/Compiler.h"
#include "P2BaseInfo.h"

namespace llvm {
    class MCContext;
    class MCInst;
    class MCOperand;
    class MachineInstr;
    class MachineFunction;
    class P2AsmPrinter;

    /// This class is used to lower an MachineInstr into an MCInst.
    class LLVM_LIBRARY_VISIBILITY P2MCInstLower {
        typedef MachineOperand::MachineOperandType MachineOperandType;
        MCContext *Ctx;
        P2AsmPrinter &AsmPrinter;

        MCOperand lowerSymbolOperand(const MachineOperand &MO, MachineOperandType MOTy) const;

        /**
         * Generate an augs or augd to modify the operand in MI given by op_num. 
         */
        void createAugInst(const MachineInstr &MI, MCInst &aug, int op_num) const;

        /**
         * create a basic aug instruction with given type and value
         */
        void createAugInst(MCInst &aug, int type, int value, int condition=P2::ALWAYS) const ;

        /**
         * return true if the instruction can by augmented with augs/d
         */
        bool canAug(const MachineInstr &MI) const;
    public:
        P2MCInstLower(P2AsmPrinter &asmprinter);
        void Initialize(MCContext* C);
        void lowerInstruction(const MachineInstr &MI, MCInst &AugMI, MCInst &OutMI) const;
        MCOperand lowerOperand(const MachineOperand& MO) const;
    };
}

#endif