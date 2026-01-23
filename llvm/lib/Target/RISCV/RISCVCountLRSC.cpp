//===-- RISCVCountLRSC.cpp - Count LR/SC instruction pairs -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that counts the # of the LR/SC pairs. This pass
// should run just before code generation.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h" 
// Added includes:
#include <unordered_map>
#include "llvm/Support/JSON.h"
#include "llvm/Support/FormatVariadic.h"
#include <string>
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"

using namespace llvm;

#define RISCV_COUNT_LR_SC_NAME "RISC-V count LR/SC instruction pairs"
#define DEBUG_TYPE "riscvcntlrsc"

namespace {

class RISCVCountLRSC : public MachineFunctionPass {
public:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;

  static char ID;

  RISCVCountLRSC() : MachineFunctionPass(ID) {}
  ~RISCVCountLRSC();

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return RISCV_COUNT_LR_SC_NAME; }

  void print(raw_ostream &OS) const;

private:
  unsigned countLRSC(MachineBasicBlock &MBB,
                     MachineFunction &MF); // added &MF as a param
  unsigned totalCount = 0;

  // How it used to be:
  /*    std::unordered_map<const MachineFunction*, std::unordered_map<const
     MachineBasicBlock*, std::unordered_map< uint16_t, int>>>
     basicBlocksFlavourCounts; //mapping of MF address -> BB address -> (mapping
     of opcode[uint16_t] -> count of LR/SC occurrences) std::unordered_map<const
     MachineFunction*, std::unordered_map<const MachineBasicBlock* , int>>
     basicBlocksCounts;// mapping of MF address -> BB address -> count of LR/SC
     occurrences in the BB std::unordered_map<const MachineFunction*, int>
     functionCounts;// mapping of MF address -> count of LR/SC occurrences in
     the function
  */

  // Made a struct that gathers all of the umaps together
  struct LRSCCounts {
    using MFKey = const MachineFunction *;
    using BBKey = const MachineBasicBlock *;
    using OpKey = std::string;

    using FlavourMap = std::unordered_map<OpKey, int>;
    using BBToFlavourMap = std::unordered_map<BBKey, FlavourMap>;
    using MFToBBFlavourMap = std::unordered_map<MFKey, BBToFlavourMap>;

    using BBCountMap = std::unordered_map<BBKey, int>;
    using MFToBBCountMap = std::unordered_map<MFKey, BBCountMap>;

    using MFCountMap = std::unordered_map<MFKey, int>;

    // MF -> BB -> (opcode -> count)
    MFToBBFlavourMap basicBlocksFlavourCounts;

    // MF -> BB -> total LR/SC occurrences (or pairs, depending on your
    // definition)
    MFToBBCountMap basicBlocksCounts;

    // MF -> total LR/SC occurrences (or pairs)
    MFCountMap functionCounts;
    
        void clear() {
      basicBlocksFlavourCounts.clear();
      basicBlocksCounts.clear();
      functionCounts.clear();
    }
    llvm::json::Value toJSON() const { //
      llvm::json::Object Root;
      llvm::json::Array Funcs;

      for (const auto &FP : functionCounts) {
        const MachineFunction *MF = FP.first;
        int FuncTotal = FP.second;

        llvm::json::Object FObj;
        FObj["function"] = MF ? MF->getName().str() : "<null>";
        FObj["total_lrsc_occurrences"] = FuncTotal;

        llvm::json::Array Blocks;

        auto ItBBTotals = basicBlocksCounts.find(MF);
        auto ItBBFlavors = basicBlocksFlavourCounts.find(MF);

        if (ItBBTotals != basicBlocksCounts.end()) {
          for (const auto &BBP : ItBBTotals->second) {
            const MachineBasicBlock *MBB = BBP.first;
            int BBTotal = BBP.second;

            llvm::json::Object BBObj;
            BBObj["bb_number"] = MBB ? MBB->getNumber() : -1;
            BBObj["bb_total_lrsc_occurrences"] = BBTotal;

            llvm::json::Object FlavorsObj; // flavor string -> count

            if (ItBBFlavors != basicBlocksFlavourCounts.end()) {
              auto ItFlavorMapForBB = ItBBFlavors->second.find(MBB);
              if (ItFlavorMapForBB != ItBBFlavors->second.end()) {
                for (const auto &OP : ItFlavorMapForBB->second) {
                  const std::string &Flavor = OP.first;
                  int Cnt = OP.second;
                  FlavorsObj.try_emplace(Flavor, Cnt);
                }
              }
            }

            BBObj["flavors"] = std::move(FlavorsObj);
            Blocks.push_back(std::move(BBObj));
          }
        }

        FObj["basic_blocks"] = std::move(Blocks);
        Funcs.push_back(std::move(FObj));
      }

      Root["riscv_lrsc_counts"] = std::move(Funcs);
      return llvm::json::Value(std::move(Root));
    }
  };
  LRSCCounts Counts;
};

} // end anonymous namespace

char RISCVCountLRSC::ID = 0;
INITIALIZE_PASS(RISCVCountLRSC, "riscv-count-lr-sc", RISCV_COUNT_LR_SC_NAME,
                false, false)

RISCVCountLRSC::~RISCVCountLRSC() { print(dbgs()); }

FunctionPass *llvm::createRISCVCountLRSCPass() { return new RISCVCountLRSC(); }

bool RISCVCountLRSC::runOnMachineFunction(MachineFunction &MF) {
  Counts.clear(); //Clears all the stats for the new function
  STI =
      &MF.getSubtarget<RISCVSubtarget>(); // Sub Target Instruction CPU features
                                          // :extensions, scheduling model, etc.
  TII = STI->getInstrInfo(); // Instruction Info Table : opcodes, pseudo
                             // expansion info, etc.
  unsigned bbCount = 0;
  unsigned mfCount = 0;

  // traverse through the machine basic blocks and
  // get the running count of detected LR/SC pairs [Ali]: We are not counting
  // pairs are we?
  for (auto &MBB : MF) {

    bbCount = countLRSC(MBB, MF); // # LR/SC occurrences in a BB
    Counts.basicBlocksCounts[&MF][&MBB] +=
        bbCount; // # LR/SC occurrences in a BB added to the mapping of MF -> BB
                 // -> count
    mfCount += bbCount; // # LR/SC occurrences in a BB added to the total number
                        // of instructions for the fucntion
  }
  totalCount += mfCount; // # LR/SC occurrences in a BB added to the total
                         // number of instructions for the whole compilation
  Counts.functionCounts[&MF] += mfCount; // # LR/SC occurrences in a BB added to
                                         // the mapping of MF -> count
  return false; // returns false right? we are not modifying the code during
                // compilation
}

unsigned
RISCVCountLRSC::countLRSC(MachineBasicBlock &MBB,
                          MachineFunction &MF) { // added &MF as a param

  MachineBasicBlock::iterator MBBI = MBB.begin();
  MachineBasicBlock::iterator E = MBB.end();

  // iterates over each MBBI and compares the opcode to every flavour of LR/SC
  // instructions and adds to the count of each flavour of each instruction in
  // an unordered mapping

  unsigned total = 0;
  uint16_t opc = 0;
  while (MBBI != E) {

    opc = MBBI->getOpcode();
    switch (opc) {
    // LR flavours
    case RISCV::LR_W:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_W"]++;
      total++;
      break;
    case RISCV::LR_D:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_D"]++;
      total++;
      break;
    case RISCV::LR_D_AQ:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_D_AQ"]++;
      total++;
      break;
    case RISCV::LR_W_AQ:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_W_AQ"]++;
      total++;
      break;
    case RISCV::LR_D_RL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_D_RL"]++;
      total++;
      break;
    case RISCV::LR_W_RL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_W_RL"]++;
      total++;
      break;
    case RISCV::LR_D_AQRL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_D_AQRL"]++;
      total++;
      break;
    case RISCV::LR_W_AQRL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["LR_W_AQRL"]++;
      total++;
      break;

      // SC flavours
    case RISCV::SC_W:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_W"]++;
      total++;
      break;
    case RISCV::SC_D:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_D"]++;
      total++;
      break;
    case RISCV::SC_D_AQ:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_D_AQ"]++;
      total++;
      break;
    case RISCV::SC_W_AQ:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_W_AQ"]++;
      total++;
      break;
    case RISCV::SC_D_RL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_D_RL"]++;
      total++;
      break;
    case RISCV::SC_W_RL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_W_RL"]++;
      total++;
      break;
    case RISCV::SC_D_AQRL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_D_AQRL"]++;
      total++;
      break;
    case RISCV::SC_W_AQRL:
      Counts.basicBlocksFlavourCounts[&MF][&MBB]["SC_W_AQRL"]++;
      total++;
      break;

    default:
      break;
    }

    MBBI++;
  }
  return total;
}

void RISCVCountLRSC::print(raw_ostream &OS) const {
  OS << "Number of LR/SC instruction: " << totalCount << "\n";

  llvm::json::Value J = Counts.toJSON();

  // Pretty print with indent=2
  OS << llvm::formatv("{0:2}", J) << "\n";
}
