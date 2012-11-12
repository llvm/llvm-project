//===-- SPUNopFiller.cpp - Add nops/lnops to align the pipelines ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The final pass just before assembly printing. This pass is the last
// checkpoint where nops and lnops are added to the instruction stream to 
// satisfy the dual issue requirements. The actual dual issue scheduling is 
// done (TODO: nowhere, currently)
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "SPUTargetMachine.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
  struct SPUNopFiller : public MachineFunctionPass {

    TargetMachine &TM;
    const TargetInstrInfo *TII;
    const InstrItineraryData *IID;
    bool isEvenPlace;  // the instruction slot (mem address) at hand is even/odd

    static char ID;
    SPUNopFiller(TargetMachine &tm) 
      : MachineFunctionPass(ID), TM(tm), TII(tm.getInstrInfo()), 
        IID(tm.getInstrItineraryData()) 
    {
      DEBUG( dbgs() << "********** SPU Nop filler **********\n" ; );
    }

    virtual const char *getPassName() const {
      return "SPU nop/lnop Filler";
    }

    void runOnMachineBasicBlock(MachineBasicBlock &MBB);

    bool runOnMachineFunction(MachineFunction &F) {
      isEvenPlace = true; //all functions get an .align 3 directive at start 
      for (MachineFunction::iterator FI = F.begin(), FE = F.end();
           FI != FE; ++FI)
        runOnMachineBasicBlock(*FI);
      return true; //never-ever do any more modifications, just print it!
    }

    typedef enum { none   = 0, // no more instructions in this function / BB
                   pseudo = 1, // this does not get executed
                   even   = 2, 
                   odd    = 3 } SPUOpPlace;
    SPUOpPlace getOpPlacement( MachineInstr &instr );

  };
  char SPUNopFiller::ID = 0;

} 

// Fill a BasicBlock to alignment. 
// In the assebly we align the functions to 'even' adresses, but
// basic blocks have an implicit alignmnet. We hereby define 
// basic blocks to have the same, even, alignment.
void SPUNopFiller::
runOnMachineBasicBlock(MachineBasicBlock &MBB) 
{
  assert( isEvenPlace && "basic block start from odd address");
  for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I)
  {
    SPUOpPlace this_optype, next_optype;
    MachineBasicBlock::iterator J = I;
    J++;

    this_optype = getOpPlacement( *I );
    next_optype = none;
    while (J!=MBB.end()){
      next_optype = getOpPlacement( *J );
      ++J;
      if (next_optype != pseudo ) 
        break;
    }

    // padd: odd(wrong), even(wrong), ...
    // to:   nop(corr), odd(corr), even(corr)...
    if( isEvenPlace && this_optype == odd && next_optype == even ) {
      DEBUG( dbgs() <<"Adding NOP before: "; );
      DEBUG( I->dump(); );
      BuildMI(MBB, I, I->getDebugLoc(), TII->get(SPU::ENOP));
      isEvenPlace=false;
    }
    
    // padd: even(wrong), odd(wrong), ...
    // to:   lnop(corr), even(corr), odd(corr)...
    else if ( !isEvenPlace && this_optype == even && next_optype == odd){
      DEBUG( dbgs() <<"Adding LNOP before: "; );
      DEBUG( I->dump(); );
      BuildMI(MBB, I, I->getDebugLoc(), TII->get(SPU::LNOP));
      isEvenPlace=true;
    }
      
    // now go to next mem slot
    if( this_optype != pseudo )
      isEvenPlace = !isEvenPlace;    

  }

  // padd basicblock end
  if( !isEvenPlace ){
    MachineBasicBlock::iterator J = MBB.end();
    J--;
    if (getOpPlacement( *J ) == odd) {
      DEBUG( dbgs() <<"Padding basic block with NOP\n"; );
      BuildMI(MBB, J, J->getDebugLoc(), TII->get(SPU::ENOP));
    }  
    else {
      J++;
      DEBUG( dbgs() <<"Padding basic block with LNOP\n"; );
      BuildMI(MBB, J, DebugLoc(), TII->get(SPU::LNOP));
    }
    isEvenPlace=true;
  }
}

FunctionPass *llvm::createSPUNopFillerPass(SPUTargetMachine &tm) {
  return new SPUNopFiller(tm);
}

// Figure out if 'instr' is executed in the even or odd pipeline
SPUNopFiller::SPUOpPlace 
SPUNopFiller::getOpPlacement( MachineInstr &instr ) {
  int sc = instr.getDesc().getSchedClass();
  const InstrStage *stage = IID->beginStage(sc);
  unsigned FUs = stage->getUnits();
  SPUOpPlace retval;

  switch( FUs ) {
    case 0: retval = pseudo; break;
    case 1: retval = odd;    break;
    case 2: retval = even;   break;
    default: retval= pseudo; 
             assert( false && "got unknown FuncUnit\n");
             break;
  };
  return retval;
}
