//===--------------------- InstrBuilder.cpp ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the InstrBuilder interface.
///
//===----------------------------------------------------------------------===//

#include "InstrBuilder.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "llvm-mca"

namespace mca {

using namespace llvm;

static void
initializeUsedResources(InstrDesc &ID, const MCSchedClassDesc &SCDesc,
                        const MCSubtargetInfo &STI,
                        ArrayRef<uint64_t> ProcResourceMasks) {
  const MCSchedModel &SM = STI.getSchedModel();

  // Populate resources consumed.
  using ResourcePlusCycles = std::pair<uint64_t, ResourceUsage>;
  std::vector<ResourcePlusCycles> Worklist;
  for (unsigned I = 0, E = SCDesc.NumWriteProcResEntries; I < E; ++I) {
    const MCWriteProcResEntry *PRE = STI.getWriteProcResBegin(&SCDesc) + I;
    const MCProcResourceDesc &PR = *SM.getProcResource(PRE->ProcResourceIdx);
    uint64_t Mask = ProcResourceMasks[PRE->ProcResourceIdx];
    if (PR.BufferSize != -1)
      ID.Buffers.push_back(Mask);
    CycleSegment RCy(0, PRE->Cycles, false);
    Worklist.emplace_back(ResourcePlusCycles(Mask, ResourceUsage(RCy)));
  }

  // Sort elements by mask popcount, so that we prioritize resource units over
  // resource groups, and smaller groups over larger groups.
  std::sort(Worklist.begin(), Worklist.end(),
            [](const ResourcePlusCycles &A, const ResourcePlusCycles &B) {
              unsigned popcntA = countPopulation(A.first);
              unsigned popcntB = countPopulation(B.first);
              if (popcntA < popcntB)
                return true;
              if (popcntA > popcntB)
                return false;
              return A.first < B.first;
            });

  uint64_t UsedResourceUnits = 0;

  // Remove cycles contributed by smaller resources.
  for (unsigned I = 0, E = Worklist.size(); I < E; ++I) {
    ResourcePlusCycles &A = Worklist[I];
    if (!A.second.size()) {
      A.second.NumUnits = 0;
      A.second.setReserved();
      ID.Resources.emplace_back(A);
      continue;
    }

    ID.Resources.emplace_back(A);
    uint64_t NormalizedMask = A.first;
    if (countPopulation(A.first) == 1) {
      UsedResourceUnits |= A.first;
    } else {
      // Remove the leading 1 from the resource group mask.
      NormalizedMask ^= PowerOf2Floor(NormalizedMask);
    }

    for (unsigned J = I + 1; J < E; ++J) {
      ResourcePlusCycles &B = Worklist[J];
      if ((NormalizedMask & B.first) == NormalizedMask) {
        B.second.CS.Subtract(A.second.size());
        if (countPopulation(B.first) > 1)
          B.second.NumUnits++;
      }
    }
  }

  // A SchedWrite may specify a number of cycles in which a resource group
  // is reserved. For example (on target x86; cpu Haswell):
  //
  //  SchedWriteRes<[HWPort0, HWPort1, HWPort01]> {
  //    let ResourceCycles = [2, 2, 3];
  //  }
  //
  // This means:
  // Resource units HWPort0 and HWPort1 are both used for 2cy.
  // Resource group HWPort01 is the union of HWPort0 and HWPort1.
  // Since this write touches both HWPort0 and HWPort1 for 2cy, HWPort01
  // will not be usable for 2 entire cycles from instruction issue.
  //
  // On top of those 2cy, SchedWriteRes explicitly specifies an extra latency
  // of 3 cycles for HWPort01. This tool assumes that the 3cy latency is an
  // extra delay on top of the 2 cycles latency.
  // During those extra cycles, HWPort01 is not usable by other instructions.
  for (ResourcePlusCycles &RPC : ID.Resources) {
    if (countPopulation(RPC.first) > 1 && !RPC.second.isReserved()) {
      // Remove the leading 1 from the resource group mask.
      uint64_t Mask = RPC.first ^ PowerOf2Floor(RPC.first);
      if ((Mask & UsedResourceUnits) == Mask)
        RPC.second.setReserved();
    }
  }

  DEBUG({
    for (const std::pair<uint64_t, ResourceUsage> &R : ID.Resources)
      dbgs() << "\t\tMask=" << R.first << ", cy=" << R.second.size() << '\n';
    for (const uint64_t R : ID.Buffers)
      dbgs() << "\t\tBuffer Mask=" << R << '\n';
  });
}

static void computeMaxLatency(InstrDesc &ID, const MCInstrDesc &MCDesc,
                              const MCSchedClassDesc &SCDesc,
                              const MCSubtargetInfo &STI) {
  if (MCDesc.isCall()) {
    // We cannot estimate how long this call will take.
    // Artificially set an arbitrarily high latency (100cy).
    ID.MaxLatency = 100U;
    return;
  }

  int Latency = MCSchedModel::computeInstrLatency(STI, SCDesc);
  // If latency is unknown, then conservatively assume a MaxLatency of 100cy.
  ID.MaxLatency = Latency < 0 ? 100U : static_cast<unsigned>(Latency);
}

static void populateWrites(InstrDesc &ID, const MCInst &MCI,
                           const MCInstrDesc &MCDesc,
                           const MCSchedClassDesc &SCDesc,
                           const MCSubtargetInfo &STI) {
  computeMaxLatency(ID, MCDesc, SCDesc, STI);

  // Set if writes through this opcode may update super registers.
  // TODO: on x86-64, a 4 byte write of a general purpose register always
  // fully updates the super-register.
  // More in general, (at least on x86) not all register writes perform
  // a partial (super-)register update.
  // For example, an AVX instruction that writes on a XMM register implicitly
  // zeroes the upper half of every aliasing super-register.
  //
  // For now, we pessimistically assume that writes are all potentially
  // partial register updates. This is a good default for most targets, execept
  // for those like x86 which implement a special semantic for certain opcodes.
  // At least on x86, this may lead to an inaccurate prediction of the
  // instruction level parallelism.
  bool FullyUpdatesSuperRegisters = false;

  // Now Populate Writes.

  // This algorithm currently works under the strong (and potentially incorrect)
  // assumption that information related to register def/uses can be obtained
  // from MCInstrDesc.
  //
  // However class MCInstrDesc is used to describe MachineInstr objects and not
  // MCInst objects. To be more specific, MCInstrDesc objects are opcode
  // descriptors that are automatically generated via tablegen based on the
  // instruction set information available from the target .td files.  That
  // means, the number of (explicit) definitions according to MCInstrDesc always
  // matches the cardinality of the `(outs)` set in tablegen.
  //
  // By constructions, definitions must appear first in the operand sequence of
  // a MachineInstr. Also, the (outs) sequence is preserved (example: the first
  // element in the outs set is the first operand in the corresponding
  // MachineInstr).  That's the reason why MCInstrDesc only needs to declare the
  // total number of register definitions, and not where those definitions are
  // in the machine operand sequence.
  //
  // Unfortunately, it is not safe to use the information from MCInstrDesc to
  // also describe MCInst objects. An MCInst object can be obtained from a
  // MachineInstr through a lowering step which may restructure the operand
  // sequence (and even remove or introduce new operands). So, there is a high
  // risk that the lowering step breaks the assumptions that register
  // definitions are always at the beginning of the machine operand sequence.
  //
  // This is a fundamental problem, and it is still an open problem. Essentially
  // we have to find a way to correlate def/use operands of a MachineInstr to
  // operands of an MCInst. Otherwise, we cannot correctly reconstruct data
  // dependencies, nor we can correctly interpret the scheduling model, which
  // heavily uses machine operand indices to define processor read-advance
  // information, and to identify processor write resources.  Essentially, we
  // either need something like a MCInstrDesc, but for MCInst, or a way
  // to map MCInst operands back to MachineInstr operands.
  //
  // Unfortunately, we don't have that information now. So, this prototype
  // currently work under the strong assumption that we can always safely trust
  // the content of an MCInstrDesc.  For example, we can query a MCInstrDesc to
  // obtain the number of explicit and implicit register defintions.  We also
  // assume that register definitions always come first in the operand sequence.
  // This last assumption usually makes sense for MachineInstr, where register
  // definitions always appear at the beginning of the operands sequence. In
  // reality, these assumptions could be broken by the lowering step, which can
  // decide to lay out operands in a different order than the original order of
  // operand as specified by the MachineInstr.
  //
  // Things get even more complicated in the presence of "optional" register
  // definitions. For MachineInstr, optional register definitions are always at
  // the end of the operand sequence. Some ARM instructions that may update the
  // status flags specify that register as a optional operand.  Since we don't
  // have operand descriptors for MCInst, we assume for now that the optional
  // definition is always the last operand of a MCInst.  Again, this assumption
  // may be okay for most targets. However, there is no guarantee that targets
  // would respect that.
  //
  // In conclusion: these are for now the strong assumptions made by the tool:
  //  * The number of explicit and implicit register definitions in a MCInst
  //    matches the number of explicit and implicit definitions according to
  //    the opcode descriptor (MCInstrDesc).
  //  * Register definitions take precedence over register uses in the operands
  //    list.
  //  * If an opcode specifies an optional definition, then the optional
  //    definition is always the last operand in the sequence, and it can be
  //    set to zero (i.e. "no register").
  //
  // These assumptions work quite well for most out-of-order in-tree targets
  // like x86. This is mainly because the vast majority of instructions is
  // expanded to MCInst using a straightforward lowering logic that preserves
  // the ordering of the operands.
  //
  // In the longer term, we need to find a proper solution for this issue.
  unsigned NumExplicitDefs = MCDesc.getNumDefs();
  unsigned NumImplicitDefs = MCDesc.getNumImplicitDefs();
  unsigned NumWriteLatencyEntries = SCDesc.NumWriteLatencyEntries;
  unsigned TotalDefs = NumExplicitDefs + NumImplicitDefs;
  if (MCDesc.hasOptionalDef())
    TotalDefs++;
  ID.Writes.resize(TotalDefs);
  // Iterate over the operands list, and skip non-register operands.
  // The first NumExplictDefs register operands are expected to be register
  // definitions.
  unsigned CurrentDef = 0;
  unsigned i = 0;
  for (; i < MCI.getNumOperands() && CurrentDef < NumExplicitDefs; ++i) {
    const MCOperand &Op = MCI.getOperand(i);
    if (!Op.isReg())
      continue;

    WriteDescriptor &Write = ID.Writes[CurrentDef];
    Write.OpIndex = i;
    if (CurrentDef < NumWriteLatencyEntries) {
      const MCWriteLatencyEntry &WLE =
          *STI.getWriteLatencyEntry(&SCDesc, CurrentDef);
      // Conservatively default to MaxLatency.
      Write.Latency = WLE.Cycles == -1 ? ID.MaxLatency : WLE.Cycles;
      Write.SClassOrWriteResourceID = WLE.WriteResourceID;
    } else {
      // Assign a default latency for this write.
      Write.Latency = ID.MaxLatency;
      Write.SClassOrWriteResourceID = 0;
    }
    Write.FullyUpdatesSuperRegs = FullyUpdatesSuperRegisters;
    Write.IsOptionalDef = false;
    DEBUG({
      dbgs() << "\t\tOpIdx=" << Write.OpIndex << ", Latency=" << Write.Latency
             << ", WriteResourceID=" << Write.SClassOrWriteResourceID << '\n';
    });
    CurrentDef++;
  }

  if (CurrentDef != NumExplicitDefs)
    llvm::report_fatal_error(
        "error: Expected more register operand definitions. ");

  CurrentDef = 0;
  for (CurrentDef = 0; CurrentDef < NumImplicitDefs; ++CurrentDef) {
    unsigned Index = NumExplicitDefs + CurrentDef;
    WriteDescriptor &Write = ID.Writes[Index];
    Write.OpIndex = -1;
    Write.RegisterID = MCDesc.getImplicitDefs()[CurrentDef];
    Write.Latency = ID.MaxLatency;
    Write.SClassOrWriteResourceID = 0;
    Write.IsOptionalDef = false;
    assert(Write.RegisterID != 0 && "Expected a valid phys register!");
    DEBUG(dbgs() << "\t\tOpIdx=" << Write.OpIndex << ", PhysReg="
                 << Write.RegisterID << ", Latency=" << Write.Latency
                 << ", WriteResourceID=" << Write.SClassOrWriteResourceID
                 << '\n');
  }

  if (MCDesc.hasOptionalDef()) {
    // Always assume that the optional definition is the last operand of the
    // MCInst sequence.
    const MCOperand &Op = MCI.getOperand(MCI.getNumOperands() - 1);
    if (i == MCI.getNumOperands() || !Op.isReg())
      llvm::report_fatal_error(
          "error: expected a register operand for an optional "
          "definition. Instruction has not be correctly analyzed.\n",
          false);

    WriteDescriptor &Write = ID.Writes[TotalDefs - 1];
    Write.OpIndex = MCI.getNumOperands() - 1;
    // Assign a default latency for this write.
    Write.Latency = ID.MaxLatency;
    Write.SClassOrWriteResourceID = 0;
    Write.IsOptionalDef = true;
  }
}

static void populateReads(InstrDesc &ID, const MCInst &MCI,
                          const MCInstrDesc &MCDesc,
                          const MCSchedClassDesc &SCDesc,
                          const MCSubtargetInfo &STI) {
  unsigned SchedClassID = MCDesc.getSchedClass();
  bool HasReadAdvanceEntries = SCDesc.NumReadAdvanceEntries > 0;

  unsigned i = 0;
  unsigned NumExplicitDefs = MCDesc.getNumDefs();
  // Skip explicit definitions.
  for (; i < MCI.getNumOperands() && NumExplicitDefs; ++i) {
    const MCOperand &Op = MCI.getOperand(i);
    if (Op.isReg())
      NumExplicitDefs--;
  }

  if (NumExplicitDefs)
    llvm::report_fatal_error(
        "error: Expected more register operand definitions. ", false);

  unsigned NumExplicitUses = MCI.getNumOperands() - i;
  unsigned NumImplicitUses = MCDesc.getNumImplicitUses();
  if (MCDesc.hasOptionalDef()) {
    assert(NumExplicitUses);
    NumExplicitUses--;
  }
  unsigned TotalUses = NumExplicitUses + NumImplicitUses;
  if (!TotalUses)
    return;

  ID.Reads.resize(TotalUses);
  for (unsigned CurrentUse = 0; CurrentUse < NumExplicitUses; ++CurrentUse) {
    ReadDescriptor &Read = ID.Reads[CurrentUse];
    Read.OpIndex = i + CurrentUse;
    Read.HasReadAdvanceEntries = HasReadAdvanceEntries;
    Read.SchedClassID = SchedClassID;
    DEBUG(dbgs() << "\t\tOpIdx=" << Read.OpIndex);
  }

  for (unsigned CurrentUse = 0; CurrentUse < NumImplicitUses; ++CurrentUse) {
    ReadDescriptor &Read = ID.Reads[NumExplicitUses + CurrentUse];
    Read.OpIndex = -1;
    Read.RegisterID = MCDesc.getImplicitUses()[CurrentUse];
    Read.HasReadAdvanceEntries = false;
    Read.SchedClassID = SchedClassID;
    DEBUG(dbgs() << "\t\tOpIdx=" << Read.OpIndex
                 << ", RegisterID=" << Read.RegisterID << '\n');
  }
}

void InstrBuilder::createInstrDescImpl(const MCInst &MCI) {
  assert(STI.getSchedModel().hasInstrSchedModel() &&
         "Itineraries are not yet supported!");

  unsigned short Opcode = MCI.getOpcode();
  // Obtain the instruction descriptor from the opcode.
  const MCInstrDesc &MCDesc = MCII.get(Opcode);
  const MCSchedModel &SM = STI.getSchedModel();

  // Then obtain the scheduling class information from the instruction.
  const MCSchedClassDesc &SCDesc =
      *SM.getSchedClassDesc(MCDesc.getSchedClass());

  // Create a new empty descriptor.
  std::unique_ptr<InstrDesc> ID = llvm::make_unique<InstrDesc>();

  if (SCDesc.isVariant()) {
    errs() << "warning: don't know how to model variant opcodes.\n"
           << "note: assume 1 micro opcode.\n";
    ID->NumMicroOps = 1U;
  } else {
    ID->NumMicroOps = SCDesc.NumMicroOps;
  }

  if (MCDesc.isCall()) {
    // We don't correctly model calls.
    errs() << "warning: found a call in the input assembly sequence.\n"
           << "note: call instructions are not correctly modeled. Assume a "
              "latency of 100cy.\n";
  }

  if (MCDesc.isReturn()) {
    errs() << "warning: found a return instruction in the input assembly "
              "sequence.\n"
           << "note: program counter updates are ignored.\n";
  }

  ID->MayLoad = MCDesc.mayLoad();
  ID->MayStore = MCDesc.mayStore();
  ID->HasSideEffects = MCDesc.hasUnmodeledSideEffects();

  initializeUsedResources(*ID, SCDesc, STI, ProcResourceMasks);
  populateWrites(*ID, MCI, MCDesc, SCDesc, STI);
  populateReads(*ID, MCI, MCDesc, SCDesc, STI);

  DEBUG(dbgs() << "\t\tMaxLatency=" << ID->MaxLatency << '\n');
  DEBUG(dbgs() << "\t\tNumMicroOps=" << ID->NumMicroOps << '\n');

  // Now add the new descriptor.
  Descriptors[Opcode] = std::move(ID);
}

const InstrDesc &InstrBuilder::getOrCreateInstrDesc(const MCInst &MCI) {
  if (Descriptors.find_as(MCI.getOpcode()) == Descriptors.end())
    createInstrDescImpl(MCI);
  return *Descriptors[MCI.getOpcode()];
}

std::unique_ptr<Instruction>
InstrBuilder::createInstruction(unsigned Idx, const MCInst &MCI) {
  const InstrDesc &D = getOrCreateInstrDesc(MCI);
  std::unique_ptr<Instruction> NewIS = llvm::make_unique<Instruction>(D);

  // Populate Reads first.
  for (const ReadDescriptor &RD : D.Reads) {
    int RegID = -1;
    if (RD.OpIndex != -1) {
      // explicit read.
      const MCOperand &Op = MCI.getOperand(RD.OpIndex);
      // Skip non-register operands.
      if (!Op.isReg())
        continue;
      RegID = Op.getReg();
    } else {
      // Implicit read.
      RegID = RD.RegisterID;
    }

    // Skip invalid register operands.
    if (!RegID)
      continue;

    // Okay, this is a register operand. Create a ReadState for it.
    assert(RegID > 0 && "Invalid register ID found!");
    NewIS->getUses().emplace_back(llvm::make_unique<ReadState>(RD, RegID));
  }

  // Now populate writes.
  for (const WriteDescriptor &WD : D.Writes) {
    unsigned RegID =
        WD.OpIndex == -1 ? WD.RegisterID : MCI.getOperand(WD.OpIndex).getReg();
    // Check if this is a optional definition that references NoReg.
    if (WD.IsOptionalDef && !RegID)
      continue;

    assert(RegID && "Expected a valid register ID!");
    NewIS->getDefs().emplace_back(llvm::make_unique<WriteState>(WD, RegID));
  }

  return NewIS;
}
} // namespace mca
