//===-- ARMSchedule.td - ARM Scheduling Definitions --------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Instruction scheduling annotations for out-of-order CPUs.
// These annotations are independent of the itinerary class defined below.
// Here we define the subtarget independent read/write per-operand resources.
// The subtarget schedule definitions will then map these to the subtarget's
// resource usages.
// For example:
// The instruction cycle timings table might contain an entry for an operation
// like the following:
// Rd <- ADD Rn, Rm, <shift> Rs
//  Uops | Latency from register | Uops - resource requirements - latency
//  2    | Rn: 1 Rm: 4 Rs: 4     | uop T0, Rm, Rs - P01 - 3
//       |                       | uopc Rd, Rn, T0 -  P01 - 1
// This is telling us that the result will be available in destination register
// Rd after a minimum of three cycles after the result in Rm and Rs is available
// and one cycle after the result in Rn is available. The micro-ops can execute
// on resource P01.
// To model this, we need to express that we need to dispatch two micro-ops,
// that the resource P01 is needed and that the latency to Rn is different than
// the latency to Rm and Rs. The scheduler can decrease Rn's producer latency by
// two.
// We will do this by assigning (abstract) resources to register defs/uses.
// ARMSchedule.td:
//   def WriteALUsr : SchedWrite;
//   def ReadAdvanceALUsr : ScheRead;
//
// ARMInstrInfo.td:
//   def ADDrs : I<>, Sched<[WriteALUsr, ReadAdvanceALUsr, ReadDefault,
//                           ReadDefault]> { ...}
// ReadAdvance read resources allow us to define "pipeline by-passes" or
// shorter latencies to certain registers as needed in the example above.
// The "ReadDefault" can be omitted.
// Next, the subtarget td file assigns resources to the abstract resources
// defined here.
// ARMScheduleSubtarget.td:
//  // Resources.
//  def P01 : ProcResource<3>; // ALU unit (3 of it).
//  ...
//  // Resource usages.
//  def : WriteRes<WriteALUsr, [P01, P01]> {
//    Latency = 4; // Latency of 4.
//    NumMicroOps = 2; // Dispatch 2 micro-ops.
//    // The two instances of resource P01 are occupied for one cycle. It is one
//    // cycle because these resources happen to be pipelined.
//    ResourceCycles = [1, 1];
//  }
//  def : ReadAdvance<ReadAdvanceALUsr, 3>;

// Basic ALU operation.
def WriteALU : SchedWrite;
def ReadALU : SchedRead;

// Basic ALU with shifts.
def WriteALUsi : SchedWrite; // Shift by immediate.
def WriteALUsr : SchedWrite; // Shift by register.
def WriteALUSsr : SchedWrite; // Shift by register (flag setting).
def ReadALUsr : SchedRead; // Some operands are read later.

// Compares.
def WriteCMP : SchedWrite;
def WriteCMPsi : SchedWrite;
def WriteCMPsr : SchedWrite;

// Division.
def WriteDiv : SchedWrite;

// Loads.
def WriteLd : SchedWrite;
def WritePreLd : SchedWrite;

// Branches.
def WriteBr : SchedWrite;
def WriteBrL : SchedWrite;
def WriteBrTbl : SchedWrite;

// Fixpoint conversions.
def WriteCvtFP : SchedWrite;

// Noop.
def WriteNoop : SchedWrite;

// Define TII for use in SchedVariant Predicates.
def : PredicateProlog<[{
  const ARMBaseInstrInfo *TII =
    static_cast<const ARMBaseInstrInfo*>(SchedModel->getInstrInfo());
  (void)TII;
}]>;

def IsPredicatedPred : SchedPredicate<[{TII->isPredicated(MI)}]>;

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for ARM
//
def IIC_iALUx      : InstrItinClass;
def IIC_iALUi      : InstrItinClass;
def IIC_iALUr      : InstrItinClass;
def IIC_iALUsi     : InstrItinClass;
def IIC_iALUsir    : InstrItinClass;
def IIC_iALUsr     : InstrItinClass;
def IIC_iBITi      : InstrItinClass;
def IIC_iBITr      : InstrItinClass;
def IIC_iBITsi     : InstrItinClass;
def IIC_iBITsr     : InstrItinClass;
def IIC_iUNAr      : InstrItinClass;
def IIC_iUNAsi     : InstrItinClass;
def IIC_iEXTr      : InstrItinClass;
def IIC_iEXTAr     : InstrItinClass;
def IIC_iEXTAsr    : InstrItinClass;
def IIC_iCMPi      : InstrItinClass;
def IIC_iCMPr      : InstrItinClass;
def IIC_iCMPsi     : InstrItinClass;
def IIC_iCMPsr     : InstrItinClass;
def IIC_iTSTi      : InstrItinClass;
def IIC_iTSTr      : InstrItinClass;
def IIC_iTSTsi     : InstrItinClass;
def IIC_iTSTsr     : InstrItinClass;
def IIC_iMOVi      : InstrItinClass;
def IIC_iMOVr      : InstrItinClass;
def IIC_iMOVsi     : InstrItinClass;
def IIC_iMOVsr     : InstrItinClass;
def IIC_iMOVix2    : InstrItinClass;
def IIC_iMOVix2addpc : InstrItinClass;
def IIC_iMOVix2ld  : InstrItinClass;
def IIC_iMVNi      : InstrItinClass;
def IIC_iMVNr      : InstrItinClass;
def IIC_iMVNsi     : InstrItinClass;
def IIC_iMVNsr     : InstrItinClass;
def IIC_iCMOVi     : InstrItinClass;
def IIC_iCMOVr     : InstrItinClass;
def IIC_iCMOVsi    : InstrItinClass;
def IIC_iCMOVsr    : InstrItinClass;
def IIC_iCMOVix2   : InstrItinClass;
def IIC_iMUL16     : InstrItinClass;
def IIC_iMAC16     : InstrItinClass;
def IIC_iMUL32     : InstrItinClass;
def IIC_iMAC32     : InstrItinClass;
def IIC_iMUL64     : InstrItinClass;
def IIC_iMAC64     : InstrItinClass;
def IIC_iDIV     : InstrItinClass;
def IIC_iLoad_i    : InstrItinClass;
def IIC_iLoad_r    : InstrItinClass;
def IIC_iLoad_si   : InstrItinClass;
def IIC_iLoad_iu   : InstrItinClass;
def IIC_iLoad_ru   : InstrItinClass;
def IIC_iLoad_siu  : InstrItinClass;
def IIC_iLoad_bh_i   : InstrItinClass;
def IIC_iLoad_bh_r   : InstrItinClass;
def IIC_iLoad_bh_si  : InstrItinClass;
def IIC_iLoad_bh_iu  : InstrItinClass;
def IIC_iLoad_bh_ru  : InstrItinClass;
def IIC_iLoad_bh_siu : InstrItinClass;
def IIC_iLoad_d_i  : InstrItinClass;
def IIC_iLoad_d_r  : InstrItinClass;
def IIC_iLoad_d_ru : InstrItinClass;
def IIC_iLoad_m    : InstrItinClass;
def IIC_iLoad_mu   : InstrItinClass;
def IIC_iLoad_mBr  : InstrItinClass;
def IIC_iPop       : InstrItinClass;
def IIC_iPop_Br    : InstrItinClass;
def IIC_iLoadiALU  : InstrItinClass;
def IIC_iStore_i   : InstrItinClass;
def IIC_iStore_r   : InstrItinClass;
def IIC_iStore_si  : InstrItinClass;
def IIC_iStore_iu  : InstrItinClass;
def IIC_iStore_ru  : InstrItinClass;
def IIC_iStore_siu : InstrItinClass;
def IIC_iStore_bh_i   : InstrItinClass;
def IIC_iStore_bh_r   : InstrItinClass;
def IIC_iStore_bh_si  : InstrItinClass;
def IIC_iStore_bh_iu  : InstrItinClass;
def IIC_iStore_bh_ru  : InstrItinClass;
def IIC_iStore_bh_siu : InstrItinClass;
def IIC_iStore_d_i   : InstrItinClass;
def IIC_iStore_d_r   : InstrItinClass;
def IIC_iStore_d_ru  : InstrItinClass;
def IIC_iStore_m   : InstrItinClass;
def IIC_iStore_mu  : InstrItinClass;
def IIC_Preload    : InstrItinClass;
def IIC_Br         : InstrItinClass;
def IIC_fpSTAT     : InstrItinClass;
def IIC_fpUNA32    : InstrItinClass;
def IIC_fpUNA64    : InstrItinClass;
def IIC_fpCMP32    : InstrItinClass;
def IIC_fpCMP64    : InstrItinClass;
def IIC_fpCVTSD    : InstrItinClass;
def IIC_fpCVTDS    : InstrItinClass;
def IIC_fpCVTSH    : InstrItinClass;
def IIC_fpCVTHS    : InstrItinClass;
def IIC_fpCVTIS    : InstrItinClass;
def IIC_fpCVTID    : InstrItinClass;
def IIC_fpCVTSI    : InstrItinClass;
def IIC_fpCVTDI    : InstrItinClass;
def IIC_fpMOVIS    : InstrItinClass;
def IIC_fpMOVID    : InstrItinClass;
def IIC_fpMOVSI    : InstrItinClass;
def IIC_fpMOVDI    : InstrItinClass;
def IIC_fpALU32    : InstrItinClass;
def IIC_fpALU64    : InstrItinClass;
def IIC_fpMUL32    : InstrItinClass;
def IIC_fpMUL64    : InstrItinClass;
def IIC_fpMAC32    : InstrItinClass;
def IIC_fpMAC64    : InstrItinClass;
def IIC_fpFMAC32   : InstrItinClass;
def IIC_fpFMAC64   : InstrItinClass;
def IIC_fpDIV32    : InstrItinClass;
def IIC_fpDIV64    : InstrItinClass;
def IIC_fpSQRT32   : InstrItinClass;
def IIC_fpSQRT64   : InstrItinClass;
def IIC_fpLoad32   : InstrItinClass;
def IIC_fpLoad64   : InstrItinClass;
def IIC_fpLoad_m   : InstrItinClass;
def IIC_fpLoad_mu  : InstrItinClass;
def IIC_fpStore32  : InstrItinClass;
def IIC_fpStore64  : InstrItinClass;
def IIC_fpStore_m  : InstrItinClass;
def IIC_fpStore_mu : InstrItinClass;
def IIC_VLD1       : InstrItinClass;
def IIC_VLD1x2     : InstrItinClass;
def IIC_VLD1x3     : InstrItinClass;
def IIC_VLD1x4     : InstrItinClass;
def IIC_VLD1u      : InstrItinClass;
def IIC_VLD1x2u    : InstrItinClass;
def IIC_VLD1x3u    : InstrItinClass;
def IIC_VLD1x4u    : InstrItinClass;
def IIC_VLD1ln     : InstrItinClass;
def IIC_VLD1lnu    : InstrItinClass;
def IIC_VLD1dup    : InstrItinClass;
def IIC_VLD1dupu   : InstrItinClass;
def IIC_VLD2       : InstrItinClass;
def IIC_VLD2x2     : InstrItinClass;
def IIC_VLD2u      : InstrItinClass;
def IIC_VLD2x2u    : InstrItinClass;
def IIC_VLD2ln     : InstrItinClass;
def IIC_VLD2lnu    : InstrItinClass;
def IIC_VLD2dup    : InstrItinClass;
def IIC_VLD2dupu   : InstrItinClass;
def IIC_VLD3       : InstrItinClass;
def IIC_VLD3ln     : InstrItinClass;
def IIC_VLD3u      : InstrItinClass;
def IIC_VLD3lnu    : InstrItinClass;
def IIC_VLD3dup    : InstrItinClass;
def IIC_VLD3dupu   : InstrItinClass;
def IIC_VLD4       : InstrItinClass;
def IIC_VLD4ln     : InstrItinClass;
def IIC_VLD4u      : InstrItinClass;
def IIC_VLD4lnu    : InstrItinClass;
def IIC_VLD4dup    : InstrItinClass;
def IIC_VLD4dupu   : InstrItinClass;
def IIC_VST1       : InstrItinClass;
def IIC_VST1x2     : InstrItinClass;
def IIC_VST1x3     : InstrItinClass;
def IIC_VST1x4     : InstrItinClass;
def IIC_VST1u      : InstrItinClass;
def IIC_VST1x2u    : InstrItinClass;
def IIC_VST1x3u    : InstrItinClass;
def IIC_VST1x4u    : InstrItinClass;
def IIC_VST1ln     : InstrItinClass;
def IIC_VST1lnu    : InstrItinClass;
def IIC_VST2       : InstrItinClass;
def IIC_VST2x2     : InstrItinClass;
def IIC_VST2u      : InstrItinClass;
def IIC_VST2x2u    : InstrItinClass;
def IIC_VST2ln     : InstrItinClass;
def IIC_VST2lnu    : InstrItinClass;
def IIC_VST3       : InstrItinClass;
def IIC_VST3u      : InstrItinClass;
def IIC_VST3ln     : InstrItinClass;
def IIC_VST3lnu    : InstrItinClass;
def IIC_VST4       : InstrItinClass;
def IIC_VST4u      : InstrItinClass;
def IIC_VST4ln     : InstrItinClass;
def IIC_VST4lnu    : InstrItinClass;
def IIC_VUNAD      : InstrItinClass;
def IIC_VUNAQ      : InstrItinClass;
def IIC_VBIND      : InstrItinClass;
def IIC_VBINQ      : InstrItinClass;
def IIC_VPBIND     : InstrItinClass;
def IIC_VFMULD     : InstrItinClass;
def IIC_VFMULQ     : InstrItinClass;
def IIC_VMOV       : InstrItinClass;
def IIC_VMOVImm    : InstrItinClass;
def IIC_VMOVD      : InstrItinClass;
def IIC_VMOVQ      : InstrItinClass;
def IIC_VMOVIS     : InstrItinClass;
def IIC_VMOVID     : InstrItinClass;
def IIC_VMOVISL    : InstrItinClass;
def IIC_VMOVSI     : InstrItinClass;
def IIC_VMOVDI     : InstrItinClass;
def IIC_VMOVN      : InstrItinClass;
def IIC_VPERMD     : InstrItinClass;
def IIC_VPERMQ     : InstrItinClass;
def IIC_VPERMQ3    : InstrItinClass;
def IIC_VMACD      : InstrItinClass;
def IIC_VMACQ      : InstrItinClass;
def IIC_VFMACD     : InstrItinClass;
def IIC_VFMACQ     : InstrItinClass;
def IIC_VRECSD     : InstrItinClass;
def IIC_VRECSQ     : InstrItinClass;
def IIC_VCNTiD     : InstrItinClass;
def IIC_VCNTiQ     : InstrItinClass;
def IIC_VUNAiD     : InstrItinClass;
def IIC_VUNAiQ     : InstrItinClass;
def IIC_VQUNAiD    : InstrItinClass;
def IIC_VQUNAiQ    : InstrItinClass;
def IIC_VBINiD     : InstrItinClass;
def IIC_VBINiQ     : InstrItinClass;
def IIC_VSUBiD     : InstrItinClass;
def IIC_VSUBiQ     : InstrItinClass;
def IIC_VBINi4D    : InstrItinClass;
def IIC_VBINi4Q    : InstrItinClass;
def IIC_VSUBi4D    : InstrItinClass;
def IIC_VSUBi4Q    : InstrItinClass;
def IIC_VABAD      : InstrItinClass;
def IIC_VABAQ      : InstrItinClass;
def IIC_VSHLiD     : InstrItinClass;
def IIC_VSHLiQ     : InstrItinClass;
def IIC_VSHLi4D    : InstrItinClass;
def IIC_VSHLi4Q    : InstrItinClass;
def IIC_VPALiD     : InstrItinClass;
def IIC_VPALiQ     : InstrItinClass;
def IIC_VMULi16D   : InstrItinClass;
def IIC_VMULi32D   : InstrItinClass;
def IIC_VMULi16Q   : InstrItinClass;
def IIC_VMULi32Q   : InstrItinClass;
def IIC_VMACi16D   : InstrItinClass;
def IIC_VMACi32D   : InstrItinClass;
def IIC_VMACi16Q   : InstrItinClass;
def IIC_VMACi32Q   : InstrItinClass;
def IIC_VEXTD      : InstrItinClass;
def IIC_VEXTQ      : InstrItinClass;
def IIC_VTB1       : InstrItinClass;
def IIC_VTB2       : InstrItinClass;
def IIC_VTB3       : InstrItinClass;
def IIC_VTB4       : InstrItinClass;
def IIC_VTBX1      : InstrItinClass;
def IIC_VTBX2      : InstrItinClass;
def IIC_VTBX3      : InstrItinClass;
def IIC_VTBX4      : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

include "ARMScheduleV6.td"
include "ARMScheduleA8.td"
include "ARMScheduleA9.td"
include "ARMScheduleSwift.td"
