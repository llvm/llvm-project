//===-- X86InstrFoldTables.cpp - X86 Instruction Folding Tables -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 memory folding tables.
//
//===----------------------------------------------------------------------===//

#include "X86InstrFoldTables.h"
#include "X86InstrInfo.h"
#include "llvm/ADT/STLExtras.h"
#include <atomic>
#include <vector>

using namespace llvm;

// These tables are sorted by their RegOp value allowing them to be binary
// searched at runtime without the need for additional storage. The enum values
// are currently emitted in X86GenInstrInfo.inc in alphabetical order. Which
// makes sorting these tables a simple matter of alphabetizing the table.
#include "X86GenFoldTables.inc"
static const X86MemoryFoldTableEntry BroadcastFoldTable2[] = {
  { X86::VADDPDZ128rr,   X86::VADDPDZ128rmb,   TB_BCAST_SD },
  { X86::VADDPDZ256rr,   X86::VADDPDZ256rmb,   TB_BCAST_SD },
  { X86::VADDPDZrr,      X86::VADDPDZrmb,      TB_BCAST_SD },
  { X86::VADDPSZ128rr,   X86::VADDPSZ128rmb,   TB_BCAST_SS },
  { X86::VADDPSZ256rr,   X86::VADDPSZ256rmb,   TB_BCAST_SS },
  { X86::VADDPSZrr,      X86::VADDPSZrmb,      TB_BCAST_SS },
  { X86::VANDNPDZ128rr,  X86::VANDNPDZ128rmb,  TB_BCAST_SD },
  { X86::VANDNPDZ256rr,  X86::VANDNPDZ256rmb,  TB_BCAST_SD },
  { X86::VANDNPDZrr,     X86::VANDNPDZrmb,     TB_BCAST_SD },
  { X86::VANDNPSZ128rr,  X86::VANDNPSZ128rmb,  TB_BCAST_SS },
  { X86::VANDNPSZ256rr,  X86::VANDNPSZ256rmb,  TB_BCAST_SS },
  { X86::VANDNPSZrr,     X86::VANDNPSZrmb,     TB_BCAST_SS },
  { X86::VANDPDZ128rr,   X86::VANDPDZ128rmb,   TB_BCAST_SD },
  { X86::VANDPDZ256rr,   X86::VANDPDZ256rmb,   TB_BCAST_SD },
  { X86::VANDPDZrr,      X86::VANDPDZrmb,      TB_BCAST_SD },
  { X86::VANDPSZ128rr,   X86::VANDPSZ128rmb,   TB_BCAST_SS },
  { X86::VANDPSZ256rr,   X86::VANDPSZ256rmb,   TB_BCAST_SS },
  { X86::VANDPSZrr,      X86::VANDPSZrmb,      TB_BCAST_SS },
  { X86::VCMPPDZ128rri,  X86::VCMPPDZ128rmbi,  TB_BCAST_SD },
  { X86::VCMPPDZ256rri,  X86::VCMPPDZ256rmbi,  TB_BCAST_SD },
  { X86::VCMPPDZrri,     X86::VCMPPDZrmbi,     TB_BCAST_SD },
  { X86::VCMPPSZ128rri,  X86::VCMPPSZ128rmbi,  TB_BCAST_SS },
  { X86::VCMPPSZ256rri,  X86::VCMPPSZ256rmbi,  TB_BCAST_SS },
  { X86::VCMPPSZrri,     X86::VCMPPSZrmbi,     TB_BCAST_SS },
  { X86::VDIVPDZ128rr,   X86::VDIVPDZ128rmb,   TB_BCAST_SD },
  { X86::VDIVPDZ256rr,   X86::VDIVPDZ256rmb,   TB_BCAST_SD },
  { X86::VDIVPDZrr,      X86::VDIVPDZrmb,      TB_BCAST_SD },
  { X86::VDIVPSZ128rr,   X86::VDIVPSZ128rmb,   TB_BCAST_SS },
  { X86::VDIVPSZ256rr,   X86::VDIVPSZ256rmb,   TB_BCAST_SS },
  { X86::VDIVPSZrr,      X86::VDIVPSZrmb,      TB_BCAST_SS },
  { X86::VMAXCPDZ128rr,  X86::VMAXCPDZ128rmb,  TB_BCAST_SD },
  { X86::VMAXCPDZ256rr,  X86::VMAXCPDZ256rmb,  TB_BCAST_SD },
  { X86::VMAXCPDZrr,     X86::VMAXCPDZrmb,     TB_BCAST_SD },
  { X86::VMAXCPSZ128rr,  X86::VMAXCPSZ128rmb,  TB_BCAST_SS },
  { X86::VMAXCPSZ256rr,  X86::VMAXCPSZ256rmb,  TB_BCAST_SS },
  { X86::VMAXCPSZrr,     X86::VMAXCPSZrmb,     TB_BCAST_SS },
  { X86::VMAXPDZ128rr,   X86::VMAXPDZ128rmb,   TB_BCAST_SD },
  { X86::VMAXPDZ256rr,   X86::VMAXPDZ256rmb,   TB_BCAST_SD },
  { X86::VMAXPDZrr,      X86::VMAXPDZrmb,      TB_BCAST_SD },
  { X86::VMAXPSZ128rr,   X86::VMAXPSZ128rmb,   TB_BCAST_SS },
  { X86::VMAXPSZ256rr,   X86::VMAXPSZ256rmb,   TB_BCAST_SS },
  { X86::VMAXPSZrr,      X86::VMAXPSZrmb,      TB_BCAST_SS },
  { X86::VMINCPDZ128rr,  X86::VMINCPDZ128rmb,  TB_BCAST_SD },
  { X86::VMINCPDZ256rr,  X86::VMINCPDZ256rmb,  TB_BCAST_SD },
  { X86::VMINCPDZrr,     X86::VMINCPDZrmb,     TB_BCAST_SD },
  { X86::VMINCPSZ128rr,  X86::VMINCPSZ128rmb,  TB_BCAST_SS },
  { X86::VMINCPSZ256rr,  X86::VMINCPSZ256rmb,  TB_BCAST_SS },
  { X86::VMINCPSZrr,     X86::VMINCPSZrmb,     TB_BCAST_SS },
  { X86::VMINPDZ128rr,   X86::VMINPDZ128rmb,   TB_BCAST_SD },
  { X86::VMINPDZ256rr,   X86::VMINPDZ256rmb,   TB_BCAST_SD },
  { X86::VMINPDZrr,      X86::VMINPDZrmb,      TB_BCAST_SD },
  { X86::VMINPSZ128rr,   X86::VMINPSZ128rmb,   TB_BCAST_SS },
  { X86::VMINPSZ256rr,   X86::VMINPSZ256rmb,   TB_BCAST_SS },
  { X86::VMINPSZrr,      X86::VMINPSZrmb,      TB_BCAST_SS },
  { X86::VMULPDZ128rr,   X86::VMULPDZ128rmb,   TB_BCAST_SD },
  { X86::VMULPDZ256rr,   X86::VMULPDZ256rmb,   TB_BCAST_SD },
  { X86::VMULPDZrr,      X86::VMULPDZrmb,      TB_BCAST_SD },
  { X86::VMULPSZ128rr,   X86::VMULPSZ128rmb,   TB_BCAST_SS },
  { X86::VMULPSZ256rr,   X86::VMULPSZ256rmb,   TB_BCAST_SS },
  { X86::VMULPSZrr,      X86::VMULPSZrmb,      TB_BCAST_SS },
  { X86::VORPDZ128rr,    X86::VORPDZ128rmb,    TB_BCAST_SD },
  { X86::VORPDZ256rr,    X86::VORPDZ256rmb,    TB_BCAST_SD },
  { X86::VORPDZrr,       X86::VORPDZrmb,       TB_BCAST_SD },
  { X86::VORPSZ128rr,    X86::VORPSZ128rmb,    TB_BCAST_SS },
  { X86::VORPSZ256rr,    X86::VORPSZ256rmb,    TB_BCAST_SS },
  { X86::VORPSZrr,       X86::VORPSZrmb,       TB_BCAST_SS },
  { X86::VPADDDZ128rr,   X86::VPADDDZ128rmb,   TB_BCAST_D },
  { X86::VPADDDZ256rr,   X86::VPADDDZ256rmb,   TB_BCAST_D },
  { X86::VPADDDZrr,      X86::VPADDDZrmb,      TB_BCAST_D },
  { X86::VPADDQZ128rr,   X86::VPADDQZ128rmb,   TB_BCAST_Q },
  { X86::VPADDQZ256rr,   X86::VPADDQZ256rmb,   TB_BCAST_Q },
  { X86::VPADDQZrr,      X86::VPADDQZrmb,      TB_BCAST_Q },
  { X86::VPANDDZ128rr,   X86::VPANDDZ128rmb,   TB_BCAST_D },
  { X86::VPANDDZ256rr,   X86::VPANDDZ256rmb,   TB_BCAST_D },
  { X86::VPANDDZrr,      X86::VPANDDZrmb,      TB_BCAST_D },
  { X86::VPANDNDZ128rr,  X86::VPANDNDZ128rmb,  TB_BCAST_D },
  { X86::VPANDNDZ256rr,  X86::VPANDNDZ256rmb,  TB_BCAST_D },
  { X86::VPANDNDZrr,     X86::VPANDNDZrmb,     TB_BCAST_D },
  { X86::VPANDNQZ128rr,  X86::VPANDNQZ128rmb,  TB_BCAST_Q },
  { X86::VPANDNQZ256rr,  X86::VPANDNQZ256rmb,  TB_BCAST_Q },
  { X86::VPANDNQZrr,     X86::VPANDNQZrmb,     TB_BCAST_Q },
  { X86::VPANDQZ128rr,   X86::VPANDQZ128rmb,   TB_BCAST_Q },
  { X86::VPANDQZ256rr,   X86::VPANDQZ256rmb,   TB_BCAST_Q },
  { X86::VPANDQZrr,      X86::VPANDQZrmb,      TB_BCAST_Q },
  { X86::VPCMPDZ128rri,  X86::VPCMPDZ128rmib,  TB_BCAST_D },
  { X86::VPCMPDZ256rri,  X86::VPCMPDZ256rmib,  TB_BCAST_D },
  { X86::VPCMPDZrri,     X86::VPCMPDZrmib,     TB_BCAST_D },
  { X86::VPCMPEQDZ128rr, X86::VPCMPEQDZ128rmb, TB_BCAST_D },
  { X86::VPCMPEQDZ256rr, X86::VPCMPEQDZ256rmb, TB_BCAST_D },
  { X86::VPCMPEQDZrr,    X86::VPCMPEQDZrmb,    TB_BCAST_D },
  { X86::VPCMPEQQZ128rr, X86::VPCMPEQQZ128rmb, TB_BCAST_Q },
  { X86::VPCMPEQQZ256rr, X86::VPCMPEQQZ256rmb, TB_BCAST_Q },
  { X86::VPCMPEQQZrr,    X86::VPCMPEQQZrmb,    TB_BCAST_Q },
  { X86::VPCMPGTDZ128rr, X86::VPCMPGTDZ128rmb, TB_BCAST_D },
  { X86::VPCMPGTDZ256rr, X86::VPCMPGTDZ256rmb, TB_BCAST_D },
  { X86::VPCMPGTDZrr,    X86::VPCMPGTDZrmb,    TB_BCAST_D },
  { X86::VPCMPGTQZ128rr, X86::VPCMPGTQZ128rmb, TB_BCAST_Q },
  { X86::VPCMPGTQZ256rr, X86::VPCMPGTQZ256rmb, TB_BCAST_Q },
  { X86::VPCMPGTQZrr,    X86::VPCMPGTQZrmb,    TB_BCAST_Q },
  { X86::VPCMPQZ128rri,  X86::VPCMPQZ128rmib,  TB_BCAST_Q },
  { X86::VPCMPQZ256rri,  X86::VPCMPQZ256rmib,  TB_BCAST_Q },
  { X86::VPCMPQZrri,     X86::VPCMPQZrmib,     TB_BCAST_Q },
  { X86::VPCMPUDZ128rri, X86::VPCMPUDZ128rmib, TB_BCAST_D },
  { X86::VPCMPUDZ256rri, X86::VPCMPUDZ256rmib, TB_BCAST_D },
  { X86::VPCMPUDZrri,    X86::VPCMPUDZrmib,    TB_BCAST_D },
  { X86::VPCMPUQZ128rri, X86::VPCMPUQZ128rmib, TB_BCAST_Q },
  { X86::VPCMPUQZ256rri, X86::VPCMPUQZ256rmib, TB_BCAST_Q },
  { X86::VPCMPUQZrri,    X86::VPCMPUQZrmib,    TB_BCAST_Q },
  { X86::VPMAXSDZ128rr,  X86::VPMAXSDZ128rmb,  TB_BCAST_D },
  { X86::VPMAXSDZ256rr,  X86::VPMAXSDZ256rmb,  TB_BCAST_D },
  { X86::VPMAXSDZrr,     X86::VPMAXSDZrmb,     TB_BCAST_D },
  { X86::VPMAXSQZ128rr,  X86::VPMAXSQZ128rmb,  TB_BCAST_Q },
  { X86::VPMAXSQZ256rr,  X86::VPMAXSQZ256rmb,  TB_BCAST_Q },
  { X86::VPMAXSQZrr,     X86::VPMAXSQZrmb,     TB_BCAST_Q },
  { X86::VPMAXUDZ128rr,  X86::VPMAXUDZ128rmb,  TB_BCAST_D },
  { X86::VPMAXUDZ256rr,  X86::VPMAXUDZ256rmb,  TB_BCAST_D },
  { X86::VPMAXUDZrr,     X86::VPMAXUDZrmb,     TB_BCAST_D },
  { X86::VPMAXUQZ128rr,  X86::VPMAXUQZ128rmb,  TB_BCAST_Q },
  { X86::VPMAXUQZ256rr,  X86::VPMAXUQZ256rmb,  TB_BCAST_Q },
  { X86::VPMAXUQZrr,     X86::VPMAXUQZrmb,     TB_BCAST_Q },
  { X86::VPMINSDZ128rr,  X86::VPMINSDZ128rmb,  TB_BCAST_D },
  { X86::VPMINSDZ256rr,  X86::VPMINSDZ256rmb,  TB_BCAST_D },
  { X86::VPMINSDZrr,     X86::VPMINSDZrmb,     TB_BCAST_D },
  { X86::VPMINSQZ128rr,  X86::VPMINSQZ128rmb,  TB_BCAST_Q },
  { X86::VPMINSQZ256rr,  X86::VPMINSQZ256rmb,  TB_BCAST_Q },
  { X86::VPMINSQZrr,     X86::VPMINSQZrmb,     TB_BCAST_Q },
  { X86::VPMINUDZ128rr,  X86::VPMINUDZ128rmb,  TB_BCAST_D },
  { X86::VPMINUDZ256rr,  X86::VPMINUDZ256rmb,  TB_BCAST_D },
  { X86::VPMINUDZrr,     X86::VPMINUDZrmb,     TB_BCAST_D },
  { X86::VPMINUQZ128rr,  X86::VPMINUQZ128rmb,  TB_BCAST_Q },
  { X86::VPMINUQZ256rr,  X86::VPMINUQZ256rmb,  TB_BCAST_Q },
  { X86::VPMINUQZrr,     X86::VPMINUQZrmb,     TB_BCAST_Q },
  { X86::VPMULLDZ128rr,  X86::VPMULLDZ128rmb,  TB_BCAST_D },
  { X86::VPMULLDZ256rr,  X86::VPMULLDZ256rmb,  TB_BCAST_D },
  { X86::VPMULLDZrr,     X86::VPMULLDZrmb,     TB_BCAST_D },
  { X86::VPMULLQZ128rr,  X86::VPMULLQZ128rmb,  TB_BCAST_Q },
  { X86::VPMULLQZ256rr,  X86::VPMULLQZ256rmb,  TB_BCAST_Q },
  { X86::VPMULLQZrr,     X86::VPMULLQZrmb,     TB_BCAST_Q },
  { X86::VPORDZ128rr,    X86::VPORDZ128rmb,    TB_BCAST_D },
  { X86::VPORDZ256rr,    X86::VPORDZ256rmb,    TB_BCAST_D },
  { X86::VPORDZrr,       X86::VPORDZrmb,       TB_BCAST_D },
  { X86::VPORQZ128rr,    X86::VPORQZ128rmb,    TB_BCAST_Q },
  { X86::VPORQZ256rr,    X86::VPORQZ256rmb,    TB_BCAST_Q },
  { X86::VPORQZrr,       X86::VPORQZrmb,       TB_BCAST_Q },
  { X86::VPTESTMDZ128rr, X86::VPTESTMDZ128rmb, TB_BCAST_D },
  { X86::VPTESTMDZ256rr, X86::VPTESTMDZ256rmb, TB_BCAST_D },
  { X86::VPTESTMDZrr,    X86::VPTESTMDZrmb,    TB_BCAST_D },
  { X86::VPTESTMQZ128rr, X86::VPTESTMQZ128rmb, TB_BCAST_Q },
  { X86::VPTESTMQZ256rr, X86::VPTESTMQZ256rmb, TB_BCAST_Q },
  { X86::VPTESTMQZrr,    X86::VPTESTMQZrmb,    TB_BCAST_Q },
  { X86::VPTESTNMDZ128rr,X86::VPTESTNMDZ128rmb,TB_BCAST_D },
  { X86::VPTESTNMDZ256rr,X86::VPTESTNMDZ256rmb,TB_BCAST_D },
  { X86::VPTESTNMDZrr,   X86::VPTESTNMDZrmb,   TB_BCAST_D },
  { X86::VPTESTNMQZ128rr,X86::VPTESTNMQZ128rmb,TB_BCAST_Q },
  { X86::VPTESTNMQZ256rr,X86::VPTESTNMQZ256rmb,TB_BCAST_Q },
  { X86::VPTESTNMQZrr,   X86::VPTESTNMQZrmb,   TB_BCAST_Q },
  { X86::VPXORDZ128rr,   X86::VPXORDZ128rmb,   TB_BCAST_D },
  { X86::VPXORDZ256rr,   X86::VPXORDZ256rmb,   TB_BCAST_D },
  { X86::VPXORDZrr,      X86::VPXORDZrmb,      TB_BCAST_D },
  { X86::VPXORQZ128rr,   X86::VPXORQZ128rmb,   TB_BCAST_Q },
  { X86::VPXORQZ256rr,   X86::VPXORQZ256rmb,   TB_BCAST_Q },
  { X86::VPXORQZrr,      X86::VPXORQZrmb,      TB_BCAST_Q },
  { X86::VSUBPDZ128rr,   X86::VSUBPDZ128rmb,   TB_BCAST_SD },
  { X86::VSUBPDZ256rr,   X86::VSUBPDZ256rmb,   TB_BCAST_SD },
  { X86::VSUBPDZrr,      X86::VSUBPDZrmb,      TB_BCAST_SD },
  { X86::VSUBPSZ128rr,   X86::VSUBPSZ128rmb,   TB_BCAST_SS },
  { X86::VSUBPSZ256rr,   X86::VSUBPSZ256rmb,   TB_BCAST_SS },
  { X86::VSUBPSZrr,      X86::VSUBPSZrmb,      TB_BCAST_SS },
  { X86::VXORPDZ128rr,   X86::VXORPDZ128rmb,   TB_BCAST_SD },
  { X86::VXORPDZ256rr,   X86::VXORPDZ256rmb,   TB_BCAST_SD },
  { X86::VXORPDZrr,      X86::VXORPDZrmb,      TB_BCAST_SD },
  { X86::VXORPSZ128rr,   X86::VXORPSZ128rmb,   TB_BCAST_SS },
  { X86::VXORPSZ256rr,   X86::VXORPSZ256rmb,   TB_BCAST_SS },
  { X86::VXORPSZrr,      X86::VXORPSZrmb,      TB_BCAST_SS },
};

static const X86MemoryFoldTableEntry BroadcastFoldTable3[] = {
  { X86::VFMADD132PDZ128r,     X86::VFMADD132PDZ128mb,    TB_BCAST_SD },
  { X86::VFMADD132PDZ256r,     X86::VFMADD132PDZ256mb,    TB_BCAST_SD },
  { X86::VFMADD132PDZr,        X86::VFMADD132PDZmb,       TB_BCAST_SD },
  { X86::VFMADD132PSZ128r,     X86::VFMADD132PSZ128mb,    TB_BCAST_SS },
  { X86::VFMADD132PSZ256r,     X86::VFMADD132PSZ256mb,    TB_BCAST_SS },
  { X86::VFMADD132PSZr,        X86::VFMADD132PSZmb,       TB_BCAST_SS },
  { X86::VFMADD213PDZ128r,     X86::VFMADD213PDZ128mb,    TB_BCAST_SD },
  { X86::VFMADD213PDZ256r,     X86::VFMADD213PDZ256mb,    TB_BCAST_SD },
  { X86::VFMADD213PDZr,        X86::VFMADD213PDZmb,       TB_BCAST_SD },
  { X86::VFMADD213PSZ128r,     X86::VFMADD213PSZ128mb,    TB_BCAST_SS },
  { X86::VFMADD213PSZ256r,     X86::VFMADD213PSZ256mb,    TB_BCAST_SS },
  { X86::VFMADD213PSZr,        X86::VFMADD213PSZmb,       TB_BCAST_SS },
  { X86::VFMADD231PDZ128r,     X86::VFMADD231PDZ128mb,    TB_BCAST_SD },
  { X86::VFMADD231PDZ256r,     X86::VFMADD231PDZ256mb,    TB_BCAST_SD },
  { X86::VFMADD231PDZr,        X86::VFMADD231PDZmb,       TB_BCAST_SD },
  { X86::VFMADD231PSZ128r,     X86::VFMADD231PSZ128mb,    TB_BCAST_SS },
  { X86::VFMADD231PSZ256r,     X86::VFMADD231PSZ256mb,    TB_BCAST_SS },
  { X86::VFMADD231PSZr,        X86::VFMADD231PSZmb,       TB_BCAST_SS },
  { X86::VFMADDSUB132PDZ128r,  X86::VFMADDSUB132PDZ128mb, TB_BCAST_SD },
  { X86::VFMADDSUB132PDZ256r,  X86::VFMADDSUB132PDZ256mb, TB_BCAST_SD },
  { X86::VFMADDSUB132PDZr,     X86::VFMADDSUB132PDZmb,    TB_BCAST_SD },
  { X86::VFMADDSUB132PSZ128r,  X86::VFMADDSUB132PSZ128mb, TB_BCAST_SS },
  { X86::VFMADDSUB132PSZ256r,  X86::VFMADDSUB132PSZ256mb, TB_BCAST_SS },
  { X86::VFMADDSUB132PSZr,     X86::VFMADDSUB132PSZmb,    TB_BCAST_SS },
  { X86::VFMADDSUB213PDZ128r,  X86::VFMADDSUB213PDZ128mb, TB_BCAST_SD },
  { X86::VFMADDSUB213PDZ256r,  X86::VFMADDSUB213PDZ256mb, TB_BCAST_SD },
  { X86::VFMADDSUB213PDZr,     X86::VFMADDSUB213PDZmb,    TB_BCAST_SD },
  { X86::VFMADDSUB213PSZ128r,  X86::VFMADDSUB213PSZ128mb, TB_BCAST_SS },
  { X86::VFMADDSUB213PSZ256r,  X86::VFMADDSUB213PSZ256mb, TB_BCAST_SS },
  { X86::VFMADDSUB213PSZr,     X86::VFMADDSUB213PSZmb,    TB_BCAST_SS },
  { X86::VFMADDSUB231PDZ128r,  X86::VFMADDSUB231PDZ128mb, TB_BCAST_SD },
  { X86::VFMADDSUB231PDZ256r,  X86::VFMADDSUB231PDZ256mb, TB_BCAST_SD },
  { X86::VFMADDSUB231PDZr,     X86::VFMADDSUB231PDZmb,    TB_BCAST_SD },
  { X86::VFMADDSUB231PSZ128r,  X86::VFMADDSUB231PSZ128mb, TB_BCAST_SS },
  { X86::VFMADDSUB231PSZ256r,  X86::VFMADDSUB231PSZ256mb, TB_BCAST_SS },
  { X86::VFMADDSUB231PSZr,     X86::VFMADDSUB231PSZmb,    TB_BCAST_SS },
  { X86::VFMSUB132PDZ128r,     X86::VFMSUB132PDZ128mb,    TB_BCAST_SD },
  { X86::VFMSUB132PDZ256r,     X86::VFMSUB132PDZ256mb,    TB_BCAST_SD },
  { X86::VFMSUB132PDZr,        X86::VFMSUB132PDZmb,       TB_BCAST_SD },
  { X86::VFMSUB132PSZ128r,     X86::VFMSUB132PSZ128mb,    TB_BCAST_SS },
  { X86::VFMSUB132PSZ256r,     X86::VFMSUB132PSZ256mb,    TB_BCAST_SS },
  { X86::VFMSUB132PSZr,        X86::VFMSUB132PSZmb,       TB_BCAST_SS },
  { X86::VFMSUB213PDZ128r,     X86::VFMSUB213PDZ128mb,    TB_BCAST_SD },
  { X86::VFMSUB213PDZ256r,     X86::VFMSUB213PDZ256mb,    TB_BCAST_SD },
  { X86::VFMSUB213PDZr,        X86::VFMSUB213PDZmb,       TB_BCAST_SD },
  { X86::VFMSUB213PSZ128r,     X86::VFMSUB213PSZ128mb,    TB_BCAST_SS },
  { X86::VFMSUB213PSZ256r,     X86::VFMSUB213PSZ256mb,    TB_BCAST_SS },
  { X86::VFMSUB213PSZr,        X86::VFMSUB213PSZmb,       TB_BCAST_SS },
  { X86::VFMSUB231PDZ128r,     X86::VFMSUB231PDZ128mb,    TB_BCAST_SD },
  { X86::VFMSUB231PDZ256r,     X86::VFMSUB231PDZ256mb,    TB_BCAST_SD },
  { X86::VFMSUB231PDZr,        X86::VFMSUB231PDZmb,       TB_BCAST_SD },
  { X86::VFMSUB231PSZ128r,     X86::VFMSUB231PSZ128mb,    TB_BCAST_SS },
  { X86::VFMSUB231PSZ256r,     X86::VFMSUB231PSZ256mb,    TB_BCAST_SS },
  { X86::VFMSUB231PSZr,        X86::VFMSUB231PSZmb,       TB_BCAST_SS },
  { X86::VFMSUBADD132PDZ128r,  X86::VFMSUBADD132PDZ128mb, TB_BCAST_SD },
  { X86::VFMSUBADD132PDZ256r,  X86::VFMSUBADD132PDZ256mb, TB_BCAST_SD },
  { X86::VFMSUBADD132PDZr,     X86::VFMSUBADD132PDZmb,    TB_BCAST_SD },
  { X86::VFMSUBADD132PSZ128r,  X86::VFMSUBADD132PSZ128mb, TB_BCAST_SS },
  { X86::VFMSUBADD132PSZ256r,  X86::VFMSUBADD132PSZ256mb, TB_BCAST_SS },
  { X86::VFMSUBADD132PSZr,     X86::VFMSUBADD132PSZmb,    TB_BCAST_SS },
  { X86::VFMSUBADD213PDZ128r,  X86::VFMSUBADD213PDZ128mb, TB_BCAST_SD },
  { X86::VFMSUBADD213PDZ256r,  X86::VFMSUBADD213PDZ256mb, TB_BCAST_SD },
  { X86::VFMSUBADD213PDZr,     X86::VFMSUBADD213PDZmb,    TB_BCAST_SD },
  { X86::VFMSUBADD213PSZ128r,  X86::VFMSUBADD213PSZ128mb, TB_BCAST_SS },
  { X86::VFMSUBADD213PSZ256r,  X86::VFMSUBADD213PSZ256mb, TB_BCAST_SS },
  { X86::VFMSUBADD213PSZr,     X86::VFMSUBADD213PSZmb,    TB_BCAST_SS },
  { X86::VFMSUBADD231PDZ128r,  X86::VFMSUBADD231PDZ128mb, TB_BCAST_SD },
  { X86::VFMSUBADD231PDZ256r,  X86::VFMSUBADD231PDZ256mb, TB_BCAST_SD },
  { X86::VFMSUBADD231PDZr,     X86::VFMSUBADD231PDZmb,    TB_BCAST_SD },
  { X86::VFMSUBADD231PSZ128r,  X86::VFMSUBADD231PSZ128mb, TB_BCAST_SS },
  { X86::VFMSUBADD231PSZ256r,  X86::VFMSUBADD231PSZ256mb, TB_BCAST_SS },
  { X86::VFMSUBADD231PSZr,     X86::VFMSUBADD231PSZmb,    TB_BCAST_SS },
  { X86::VFNMADD132PDZ128r,    X86::VFNMADD132PDZ128mb,   TB_BCAST_SD },
  { X86::VFNMADD132PDZ256r,    X86::VFNMADD132PDZ256mb,   TB_BCAST_SD },
  { X86::VFNMADD132PDZr,       X86::VFNMADD132PDZmb,      TB_BCAST_SD },
  { X86::VFNMADD132PSZ128r,    X86::VFNMADD132PSZ128mb,   TB_BCAST_SS },
  { X86::VFNMADD132PSZ256r,    X86::VFNMADD132PSZ256mb,   TB_BCAST_SS },
  { X86::VFNMADD132PSZr,       X86::VFNMADD132PSZmb,      TB_BCAST_SS },
  { X86::VFNMADD213PDZ128r,    X86::VFNMADD213PDZ128mb,   TB_BCAST_SD },
  { X86::VFNMADD213PDZ256r,    X86::VFNMADD213PDZ256mb,   TB_BCAST_SD },
  { X86::VFNMADD213PDZr,       X86::VFNMADD213PDZmb,      TB_BCAST_SD },
  { X86::VFNMADD213PSZ128r,    X86::VFNMADD213PSZ128mb,   TB_BCAST_SS },
  { X86::VFNMADD213PSZ256r,    X86::VFNMADD213PSZ256mb,   TB_BCAST_SS },
  { X86::VFNMADD213PSZr,       X86::VFNMADD213PSZmb,      TB_BCAST_SS },
  { X86::VFNMADD231PDZ128r,    X86::VFNMADD231PDZ128mb,   TB_BCAST_SD },
  { X86::VFNMADD231PDZ256r,    X86::VFNMADD231PDZ256mb,   TB_BCAST_SD },
  { X86::VFNMADD231PDZr,       X86::VFNMADD231PDZmb,      TB_BCAST_SD },
  { X86::VFNMADD231PSZ128r,    X86::VFNMADD231PSZ128mb,   TB_BCAST_SS },
  { X86::VFNMADD231PSZ256r,    X86::VFNMADD231PSZ256mb,   TB_BCAST_SS },
  { X86::VFNMADD231PSZr,       X86::VFNMADD231PSZmb,      TB_BCAST_SS },
  { X86::VFNMSUB132PDZ128r,    X86::VFNMSUB132PDZ128mb,   TB_BCAST_SD },
  { X86::VFNMSUB132PDZ256r,    X86::VFNMSUB132PDZ256mb,   TB_BCAST_SD },
  { X86::VFNMSUB132PDZr,       X86::VFNMSUB132PDZmb,      TB_BCAST_SD },
  { X86::VFNMSUB132PSZ128r,    X86::VFNMSUB132PSZ128mb,   TB_BCAST_SS },
  { X86::VFNMSUB132PSZ256r,    X86::VFNMSUB132PSZ256mb,   TB_BCAST_SS },
  { X86::VFNMSUB132PSZr,       X86::VFNMSUB132PSZmb,      TB_BCAST_SS },
  { X86::VFNMSUB213PDZ128r,    X86::VFNMSUB213PDZ128mb,   TB_BCAST_SD },
  { X86::VFNMSUB213PDZ256r,    X86::VFNMSUB213PDZ256mb,   TB_BCAST_SD },
  { X86::VFNMSUB213PDZr,       X86::VFNMSUB213PDZmb,      TB_BCAST_SD },
  { X86::VFNMSUB213PSZ128r,    X86::VFNMSUB213PSZ128mb,   TB_BCAST_SS },
  { X86::VFNMSUB213PSZ256r,    X86::VFNMSUB213PSZ256mb,   TB_BCAST_SS },
  { X86::VFNMSUB213PSZr,       X86::VFNMSUB213PSZmb,      TB_BCAST_SS },
  { X86::VFNMSUB231PDZ128r,    X86::VFNMSUB231PDZ128mb,   TB_BCAST_SD },
  { X86::VFNMSUB231PDZ256r,    X86::VFNMSUB231PDZ256mb,   TB_BCAST_SD },
  { X86::VFNMSUB231PDZr,       X86::VFNMSUB231PDZmb,      TB_BCAST_SD },
  { X86::VFNMSUB231PSZ128r,    X86::VFNMSUB231PSZ128mb,   TB_BCAST_SS },
  { X86::VFNMSUB231PSZ256r,    X86::VFNMSUB231PSZ256mb,   TB_BCAST_SS },
  { X86::VFNMSUB231PSZr,       X86::VFNMSUB231PSZmb,      TB_BCAST_SS },
  { X86::VPTERNLOGDZ128rri,    X86::VPTERNLOGDZ128rmbi,   TB_BCAST_D },
  { X86::VPTERNLOGDZ256rri,    X86::VPTERNLOGDZ256rmbi,   TB_BCAST_D },
  { X86::VPTERNLOGDZrri,       X86::VPTERNLOGDZrmbi,      TB_BCAST_D },
  { X86::VPTERNLOGQZ128rri,    X86::VPTERNLOGQZ128rmbi,   TB_BCAST_Q },
  { X86::VPTERNLOGQZ256rri,    X86::VPTERNLOGQZ256rmbi,   TB_BCAST_Q },
  { X86::VPTERNLOGQZrri,       X86::VPTERNLOGQZrmbi,      TB_BCAST_Q },
};

// Table to map instructions safe to broadcast using a different width from the
// element width.
static const X86MemoryFoldTableEntry BroadcastSizeFoldTable2[] = {
  { X86::VANDNPDZ128rr,        X86::VANDNPSZ128rmb,       TB_BCAST_SS },
  { X86::VANDNPDZ256rr,        X86::VANDNPSZ256rmb,       TB_BCAST_SS },
  { X86::VANDNPDZrr,           X86::VANDNPSZrmb,          TB_BCAST_SS },
  { X86::VANDNPSZ128rr,        X86::VANDNPDZ128rmb,       TB_BCAST_SD },
  { X86::VANDNPSZ256rr,        X86::VANDNPDZ256rmb,       TB_BCAST_SD },
  { X86::VANDNPSZrr,           X86::VANDNPDZrmb,          TB_BCAST_SD },
  { X86::VANDPDZ128rr,         X86::VANDPSZ128rmb,        TB_BCAST_SS },
  { X86::VANDPDZ256rr,         X86::VANDPSZ256rmb,        TB_BCAST_SS },
  { X86::VANDPDZrr,            X86::VANDPSZrmb,           TB_BCAST_SS },
  { X86::VANDPSZ128rr,         X86::VANDPDZ128rmb,        TB_BCAST_SD },
  { X86::VANDPSZ256rr,         X86::VANDPDZ256rmb,        TB_BCAST_SD },
  { X86::VANDPSZrr,            X86::VANDPDZrmb,           TB_BCAST_SD },
  { X86::VORPDZ128rr,          X86::VORPSZ128rmb,         TB_BCAST_SS },
  { X86::VORPDZ256rr,          X86::VORPSZ256rmb,         TB_BCAST_SS },
  { X86::VORPDZrr,             X86::VORPSZrmb,            TB_BCAST_SS },
  { X86::VORPSZ128rr,          X86::VORPDZ128rmb,         TB_BCAST_SD },
  { X86::VORPSZ256rr,          X86::VORPDZ256rmb,         TB_BCAST_SD },
  { X86::VORPSZrr,             X86::VORPDZrmb,            TB_BCAST_SD },
  { X86::VPANDDZ128rr,         X86::VPANDQZ128rmb,        TB_BCAST_Q },
  { X86::VPANDDZ256rr,         X86::VPANDQZ256rmb,        TB_BCAST_Q },
  { X86::VPANDDZrr,            X86::VPANDQZrmb,           TB_BCAST_Q },
  { X86::VPANDNDZ128rr,        X86::VPANDNQZ128rmb,       TB_BCAST_Q },
  { X86::VPANDNDZ256rr,        X86::VPANDNQZ256rmb,       TB_BCAST_Q },
  { X86::VPANDNDZrr,           X86::VPANDNQZrmb,          TB_BCAST_Q },
  { X86::VPANDNQZ128rr,        X86::VPANDNDZ128rmb,       TB_BCAST_D },
  { X86::VPANDNQZ256rr,        X86::VPANDNDZ256rmb,       TB_BCAST_D },
  { X86::VPANDNQZrr,           X86::VPANDNDZrmb,          TB_BCAST_D },
  { X86::VPANDQZ128rr,         X86::VPANDDZ128rmb,        TB_BCAST_D },
  { X86::VPANDQZ256rr,         X86::VPANDDZ256rmb,        TB_BCAST_D },
  { X86::VPANDQZrr,            X86::VPANDDZrmb,           TB_BCAST_D },
  { X86::VPORDZ128rr,          X86::VPORQZ128rmb,         TB_BCAST_Q },
  { X86::VPORDZ256rr,          X86::VPORQZ256rmb,         TB_BCAST_Q },
  { X86::VPORDZrr,             X86::VPORQZrmb,            TB_BCAST_Q },
  { X86::VPORQZ128rr,          X86::VPORDZ128rmb,         TB_BCAST_D },
  { X86::VPORQZ256rr,          X86::VPORDZ256rmb,         TB_BCAST_D },
  { X86::VPORQZrr,             X86::VPORDZrmb,            TB_BCAST_D },
  { X86::VPXORDZ128rr,         X86::VPXORQZ128rmb,        TB_BCAST_Q },
  { X86::VPXORDZ256rr,         X86::VPXORQZ256rmb,        TB_BCAST_Q },
  { X86::VPXORDZrr,            X86::VPXORQZrmb,           TB_BCAST_Q },
  { X86::VPXORQZ128rr,         X86::VPXORDZ128rmb,        TB_BCAST_D },
  { X86::VPXORQZ256rr,         X86::VPXORDZ256rmb,        TB_BCAST_D },
  { X86::VPXORQZrr,            X86::VPXORDZrmb,           TB_BCAST_D },
  { X86::VXORPDZ128rr,         X86::VXORPSZ128rmb,        TB_BCAST_SS },
  { X86::VXORPDZ256rr,         X86::VXORPSZ256rmb,        TB_BCAST_SS },
  { X86::VXORPDZrr,            X86::VXORPSZrmb,           TB_BCAST_SS },
  { X86::VXORPSZ128rr,         X86::VXORPDZ128rmb,        TB_BCAST_SD },
  { X86::VXORPSZ256rr,         X86::VXORPDZ256rmb,        TB_BCAST_SD },
  { X86::VXORPSZrr,            X86::VXORPDZrmb,           TB_BCAST_SD },
};

static const X86MemoryFoldTableEntry BroadcastSizeFoldTable3[] = {
  { X86::VPTERNLOGDZ128rri,    X86::VPTERNLOGQZ128rmbi,   TB_BCAST_Q },
  { X86::VPTERNLOGDZ256rri,    X86::VPTERNLOGQZ256rmbi,   TB_BCAST_Q },
  { X86::VPTERNLOGDZrri,       X86::VPTERNLOGQZrmbi,      TB_BCAST_Q },
  { X86::VPTERNLOGQZ128rri,    X86::VPTERNLOGDZ128rmbi,   TB_BCAST_D },
  { X86::VPTERNLOGQZ256rri,    X86::VPTERNLOGDZ256rmbi,   TB_BCAST_D },
  { X86::VPTERNLOGQZrri,       X86::VPTERNLOGDZrmbi,      TB_BCAST_D },
};

static const X86MemoryFoldTableEntry *
lookupFoldTableImpl(ArrayRef<X86MemoryFoldTableEntry> Table, unsigned RegOp) {
#ifndef NDEBUG
  // Make sure the tables are sorted.
  static std::atomic<bool> FoldTablesChecked(false);
  if (!FoldTablesChecked.load(std::memory_order_relaxed)) {
    assert(llvm::is_sorted(MemoryFoldTable2Addr) &&
           std::adjacent_find(std::begin(MemoryFoldTable2Addr),
                              std::end(MemoryFoldTable2Addr)) ==
               std::end(MemoryFoldTable2Addr) &&
           "MemoryFoldTable2Addr is not sorted and unique!");
    assert(llvm::is_sorted(MemoryFoldTable0) &&
           std::adjacent_find(std::begin(MemoryFoldTable0),
                              std::end(MemoryFoldTable0)) ==
               std::end(MemoryFoldTable0) &&
           "MemoryFoldTable0 is not sorted and unique!");
    assert(llvm::is_sorted(MemoryFoldTable1) &&
           std::adjacent_find(std::begin(MemoryFoldTable1),
                              std::end(MemoryFoldTable1)) ==
               std::end(MemoryFoldTable1) &&
           "MemoryFoldTable1 is not sorted and unique!");
    assert(llvm::is_sorted(MemoryFoldTable2) &&
           std::adjacent_find(std::begin(MemoryFoldTable2),
                              std::end(MemoryFoldTable2)) ==
               std::end(MemoryFoldTable2) &&
           "MemoryFoldTable2 is not sorted and unique!");
    assert(llvm::is_sorted(MemoryFoldTable3) &&
           std::adjacent_find(std::begin(MemoryFoldTable3),
                              std::end(MemoryFoldTable3)) ==
               std::end(MemoryFoldTable3) &&
           "MemoryFoldTable3 is not sorted and unique!");
    assert(llvm::is_sorted(MemoryFoldTable4) &&
           std::adjacent_find(std::begin(MemoryFoldTable4),
                              std::end(MemoryFoldTable4)) ==
               std::end(MemoryFoldTable4) &&
           "MemoryFoldTable4 is not sorted and unique!");
    assert(llvm::is_sorted(BroadcastFoldTable2) &&
           std::adjacent_find(std::begin(BroadcastFoldTable2),
                              std::end(BroadcastFoldTable2)) ==
               std::end(BroadcastFoldTable2) &&
           "BroadcastFoldTable2 is not sorted and unique!");
    assert(llvm::is_sorted(BroadcastFoldTable3) &&
           std::adjacent_find(std::begin(BroadcastFoldTable3),
                              std::end(BroadcastFoldTable3)) ==
               std::end(BroadcastFoldTable3) &&
           "BroadcastFoldTable3 is not sorted and unique!");
    assert(llvm::is_sorted(BroadcastSizeFoldTable2) &&
           std::adjacent_find(std::begin(BroadcastSizeFoldTable2),
                              std::end(BroadcastSizeFoldTable2)) ==
               std::end(BroadcastSizeFoldTable2) &&
           "BroadcastSizeFoldTable2 is not sorted and unique!");
    assert(llvm::is_sorted(BroadcastSizeFoldTable3) &&
           std::adjacent_find(std::begin(BroadcastSizeFoldTable3),
                              std::end(BroadcastSizeFoldTable3)) ==
               std::end(BroadcastSizeFoldTable3) &&
           "BroadcastSizeFoldTable3 is not sorted and unique!");
    FoldTablesChecked.store(true, std::memory_order_relaxed);
  }
#endif

  const X86MemoryFoldTableEntry *Data = llvm::lower_bound(Table, RegOp);
  if (Data != Table.end() && Data->KeyOp == RegOp &&
      !(Data->Flags & TB_NO_FORWARD))
    return Data;
  return nullptr;
}

const X86MemoryFoldTableEntry *
llvm::lookupTwoAddrFoldTable(unsigned RegOp) {
  return lookupFoldTableImpl(MemoryFoldTable2Addr, RegOp);
}

const X86MemoryFoldTableEntry *
llvm::lookupFoldTable(unsigned RegOp, unsigned OpNum) {
  ArrayRef<X86MemoryFoldTableEntry> FoldTable;
  if (OpNum == 0)
    FoldTable = ArrayRef(MemoryFoldTable0);
  else if (OpNum == 1)
    FoldTable = ArrayRef(MemoryFoldTable1);
  else if (OpNum == 2)
    FoldTable = ArrayRef(MemoryFoldTable2);
  else if (OpNum == 3)
    FoldTable = ArrayRef(MemoryFoldTable3);
  else if (OpNum == 4)
    FoldTable = ArrayRef(MemoryFoldTable4);
  else
    return nullptr;

  return lookupFoldTableImpl(FoldTable, RegOp);
}

namespace {

// This class stores the memory unfolding tables. It is instantiated as a
// function scope static variable to lazily init the unfolding table.
struct X86MemUnfoldTable {
  // Stores memory unfolding tables entries sorted by opcode.
  std::vector<X86MemoryFoldTableEntry> Table;

  X86MemUnfoldTable() {
    for (const X86MemoryFoldTableEntry &Entry : MemoryFoldTable2Addr)
      // Index 0, folded load and store, no alignment requirement.
      addTableEntry(Entry, TB_INDEX_0 | TB_FOLDED_LOAD | TB_FOLDED_STORE);

    for (const X86MemoryFoldTableEntry &Entry : MemoryFoldTable0)
      // Index 0, mix of loads and stores.
      addTableEntry(Entry, TB_INDEX_0);

    for (const X86MemoryFoldTableEntry &Entry : MemoryFoldTable1)
      // Index 1, folded load
      addTableEntry(Entry, TB_INDEX_1 | TB_FOLDED_LOAD);

    for (const X86MemoryFoldTableEntry &Entry : MemoryFoldTable2)
      // Index 2, folded load
      addTableEntry(Entry, TB_INDEX_2 | TB_FOLDED_LOAD);

    for (const X86MemoryFoldTableEntry &Entry : MemoryFoldTable3)
      // Index 3, folded load
      addTableEntry(Entry, TB_INDEX_3 | TB_FOLDED_LOAD);

    for (const X86MemoryFoldTableEntry &Entry : MemoryFoldTable4)
      // Index 4, folded load
      addTableEntry(Entry, TB_INDEX_4 | TB_FOLDED_LOAD);

    // Broadcast tables.
    for (const X86MemoryFoldTableEntry &Entry : BroadcastFoldTable2)
      // Index 2, folded broadcast
      addTableEntry(Entry, TB_INDEX_2 | TB_FOLDED_LOAD | TB_FOLDED_BCAST);

    for (const X86MemoryFoldTableEntry &Entry : BroadcastFoldTable3)
      // Index 3, folded broadcast
      addTableEntry(Entry, TB_INDEX_3 | TB_FOLDED_LOAD | TB_FOLDED_BCAST);

    // Sort the memory->reg unfold table.
    array_pod_sort(Table.begin(), Table.end());

    // Now that it's sorted, ensure its unique.
    assert(std::adjacent_find(Table.begin(), Table.end()) == Table.end() &&
           "Memory unfolding table is not unique!");
  }

  void addTableEntry(const X86MemoryFoldTableEntry &Entry,
                     uint16_t ExtraFlags) {
    // NOTE: This swaps the KeyOp and DstOp in the table so we can sort it.
    if ((Entry.Flags & TB_NO_REVERSE) == 0)
      Table.push_back({Entry.DstOp, Entry.KeyOp,
                      static_cast<uint16_t>(Entry.Flags | ExtraFlags) });
  }
};
}

const X86MemoryFoldTableEntry *
llvm::lookupUnfoldTable(unsigned MemOp) {
  static X86MemUnfoldTable MemUnfoldTable;
  auto &Table = MemUnfoldTable.Table;
  auto I = llvm::lower_bound(Table, MemOp);
  if (I != Table.end() && I->KeyOp == MemOp)
    return &*I;
  return nullptr;
}

namespace {

// This class stores the memory -> broadcast folding tables. It is instantiated
// as a function scope static variable to lazily init the folding table.
struct X86MemBroadcastFoldTable {
  // Stores memory broadcast folding tables entries sorted by opcode.
  std::vector<X86MemoryFoldTableEntry> Table;

  X86MemBroadcastFoldTable() {
    // Broadcast tables.
    for (const X86MemoryFoldTableEntry &Reg2Bcst : BroadcastFoldTable2) {
      unsigned RegOp = Reg2Bcst.KeyOp;
      unsigned BcstOp = Reg2Bcst.DstOp;
      if (const X86MemoryFoldTableEntry *Reg2Mem = lookupFoldTable(RegOp, 2)) {
        unsigned MemOp = Reg2Mem->DstOp;
        uint16_t Flags = Reg2Mem->Flags | Reg2Bcst.Flags | TB_INDEX_2 |
                         TB_FOLDED_LOAD | TB_FOLDED_BCAST;
        Table.push_back({MemOp, BcstOp, Flags});
      }
    }
    for (const X86MemoryFoldTableEntry &Reg2Bcst : BroadcastSizeFoldTable2) {
      unsigned RegOp = Reg2Bcst.KeyOp;
      unsigned BcstOp = Reg2Bcst.DstOp;
      if (const X86MemoryFoldTableEntry *Reg2Mem = lookupFoldTable(RegOp, 2)) {
        unsigned MemOp = Reg2Mem->DstOp;
        uint16_t Flags = Reg2Mem->Flags | Reg2Bcst.Flags | TB_INDEX_2 |
                         TB_FOLDED_LOAD | TB_FOLDED_BCAST;
        Table.push_back({MemOp, BcstOp, Flags});
      }
    }

    for (const X86MemoryFoldTableEntry &Reg2Bcst : BroadcastFoldTable3) {
      unsigned RegOp = Reg2Bcst.KeyOp;
      unsigned BcstOp = Reg2Bcst.DstOp;
      if (const X86MemoryFoldTableEntry *Reg2Mem = lookupFoldTable(RegOp, 3)) {
        unsigned MemOp = Reg2Mem->DstOp;
        uint16_t Flags = Reg2Mem->Flags | Reg2Bcst.Flags | TB_INDEX_3 |
                         TB_FOLDED_LOAD | TB_FOLDED_BCAST;
        Table.push_back({MemOp, BcstOp, Flags});
      }
    }
    for (const X86MemoryFoldTableEntry &Reg2Bcst : BroadcastSizeFoldTable3) {
      unsigned RegOp = Reg2Bcst.KeyOp;
      unsigned BcstOp = Reg2Bcst.DstOp;
      if (const X86MemoryFoldTableEntry *Reg2Mem = lookupFoldTable(RegOp, 3)) {
        unsigned MemOp = Reg2Mem->DstOp;
        uint16_t Flags = Reg2Mem->Flags | Reg2Bcst.Flags | TB_INDEX_3 |
                         TB_FOLDED_LOAD | TB_FOLDED_BCAST;
        Table.push_back({MemOp, BcstOp, Flags});
      }
    }

    // Sort the memory->broadcast fold table.
    array_pod_sort(Table.begin(), Table.end());
  }
};
} // namespace

static bool matchBroadcastSize(const X86MemoryFoldTableEntry &Entry,
                               unsigned BroadcastBits) {
  switch (Entry.Flags & TB_BCAST_MASK) {
  case TB_BCAST_SD:
  case TB_BCAST_Q:
    return BroadcastBits == 64;
  case TB_BCAST_SS:
  case TB_BCAST_D:
    return BroadcastBits == 32;
  }
  return false;
}

const X86MemoryFoldTableEntry *
llvm::lookupBroadcastFoldTable(unsigned MemOp, unsigned BroadcastBits) {
  static X86MemBroadcastFoldTable MemBroadcastFoldTable;
  auto &Table = MemBroadcastFoldTable.Table;
  for (auto I = llvm::lower_bound(Table, MemOp);
       I != Table.end() && I->KeyOp == MemOp; ++I) {
    if (matchBroadcastSize(*I, BroadcastBits))
      return &*I;
  }
  return nullptr;
}
