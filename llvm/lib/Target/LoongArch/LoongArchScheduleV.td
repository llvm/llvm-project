//==- LoongArchScheduleV.td - LoongArch V Scheduling Defs -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/// Define scheduler resources associated with def operands.
///
/// Vector instruction - LSX (128-bit) / LASX (256-bit)
///
/// The 128-bit and the 256-bit form of an operation are kept apart because a
/// few of them do not share the same timing, e.g. XVINSGR2VR.W is slower than
/// VINSGR2VR.W and XVSETEQZ.V is slower than VSETEQZ.V.

//===----------------------------------------------------------------------===//
/// Vector integer arithmetic
//===----------------------------------------------------------------------===//

def WriteVALU128              : SchedWrite;  // 128-bit integer add / sub
def WriteVALU256              : SchedWrite;  // 256-bit integer add / sub
def ReadVALU128               : SchedRead;   // 128-bit integer add / sub
def ReadVALU256               : SchedRead;   // 256-bit integer add / sub
//===----------------------------------------------------------------------===//
// VADD.B  VADD.H  VADD.W  VADD.D  VSUB.B  VSUB.H  VSUB.W  VSUB.D  VSADD.B
// VSADD.H  VSADD.W  VSADD.D  VSADD.BU  VSADD.HU  VSADD.WU  VSADD.DU
// VSSUB.B  VSSUB.H  VSSUB.W  VSSUB.D  VSSUB.BU  VSSUB.HU  VSSUB.WU
// VSSUB.DU
//   Sched<[WriteVALU128, ReadVALU128, ReadVALU128]>
// VADDI.BU  VADDI.HU  VADDI.WU  VADDI.DU  VSUBI.BU  VSUBI.HU  VSUBI.WU
// VSUBI.DU  VNEG.B  VNEG.H  VNEG.W  VNEG.D
//   Sched<[WriteVALU128, ReadVALU128]>
// XVADD.B  XVADD.H  XVADD.W  XVADD.D  XVSUB.B  XVSUB.H  XVSUB.W  XVSUB.D
// XVSADD.B  XVSADD.H  XVSADD.W  XVSADD.D  XVSADD.BU  XVSADD.HU  XVSADD.WU
// XVSADD.DU  XVSSUB.B  XVSSUB.H  XVSSUB.W  XVSSUB.D  XVSSUB.BU  XVSSUB.HU
// XVSSUB.WU  XVSSUB.DU
//   Sched<[WriteVALU256, ReadVALU256, ReadVALU256]>
// XVADDI.BU  XVADDI.HU  XVADDI.WU  XVADDI.DU  XVSUBI.BU  XVSUBI.HU
// XVSUBI.WU  XVSUBI.DU  XVNEG.B  XVNEG.H  XVNEG.W  XVNEG.D
//   Sched<[WriteVALU256, ReadVALU256]>
//===----------------------------------------------------------------------===//

def WriteVALUQ128             : SchedWrite;  // 128-bit quadword add / sub
def WriteVALUQ256             : SchedWrite;  // 256-bit quadword add / sub
def ReadVALUQ128              : SchedRead;   // 128-bit quadword add / sub
def ReadVALUQ256              : SchedRead;   // 256-bit quadword add / sub
//===----------------------------------------------------------------------===//
// VADD.Q  VSUB.Q
//   Sched<[WriteVALUQ128, ReadVALUQ128, ReadVALUQ128]>
// XVADD.Q  XVSUB.Q
//   Sched<[WriteVALUQ256, ReadVALUQ256, ReadVALUQ256]>
//===----------------------------------------------------------------------===//

def WriteVAVG128              : SchedWrite;  // 128-bit integer average
def WriteVAVG256              : SchedWrite;  // 256-bit integer average
def ReadVAVG128               : SchedRead;   // 128-bit integer average
def ReadVAVG256               : SchedRead;   // 256-bit integer average
//===----------------------------------------------------------------------===//
// VAVG.B  VAVG.H  VAVG.W  VAVG.BU  VAVG.HU  VAVG.WU  VAVGR.B  VAVGR.H
// VAVGR.W  VAVGR.BU  VAVGR.HU  VAVGR.WU
//   Sched<[WriteVAVG128, ReadVAVG128, ReadVAVG128]>
// XVAVG.B  XVAVG.H  XVAVG.W  XVAVG.BU  XVAVG.HU  XVAVG.WU  XVAVGR.B
// XVAVGR.H  XVAVGR.W  XVAVGR.BU  XVAVGR.HU  XVAVGR.WU
//   Sched<[WriteVAVG256, ReadVAVG256, ReadVAVG256]>
//===----------------------------------------------------------------------===//

def WriteVAVGD128             : SchedWrite;  // 128-bit doubleword average
def WriteVAVGD256             : SchedWrite;  // 256-bit doubleword average
def ReadVAVGD128              : SchedRead;   // 128-bit doubleword average
def ReadVAVGD256              : SchedRead;   // 256-bit doubleword average
//===----------------------------------------------------------------------===//
// VAVG.D  VAVG.DU  VAVGR.D  VAVGR.DU
//   Sched<[WriteVAVGD128, ReadVAVGD128, ReadVAVGD128]>
// XVAVG.D  XVAVG.DU  XVAVGR.D  XVAVGR.DU
//   Sched<[WriteVAVGD256, ReadVAVGD256, ReadVAVGD256]>
//===----------------------------------------------------------------------===//

def WriteVABSD128             : SchedWrite;  // 128-bit integer abs difference
def WriteVABSD256             : SchedWrite;  // 256-bit integer abs difference
def ReadVABSD128              : SchedRead;   // 128-bit integer abs difference
def ReadVABSD256              : SchedRead;   // 256-bit integer abs difference
//===----------------------------------------------------------------------===//
// VABSD.B  VABSD.H  VABSD.W  VABSD.D  VABSD.BU  VABSD.HU  VABSD.WU
// VABSD.DU
//   Sched<[WriteVABSD128, ReadVABSD128, ReadVABSD128]>
// XVABSD.B  XVABSD.H  XVABSD.W  XVABSD.D  XVABSD.BU  XVABSD.HU  XVABSD.WU
// XVABSD.DU
//   Sched<[WriteVABSD256, ReadVABSD256, ReadVABSD256]>
//===----------------------------------------------------------------------===//

def WriteVADDA128             : SchedWrite;  // 128-bit integer add absolute
def WriteVADDA256             : SchedWrite;  // 256-bit integer add absolute
def ReadVADDA128              : SchedRead;   // 128-bit integer add absolute
def ReadVADDA256              : SchedRead;   // 256-bit integer add absolute
//===----------------------------------------------------------------------===//
// VADDA.B  VADDA.H  VADDA.W  VADDA.D
//   Sched<[WriteVADDA128, ReadVADDA128, ReadVADDA128]>
// XVADDA.B  XVADDA.H  XVADDA.W  XVADDA.D
//   Sched<[WriteVADDA256, ReadVADDA256, ReadVADDA256]>
//===----------------------------------------------------------------------===//

def WriteVHADDW128            : SchedWrite;  // 128-bit horizontal add / sub
def WriteVHADDW256            : SchedWrite;  // 256-bit horizontal add / sub
def ReadVHADDW128             : SchedRead;   // 128-bit horizontal add / sub
def ReadVHADDW256             : SchedRead;   // 256-bit horizontal add / sub
//===----------------------------------------------------------------------===//
// VHADDW.H.B  VHADDW.W.H  VHADDW.D.W  VHADDW.HU.BU  VHADDW.WU.HU
// VHADDW.DU.WU  VHSUBW.H.B  VHSUBW.W.H  VHSUBW.D.W  VHSUBW.HU.BU
// VHSUBW.WU.HU  VHSUBW.DU.WU
//   Sched<[WriteVHADDW128, ReadVHADDW128, ReadVHADDW128]>
// XVHADDW.H.B  XVHADDW.W.H  XVHADDW.D.W  XVHADDW.HU.BU  XVHADDW.WU.HU
// XVHADDW.DU.WU  XVHSUBW.H.B  XVHSUBW.W.H  XVHSUBW.D.W  XVHSUBW.HU.BU
// XVHSUBW.WU.HU  XVHSUBW.DU.WU
//   Sched<[WriteVHADDW256, ReadVHADDW256, ReadVHADDW256]>
//===----------------------------------------------------------------------===//

def WriteVHADDWQ128           : SchedWrite;  // 128-bit quadword horizontal add
def WriteVHADDWQ256           : SchedWrite;  // 256-bit quadword horizontal add
def ReadVHADDWQ128            : SchedRead;   // 128-bit quadword horizontal add
def ReadVHADDWQ256            : SchedRead;   // 256-bit quadword horizontal add
//===----------------------------------------------------------------------===//
// VHADDW.Q.D  VHADDW.QU.DU  VHSUBW.Q.D  VHSUBW.QU.DU
//   Sched<[WriteVHADDWQ128, ReadVHADDWQ128, ReadVHADDWQ128]>
// XVHADDW.Q.D  XVHADDW.QU.DU  XVHSUBW.Q.D  XVHSUBW.QU.DU
//   Sched<[WriteVHADDWQ256, ReadVHADDWQ256, ReadVHADDWQ256]>
//===----------------------------------------------------------------------===//

def WriteVWADD128             : SchedWrite;  // 128-bit widening add / sub
def WriteVWADD256             : SchedWrite;  // 256-bit widening add / sub
def ReadVWADD128              : SchedRead;   // 128-bit widening add / sub
def ReadVWADD256              : SchedRead;   // 256-bit widening add / sub
//===----------------------------------------------------------------------===//
// VADDWEV.H.B  VADDWEV.W.H  VADDWEV.D.W  VADDWOD.H.B  VADDWOD.W.H
// VADDWOD.D.W  VSUBWEV.H.B  VSUBWEV.W.H  VSUBWEV.D.W  VSUBWOD.H.B
// VSUBWOD.W.H  VSUBWOD.D.W  VADDWEV.H.BU  VADDWEV.W.HU  VADDWEV.D.WU
// VADDWOD.H.BU  VADDWOD.W.HU  VADDWOD.D.WU  VSUBWEV.H.BU  VSUBWEV.W.HU
// VSUBWEV.D.WU  VSUBWOD.H.BU  VSUBWOD.W.HU  VSUBWOD.D.WU  VADDWEV.H.BU.B
// VADDWEV.W.HU.H  VADDWEV.D.WU.W  VADDWOD.H.BU.B  VADDWOD.W.HU.H
// VADDWOD.D.WU.W
//   Sched<[WriteVWADD128, ReadVWADD128, ReadVWADD128]>
// XVADDWEV.H.B  XVADDWEV.W.H  XVADDWEV.D.W  XVADDWOD.H.B  XVADDWOD.W.H
// XVADDWOD.D.W  XVSUBWEV.H.B  XVSUBWEV.W.H  XVSUBWEV.D.W  XVSUBWOD.H.B
// XVSUBWOD.W.H  XVSUBWOD.D.W  XVADDWEV.H.BU  XVADDWEV.W.HU  XVADDWEV.D.WU
// XVADDWOD.H.BU  XVADDWOD.W.HU  XVADDWOD.D.WU  XVSUBWEV.H.BU  XVSUBWEV.W.HU
// XVSUBWEV.D.WU  XVSUBWOD.H.BU  XVSUBWOD.W.HU  XVSUBWOD.D.WU
// XVADDWEV.H.BU.B  XVADDWEV.W.HU.H  XVADDWEV.D.WU.W  XVADDWOD.H.BU.B
// XVADDWOD.W.HU.H  XVADDWOD.D.WU.W
//   Sched<[WriteVWADD256, ReadVWADD256, ReadVWADD256]>
//===----------------------------------------------------------------------===//

def WriteVWADDQ128            : SchedWrite;  // 128-bit quadword widening add
def WriteVWADDQ256            : SchedWrite;  // 256-bit quadword widening add
def ReadVWADDQ128             : SchedRead;   // 128-bit quadword widening add
def ReadVWADDQ256             : SchedRead;   // 256-bit quadword widening add
//===----------------------------------------------------------------------===//
// VADDWEV.Q.D  VADDWOD.Q.D  VSUBWEV.Q.D  VSUBWOD.Q.D  VADDWEV.Q.DU
// VADDWOD.Q.DU  VSUBWEV.Q.DU  VSUBWOD.Q.DU  VADDWEV.Q.DU.D  VADDWOD.Q.DU.D
//   Sched<[WriteVWADDQ128, ReadVWADDQ128, ReadVWADDQ128]>
// XVADDWEV.Q.D  XVADDWOD.Q.D  XVSUBWEV.Q.D  XVSUBWOD.Q.D  XVADDWEV.Q.DU
// XVADDWOD.Q.DU  XVSUBWEV.Q.DU  XVSUBWOD.Q.DU  XVADDWEV.Q.DU.D
// XVADDWOD.Q.DU.D
//   Sched<[WriteVWADDQ256, ReadVWADDQ256, ReadVWADDQ256]>
//===----------------------------------------------------------------------===//

def WriteVMINMAX128           : SchedWrite;  // 128-bit integer min / max
def WriteVMINMAX256           : SchedWrite;  // 256-bit integer min / max
def ReadVMINMAX128            : SchedRead;   // 128-bit integer min / max
def ReadVMINMAX256            : SchedRead;   // 256-bit integer min / max
//===----------------------------------------------------------------------===//
// VMAX.B  VMAX.H  VMAX.W  VMAX.BU  VMAX.HU  VMAX.WU  VMIN.B  VMIN.H  VMIN.W
// VMIN.BU  VMIN.HU  VMIN.WU
//   Sched<[WriteVMINMAX128, ReadVMINMAX128, ReadVMINMAX128]>
// VMAXI.B  VMAXI.H  VMAXI.W  VMAXI.BU  VMAXI.HU  VMAXI.WU  VMINI.B  VMINI.H
// VMINI.W  VMINI.BU  VMINI.HU  VMINI.WU
//   Sched<[WriteVMINMAX128, ReadVMINMAX128]>
// XVMAX.B  XVMAX.H  XVMAX.W  XVMAX.BU  XVMAX.HU  XVMAX.WU  XVMIN.B  XVMIN.H
// XVMIN.W  XVMIN.BU  XVMIN.HU  XVMIN.WU
//   Sched<[WriteVMINMAX256, ReadVMINMAX256, ReadVMINMAX256]>
// XVMAXI.B  XVMAXI.H  XVMAXI.W  XVMAXI.BU  XVMAXI.HU  XVMAXI.WU  XVMINI.B
// XVMINI.H  XVMINI.W  XVMINI.BU  XVMINI.HU  XVMINI.WU
//   Sched<[WriteVMINMAX256, ReadVMINMAX256]>
//===----------------------------------------------------------------------===//

def WriteVMINMAXD128          : SchedWrite;  // 128-bit doubleword min / max
def WriteVMINMAXD256          : SchedWrite;  // 256-bit doubleword min / max
def ReadVMINMAXD128           : SchedRead;   // 128-bit doubleword min / max
def ReadVMINMAXD256           : SchedRead;   // 256-bit doubleword min / max
//===----------------------------------------------------------------------===//
// VMAX.D  VMAX.DU  VMIN.D  VMIN.DU
//   Sched<[WriteVMINMAXD128, ReadVMINMAXD128, ReadVMINMAXD128]>
// VMAXI.D  VMAXI.DU  VMINI.D  VMINI.DU
//   Sched<[WriteVMINMAXD128, ReadVMINMAXD128]>
// XVMAX.D  XVMAX.DU  XVMIN.D  XVMIN.DU
//   Sched<[WriteVMINMAXD256, ReadVMINMAXD256, ReadVMINMAXD256]>
// XVMAXI.D  XVMAXI.DU  XVMINI.D  XVMINI.DU
//   Sched<[WriteVMINMAXD256, ReadVMINMAXD256]>
//===----------------------------------------------------------------------===//

def WriteVMUL128              : SchedWrite;  // 128-bit integer multiply
def WriteVMUL256              : SchedWrite;  // 256-bit integer multiply
def ReadVMUL128               : SchedRead;   // 128-bit integer multiply
def ReadVMUL256               : SchedRead;   // 256-bit integer multiply
//===----------------------------------------------------------------------===//
// VMUL.B  VMUL.H  VMUL.W  VMUL.D  VMUH.B  VMUH.H  VMUH.W  VMUH.D  VMUH.BU
// VMUH.HU  VMUH.WU  VMUH.DU  VMULWEV.H.B  VMULWEV.W.H  VMULWEV.D.W
// VMULWOD.H.B  VMULWOD.W.H  VMULWOD.D.W  VMULWEV.H.BU  VMULWEV.W.HU
// VMULWEV.D.WU  VMULWOD.H.BU  VMULWOD.W.HU  VMULWOD.D.WU  VMULWEV.H.BU.B
// VMULWEV.W.HU.H  VMULWEV.D.WU.W  VMULWOD.H.BU.B  VMULWOD.W.HU.H
// VMULWOD.D.WU.W
//   Sched<[WriteVMUL128, ReadVMUL128, ReadVMUL128]>
// XVMUL.B  XVMUL.H  XVMUL.W  XVMUL.D  XVMUH.B  XVMUH.H  XVMUH.W  XVMUH.D
// XVMUH.BU  XVMUH.HU  XVMUH.WU  XVMUH.DU  XVMULWEV.H.B  XVMULWEV.W.H
// XVMULWEV.D.W  XVMULWOD.H.B  XVMULWOD.W.H  XVMULWOD.D.W  XVMULWEV.H.BU
// XVMULWEV.W.HU  XVMULWEV.D.WU  XVMULWOD.H.BU  XVMULWOD.W.HU  XVMULWOD.D.WU
// XVMULWEV.H.BU.B  XVMULWEV.W.HU.H  XVMULWEV.D.WU.W  XVMULWOD.H.BU.B
// XVMULWOD.W.HU.H  XVMULWOD.D.WU.W
//   Sched<[WriteVMUL256, ReadVMUL256, ReadVMUL256]>
//===----------------------------------------------------------------------===//

def WriteVMULQ128             : SchedWrite;  // 128-bit quadword multiply
def WriteVMULQ256             : SchedWrite;  // 256-bit quadword multiply
def ReadVMULQ128              : SchedRead;   // 128-bit quadword multiply
def ReadVMULQ256              : SchedRead;   // 256-bit quadword multiply
//===----------------------------------------------------------------------===//
// VMULWEV.Q.D  VMULWOD.Q.D  VMULWEV.Q.DU  VMULWOD.Q.DU  VMULWEV.Q.DU.D
// VMULWOD.Q.DU.D
//   Sched<[WriteVMULQ128, ReadVMULQ128, ReadVMULQ128]>
// XVMULWEV.Q.D  XVMULWOD.Q.D  XVMULWEV.Q.DU  XVMULWOD.Q.DU  XVMULWEV.Q.DU.D
// XVMULWOD.Q.DU.D
//   Sched<[WriteVMULQ256, ReadVMULQ256, ReadVMULQ256]>
//===----------------------------------------------------------------------===//

def WriteVMADD128             : SchedWrite;  // 128-bit integer multiply-add
def WriteVMADD256             : SchedWrite;  // 256-bit integer multiply-add
def ReadVMADD128              : SchedRead;   // 128-bit integer multiply-add
def ReadVMADD256              : SchedRead;   // 256-bit integer multiply-add
//===----------------------------------------------------------------------===//
// VMADD.B  VMADD.H  VMADD.W  VMADD.D  VMSUB.B  VMSUB.H  VMSUB.W  VMSUB.D
// VMADDWEV.H.B  VMADDWEV.W.H  VMADDWEV.D.W  VMADDWOD.H.B  VMADDWOD.W.H
// VMADDWOD.D.W  VMADDWEV.H.BU  VMADDWEV.W.HU  VMADDWEV.D.WU  VMADDWOD.H.BU
// VMADDWOD.W.HU  VMADDWOD.D.WU  VMADDWEV.H.BU.B  VMADDWEV.W.HU.H
// VMADDWEV.D.WU.W  VMADDWOD.H.BU.B  VMADDWOD.W.HU.H  VMADDWOD.D.WU.W
//   Sched<[WriteVMADD128, ReadVMADD128, ReadVMADD128, ReadVMADD128]>
// XVMADD.B  XVMADD.H  XVMADD.W  XVMADD.D  XVMSUB.B  XVMSUB.H  XVMSUB.W
// XVMSUB.D  XVMADDWEV.H.B  XVMADDWEV.W.H  XVMADDWEV.D.W  XVMADDWOD.H.B
// XVMADDWOD.W.H  XVMADDWOD.D.W  XVMADDWEV.H.BU  XVMADDWEV.W.HU
// XVMADDWEV.D.WU  XVMADDWOD.H.BU  XVMADDWOD.W.HU  XVMADDWOD.D.WU
// XVMADDWEV.H.BU.B  XVMADDWEV.W.HU.H  XVMADDWEV.D.WU.W  XVMADDWOD.H.BU.B
// XVMADDWOD.W.HU.H  XVMADDWOD.D.WU.W
//   Sched<[WriteVMADD256, ReadVMADD256, ReadVMADD256, ReadVMADD256]>
//===----------------------------------------------------------------------===//

def WriteVMADDQ128            : SchedWrite;  // 128-bit quadword multiply-add
def WriteVMADDQ256            : SchedWrite;  // 256-bit quadword multiply-add
def ReadVMADDQ128             : SchedRead;   // 128-bit quadword multiply-add
def ReadVMADDQ256             : SchedRead;   // 256-bit quadword multiply-add
//===----------------------------------------------------------------------===//
// VMADDWEV.Q.D  VMADDWOD.Q.D  VMADDWEV.Q.DU  VMADDWOD.Q.DU  VMADDWEV.Q.DU.D
// VMADDWOD.Q.DU.D
//   Sched<[WriteVMADDQ128, ReadVMADDQ128, ReadVMADDQ128, ReadVMADDQ128]>
// XVMADDWEV.Q.D  XVMADDWOD.Q.D  XVMADDWEV.Q.DU  XVMADDWOD.Q.DU
// XVMADDWEV.Q.DU.D  XVMADDWOD.Q.DU.D
//   Sched<[WriteVMADDQ256, ReadVMADDQ256, ReadVMADDQ256, ReadVMADDQ256]>
//===----------------------------------------------------------------------===//

def WriteVDIVB128             : SchedWrite;  // 128-bit byte integer divide
def WriteVDIVB256             : SchedWrite;  // 256-bit byte integer divide
def ReadVDIVB128              : SchedRead;   // 128-bit byte integer divide
def ReadVDIVB256              : SchedRead;   // 256-bit byte integer divide
//===----------------------------------------------------------------------===//
// VDIV.B  VDIV.BU  VMOD.B  VMOD.BU
//   Sched<[WriteVDIVB128, ReadVDIVB128, ReadVDIVB128]>
// XVDIV.B  XVDIV.BU  XVMOD.B  XVMOD.BU
//   Sched<[WriteVDIVB256, ReadVDIVB256, ReadVDIVB256]>
//===----------------------------------------------------------------------===//

def WriteVDIVH128             : SchedWrite;  // 128-bit halfword divide
def WriteVDIVH256             : SchedWrite;  // 256-bit halfword divide
def ReadVDIVH128              : SchedRead;   // 128-bit halfword divide
def ReadVDIVH256              : SchedRead;   // 256-bit halfword divide
//===----------------------------------------------------------------------===//
// VDIV.H  VDIV.HU  VMOD.H  VMOD.HU
//   Sched<[WriteVDIVH128, ReadVDIVH128, ReadVDIVH128]>
// XVDIV.H  XVDIV.HU  XVMOD.H  XVMOD.HU
//   Sched<[WriteVDIVH256, ReadVDIVH256, ReadVDIVH256]>
//===----------------------------------------------------------------------===//

def WriteVDIVW128             : SchedWrite;  // 128-bit word integer divide
def WriteVDIVW256             : SchedWrite;  // 256-bit word integer divide
def ReadVDIVW128              : SchedRead;   // 128-bit word integer divide
def ReadVDIVW256              : SchedRead;   // 256-bit word integer divide
//===----------------------------------------------------------------------===//
// VDIV.W  VDIV.WU  VMOD.W  VMOD.WU
//   Sched<[WriteVDIVW128, ReadVDIVW128, ReadVDIVW128]>
// XVDIV.W  XVDIV.WU  XVMOD.W  XVMOD.WU
//   Sched<[WriteVDIVW256, ReadVDIVW256, ReadVDIVW256]>
//===----------------------------------------------------------------------===//

def WriteVDIVD128             : SchedWrite;  // 128-bit doubleword divide
def WriteVDIVD256             : SchedWrite;  // 256-bit doubleword divide
def ReadVDIVD128              : SchedRead;   // 128-bit doubleword divide
def ReadVDIVD256              : SchedRead;   // 256-bit doubleword divide
//===----------------------------------------------------------------------===//
// VDIV.D  VDIV.DU  VMOD.D  VMOD.DU
//   Sched<[WriteVDIVD128, ReadVDIVD128, ReadVDIVD128]>
// XVDIV.D  XVDIV.DU  XVMOD.D  XVMOD.DU
//   Sched<[WriteVDIVD256, ReadVDIVD256, ReadVDIVD256]>
//===----------------------------------------------------------------------===//

def WriteVSAT128              : SchedWrite;  // 128-bit integer saturate
def WriteVSAT256              : SchedWrite;  // 256-bit integer saturate
def ReadVSAT128               : SchedRead;   // 128-bit integer saturate
def ReadVSAT256               : SchedRead;   // 256-bit integer saturate
//===----------------------------------------------------------------------===//
// VSAT.B  VSAT.H  VSAT.W  VSAT.D  VSAT.BU  VSAT.HU  VSAT.WU  VSAT.DU
//   Sched<[WriteVSAT128, ReadVSAT128]>
// XVSAT.B  XVSAT.H  XVSAT.W  XVSAT.D  XVSAT.BU  XVSAT.HU  XVSAT.WU
// XVSAT.DU
//   Sched<[WriteVSAT256, ReadVSAT256]>
//===----------------------------------------------------------------------===//

def WriteVSIGNCOV128          : SchedWrite;  // 128-bit integer sign transfer
def WriteVSIGNCOV256          : SchedWrite;  // 256-bit integer sign transfer
def ReadVSIGNCOV128           : SchedRead;   // 128-bit integer sign transfer
def ReadVSIGNCOV256           : SchedRead;   // 256-bit integer sign transfer
//===----------------------------------------------------------------------===//
// VSIGNCOV.B  VSIGNCOV.H  VSIGNCOV.W  VSIGNCOV.D
//   Sched<[WriteVSIGNCOV128, ReadVSIGNCOV128, ReadVSIGNCOV128]>
// XVSIGNCOV.B  XVSIGNCOV.H  XVSIGNCOV.W  XVSIGNCOV.D
//   Sched<[WriteVSIGNCOV256, ReadVSIGNCOV256, ReadVSIGNCOV256]>
//===----------------------------------------------------------------------===//

def WriteVEXTH128             : SchedWrite;  // 128-bit high-half extend
def WriteVEXTH256             : SchedWrite;  // 256-bit high-half extend
def ReadVEXTH128              : SchedRead;   // 128-bit high-half extend
def ReadVEXTH256              : SchedRead;   // 256-bit high-half extend
//===----------------------------------------------------------------------===//
// VEXTH.H.B  VEXTH.W.H  VEXTH.D.W  VEXTH.Q.D  VEXTH.HU.BU  VEXTH.WU.HU
// VEXTH.DU.WU  VEXTH.QU.DU
//   Sched<[WriteVEXTH128, ReadVEXTH128]>
// XVEXTH.H.B  XVEXTH.W.H  XVEXTH.D.W  XVEXTH.Q.D  XVEXTH.HU.BU
// XVEXTH.WU.HU  XVEXTH.DU.WU  XVEXTH.QU.DU
//   Sched<[WriteVEXTH256, ReadVEXTH256]>
//===----------------------------------------------------------------------===//

def WriteVEXT2XV256           : SchedWrite;  // 256-bit low-half extend
def ReadVEXT2XV256            : SchedRead;   // 256-bit low-half extend
//===----------------------------------------------------------------------===//
// VEXT2XV.H.B  VEXT2XV.W.B  VEXT2XV.D.B  VEXT2XV.W.H  VEXT2XV.D.H
// VEXT2XV.D.W  VEXT2XV.HU.BU  VEXT2XV.WU.BU  VEXT2XV.DU.BU  VEXT2XV.WU.HU
// VEXT2XV.DU.HU  VEXT2XV.DU.WU
//   Sched<[WriteVEXT2XV256, ReadVEXT2XV256]>
//===----------------------------------------------------------------------===//

def WriteVEXTL128             : SchedWrite;  // 128-bit extend to quadword
def WriteVEXTL256             : SchedWrite;  // 256-bit extend to quadword
def ReadVEXTL128              : SchedRead;   // 128-bit extend to quadword
def ReadVEXTL256              : SchedRead;   // 256-bit extend to quadword
//===----------------------------------------------------------------------===//
// VEXTL.Q.D  VEXTL.QU.DU
//   Sched<[WriteVEXTL128, ReadVEXTL128]>
// XVEXTL.Q.D  XVEXTL.QU.DU
//   Sched<[WriteVEXTL256, ReadVEXTL256]>
//===----------------------------------------------------------------------===//

def WriteVLDI128              : SchedWrite;  // 128-bit immediate load
def WriteVLDI256              : SchedWrite;  // 256-bit immediate load
def ReadVLDI128               : SchedRead;   // 128-bit immediate load
def ReadVLDI256               : SchedRead;   // 256-bit immediate load
//===----------------------------------------------------------------------===//
// VLDI
//   Sched<[WriteVLDI128]>
// XVLDI
//   Sched<[WriteVLDI256]>
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Vector compare, mask and condition flag
//===----------------------------------------------------------------------===//

def WriteVCMP128              : SchedWrite;  // 128-bit integer compare
def WriteVCMP256              : SchedWrite;  // 256-bit integer compare
def ReadVCMP128               : SchedRead;   // 128-bit integer compare
def ReadVCMP256               : SchedRead;   // 256-bit integer compare
//===----------------------------------------------------------------------===//
// VSEQ.B  VSEQ.H  VSEQ.W  VSEQ.D  VSLE.B  VSLE.H  VSLE.W  VSLE.BU  VSLE.HU
// VSLE.WU  VSLT.B  VSLT.H  VSLT.W  VSLT.BU  VSLT.HU  VSLT.WU
//   Sched<[WriteVCMP128, ReadVCMP128, ReadVCMP128]>
// VSEQI.B  VSEQI.H  VSEQI.W  VSEQI.D  VSLEI.B  VSLEI.H  VSLEI.W  VSLEI.BU
// VSLEI.HU  VSLEI.WU  VSLTI.B  VSLTI.H  VSLTI.W  VSLTI.BU  VSLTI.HU
// VSLTI.WU
//   Sched<[WriteVCMP128, ReadVCMP128]>
// XVSEQ.B  XVSEQ.H  XVSEQ.W  XVSEQ.D  XVSLE.B  XVSLE.H  XVSLE.W  XVSLE.BU
// XVSLE.HU  XVSLE.WU  XVSLT.B  XVSLT.H  XVSLT.W  XVSLT.BU  XVSLT.HU
// XVSLT.WU
//   Sched<[WriteVCMP256, ReadVCMP256, ReadVCMP256]>
// XVSEQI.B  XVSEQI.H  XVSEQI.W  XVSEQI.D  XVSLEI.B  XVSLEI.H  XVSLEI.W
// XVSLEI.BU  XVSLEI.HU  XVSLEI.WU  XVSLTI.B  XVSLTI.H  XVSLTI.W  XVSLTI.BU
// XVSLTI.HU  XVSLTI.WU
//   Sched<[WriteVCMP256, ReadVCMP256]>
//===----------------------------------------------------------------------===//

def WriteVCMPD128             : SchedWrite;  // 128-bit doubleword compare
def WriteVCMPD256             : SchedWrite;  // 256-bit doubleword compare
def ReadVCMPD128              : SchedRead;   // 128-bit doubleword compare
def ReadVCMPD256              : SchedRead;   // 256-bit doubleword compare
//===----------------------------------------------------------------------===//
// VSLE.D  VSLE.DU  VSLT.D  VSLT.DU
//   Sched<[WriteVCMPD128, ReadVCMPD128, ReadVCMPD128]>
// VSLEI.D  VSLEI.DU  VSLTI.D  VSLTI.DU
//   Sched<[WriteVCMPD128, ReadVCMPD128]>
// XVSLE.D  XVSLE.DU  XVSLT.D  XVSLT.DU
//   Sched<[WriteVCMPD256, ReadVCMPD256, ReadVCMPD256]>
// XVSLEI.D  XVSLEI.DU  XVSLTI.D  XVSLTI.DU
//   Sched<[WriteVCMPD256, ReadVCMPD256]>
//===----------------------------------------------------------------------===//

def WriteVFCMP128             : SchedWrite;  // 128-bit FP compare
def WriteVFCMP256             : SchedWrite;  // 256-bit FP compare
def ReadVFCMP128              : SchedRead;   // 128-bit FP compare
def ReadVFCMP256              : SchedRead;   // 256-bit FP compare
//===----------------------------------------------------------------------===//
// VFCMP.CAF.S  VFCMP.SAF.S  VFCMP.CLT.S  VFCMP.SLT.S  VFCMP.CEQ.S
// VFCMP.SEQ.S  VFCMP.CLE.S  VFCMP.SLE.S  VFCMP.CUN.S  VFCMP.SUN.S
// VFCMP.CULT.S  VFCMP.SULT.S  VFCMP.CUEQ.S  VFCMP.SUEQ.S  VFCMP.CULE.S
// VFCMP.SULE.S  VFCMP.CNE.S  VFCMP.SNE.S  VFCMP.COR.S  VFCMP.SOR.S
// VFCMP.CUNE.S  VFCMP.SUNE.S  VFCMP.CAF.D  VFCMP.SAF.D  VFCMP.CLT.D
// VFCMP.SLT.D  VFCMP.CEQ.D  VFCMP.SEQ.D  VFCMP.CLE.D  VFCMP.SLE.D
// VFCMP.CUN.D  VFCMP.SUN.D  VFCMP.CULT.D  VFCMP.SULT.D  VFCMP.CUEQ.D
// VFCMP.SUEQ.D  VFCMP.CULE.D  VFCMP.SULE.D  VFCMP.CNE.D  VFCMP.SNE.D
// VFCMP.COR.D  VFCMP.SOR.D  VFCMP.CUNE.D  VFCMP.SUNE.D
//   Sched<[WriteVFCMP128, ReadVFCMP128, ReadVFCMP128]>
// XVFCMP.CAF.S  XVFCMP.SAF.S  XVFCMP.CLT.S  XVFCMP.SLT.S  XVFCMP.CEQ.S
// XVFCMP.SEQ.S  XVFCMP.CLE.S  XVFCMP.SLE.S  XVFCMP.CUN.S  XVFCMP.SUN.S
// XVFCMP.CULT.S  XVFCMP.SULT.S  XVFCMP.CUEQ.S  XVFCMP.SUEQ.S  XVFCMP.CULE.S
// XVFCMP.SULE.S  XVFCMP.CNE.S  XVFCMP.SNE.S  XVFCMP.COR.S  XVFCMP.SOR.S
// XVFCMP.CUNE.S  XVFCMP.SUNE.S  XVFCMP.CAF.D  XVFCMP.SAF.D  XVFCMP.CLT.D
// XVFCMP.SLT.D  XVFCMP.CEQ.D  XVFCMP.SEQ.D  XVFCMP.CLE.D  XVFCMP.SLE.D
// XVFCMP.CUN.D  XVFCMP.SUN.D  XVFCMP.CULT.D  XVFCMP.SULT.D  XVFCMP.CUEQ.D
// XVFCMP.SUEQ.D  XVFCMP.CULE.D  XVFCMP.SULE.D  XVFCMP.CNE.D  XVFCMP.SNE.D
// XVFCMP.COR.D  XVFCMP.SOR.D  XVFCMP.CUNE.D  XVFCMP.SUNE.D
//   Sched<[WriteVFCMP256, ReadVFCMP256, ReadVFCMP256]>
//===----------------------------------------------------------------------===//

def WriteVMSK128              : SchedWrite;  // 128-bit mask generation
def WriteVMSK256              : SchedWrite;  // 256-bit mask generation
def ReadVMSK128               : SchedRead;   // 128-bit mask generation
def ReadVMSK256               : SchedRead;   // 256-bit mask generation
//===----------------------------------------------------------------------===//
// VMSKLTZ.B  VMSKLTZ.H  VMSKLTZ.W  VMSKLTZ.D  VMSKGEZ.B  VMSKNZ.B
//   Sched<[WriteVMSK128, ReadVMSK128]>
// XVMSKLTZ.B  XVMSKLTZ.H  XVMSKLTZ.W  XVMSKLTZ.D  XVMSKGEZ.B  XVMSKNZ.B
//   Sched<[WriteVMSK256, ReadVMSK256]>
//===----------------------------------------------------------------------===//

def WriteVSET128              : SchedWrite;  // 128-bit condition flag set
def WriteVSET256              : SchedWrite;  // 256-bit condition flag set
def ReadVSET128               : SchedRead;   // 128-bit condition flag set
def ReadVSET256               : SchedRead;   // 256-bit condition flag set
//===----------------------------------------------------------------------===//
// VSETEQZ.V  VSETNEZ.V  VSETANYEQZ.B  VSETANYEQZ.H  VSETANYEQZ.W
// VSETANYEQZ.D  VSETALLNEZ.B  VSETALLNEZ.H  VSETALLNEZ.W  VSETALLNEZ.D
//   Sched<[WriteVSET128, ReadVSET128]>
// XVSETEQZ.V  XVSETNEZ.V  XVSETANYEQZ.B  XVSETANYEQZ.H  XVSETANYEQZ.W
// XVSETANYEQZ.D  XVSETALLNEZ.B  XVSETALLNEZ.H  XVSETALLNEZ.W  XVSETALLNEZ.D
//   Sched<[WriteVSET256, ReadVSET256]>
//===----------------------------------------------------------------------===//

def WriteVFRSTP128            : SchedWrite;  // 128-bit find first negative elem
def WriteVFRSTP256            : SchedWrite;  // 256-bit find first negative elem
def ReadVFRSTP128             : SchedRead;   // 128-bit find first negative elem
def ReadVFRSTP256             : SchedRead;   // 256-bit find first negative elem
//===----------------------------------------------------------------------===//
// VFRSTP.B  VFRSTP.H
//   Sched<[WriteVFRSTP128, ReadVFRSTP128, ReadVFRSTP128, ReadVFRSTP128]>
// VFRSTPI.B  VFRSTPI.H
//   Sched<[WriteVFRSTP128, ReadVFRSTP128, ReadVFRSTP128]>
// XVFRSTP.B  XVFRSTP.H
//   Sched<[WriteVFRSTP256, ReadVFRSTP256, ReadVFRSTP256, ReadVFRSTP256]>
// XVFRSTPI.B  XVFRSTPI.H
//   Sched<[WriteVFRSTP256, ReadVFRSTP256, ReadVFRSTP256]>
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Vector bit manipulation
//===----------------------------------------------------------------------===//

def WriteVBIT128              : SchedWrite;  // 128-bit bitwise logic
def WriteVBIT256              : SchedWrite;  // 256-bit bitwise logic
def ReadVBIT128               : SchedRead;   // 128-bit bitwise logic
def ReadVBIT256               : SchedRead;   // 256-bit bitwise logic
//===----------------------------------------------------------------------===//
// VAND.V  VOR.V  VXOR.V  VNOR.V  VANDN.V  VORN.V
//   Sched<[WriteVBIT128, ReadVBIT128, ReadVBIT128]>
// VANDI.B  VORI.B  VXORI.B  VNORI.B
//   Sched<[WriteVBIT128, ReadVBIT128]>
// XVAND.V  XVOR.V  XVXOR.V  XVNOR.V  XVANDN.V  XVORN.V
//   Sched<[WriteVBIT256, ReadVBIT256, ReadVBIT256]>
// XVANDI.B  XVORI.B  XVXORI.B  XVNORI.B
//   Sched<[WriteVBIT256, ReadVBIT256]>
//===----------------------------------------------------------------------===//

def WriteVBITOPS128           : SchedWrite;  // 128-bit single-bit operation
def WriteVBITOPS256           : SchedWrite;  // 256-bit single-bit operation
def ReadVBITOPS128            : SchedRead;   // 128-bit single-bit operation
def ReadVBITOPS256            : SchedRead;   // 256-bit single-bit operation
//===----------------------------------------------------------------------===//
// VBITCLR.B  VBITCLR.H  VBITCLR.W  VBITCLR.D  VBITSET.B  VBITSET.H
// VBITSET.W  VBITSET.D  VBITREV.B  VBITREV.H  VBITREV.W  VBITREV.D
//   Sched<[WriteVBITOPS128, ReadVBITOPS128, ReadVBITOPS128]>
// VBITCLRI.B  VBITCLRI.H  VBITCLRI.W  VBITCLRI.D  VBITSETI.B  VBITSETI.H
// VBITSETI.W  VBITSETI.D  VBITREVI.B  VBITREVI.H  VBITREVI.W  VBITREVI.D
//   Sched<[WriteVBITOPS128, ReadVBITOPS128]>
// XVBITCLR.B  XVBITCLR.H  XVBITCLR.W  XVBITCLR.D  XVBITSET.B  XVBITSET.H
// XVBITSET.W  XVBITSET.D  XVBITREV.B  XVBITREV.H  XVBITREV.W  XVBITREV.D
//   Sched<[WriteVBITOPS256, ReadVBITOPS256, ReadVBITOPS256]>
// XVBITCLRI.B  XVBITCLRI.H  XVBITCLRI.W  XVBITCLRI.D  XVBITSETI.B
// XVBITSETI.H  XVBITSETI.W  XVBITSETI.D  XVBITREVI.B  XVBITREVI.H
// XVBITREVI.W  XVBITREVI.D
//   Sched<[WriteVBITOPS256, ReadVBITOPS256]>
//===----------------------------------------------------------------------===//

def WriteVBITSEL128           : SchedWrite;  // 128-bit bitwise select
def WriteVBITSEL256           : SchedWrite;  // 256-bit bitwise select
def ReadVBITSEL128            : SchedRead;   // 128-bit bitwise select
def ReadVBITSEL256            : SchedRead;   // 256-bit bitwise select
//===----------------------------------------------------------------------===//
// VBITSEL.V
//   Sched<[WriteVBITSEL128, ReadVBITSEL128, ReadVBITSEL128, ReadVBITSEL128]>
// VBITSELI.B
//   Sched<[WriteVBITSEL128, ReadVBITSEL128, ReadVBITSEL128]>
// XVBITSEL.V
//   Sched<[WriteVBITSEL256, ReadVBITSEL256, ReadVBITSEL256, ReadVBITSEL256]>
// XVBITSELI.B
//   Sched<[WriteVBITSEL256, ReadVBITSEL256, ReadVBITSEL256]>
//===----------------------------------------------------------------------===//

def WriteVPCNT128             : SchedWrite;  // 128-bit population count
def WriteVPCNT256             : SchedWrite;  // 256-bit population count
def ReadVPCNT128              : SchedRead;   // 128-bit population count
def ReadVPCNT256              : SchedRead;   // 256-bit population count
//===----------------------------------------------------------------------===//
// VPCNT.B  VPCNT.H  VPCNT.W  VPCNT.D
//   Sched<[WriteVPCNT128, ReadVPCNT128]>
// XVPCNT.B  XVPCNT.H  XVPCNT.W  XVPCNT.D
//   Sched<[WriteVPCNT256, ReadVPCNT256]>
//===----------------------------------------------------------------------===//

def WriteVCL128               : SchedWrite;  // 128-bit count leading one / zero
def WriteVCL256               : SchedWrite;  // 256-bit count leading one / zero
def ReadVCL128                : SchedRead;   // 128-bit count leading one / zero
def ReadVCL256                : SchedRead;   // 256-bit count leading one / zero
//===----------------------------------------------------------------------===//
// VCLO.B  VCLO.H  VCLO.W  VCLO.D  VCLZ.B  VCLZ.H  VCLZ.W  VCLZ.D
//   Sched<[WriteVCL128, ReadVCL128]>
// XVCLO.B  XVCLO.H  XVCLO.W  XVCLO.D  XVCLZ.B  XVCLZ.H  XVCLZ.W  XVCLZ.D
//   Sched<[WriteVCL256, ReadVCL256]>
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Vector shift
//===----------------------------------------------------------------------===//

def WriteVSHIFT128            : SchedWrite;  // 128-bit integer shift
def WriteVSHIFT256            : SchedWrite;  // 256-bit integer shift
def ReadVSHIFT128             : SchedRead;   // 128-bit integer shift
def ReadVSHIFT256             : SchedRead;   // 256-bit integer shift
//===----------------------------------------------------------------------===//
// VSLL.B  VSLL.H  VSLL.W  VSLL.D  VSRL.B  VSRL.H  VSRL.W  VSRL.D  VSRA.B
// VSRA.H  VSRA.W  VSRA.D
//   Sched<[WriteVSHIFT128, ReadVSHIFT128, ReadVSHIFT128]>
// VSLLI.B  VSLLI.H  VSLLI.W  VSLLI.D  VSRLI.B  VSRLI.H  VSRLI.W  VSRLI.D
// VSRAI.B  VSRAI.H  VSRAI.W  VSRAI.D  VBSLL.V  VBSRL.V
//   Sched<[WriteVSHIFT128, ReadVSHIFT128]>
// XVSLL.B  XVSLL.H  XVSLL.W  XVSLL.D  XVSRL.B  XVSRL.H  XVSRL.W  XVSRL.D
// XVSRA.B  XVSRA.H  XVSRA.W  XVSRA.D
//   Sched<[WriteVSHIFT256, ReadVSHIFT256, ReadVSHIFT256]>
// XVSLLI.B  XVSLLI.H  XVSLLI.W  XVSLLI.D  XVSRLI.B  XVSRLI.H  XVSRLI.W
// XVSRLI.D  XVSRAI.B  XVSRAI.H  XVSRAI.W  XVSRAI.D  XVBSLL.V  XVBSRL.V
//   Sched<[WriteVSHIFT256, ReadVSHIFT256]>
//===----------------------------------------------------------------------===//

def WriteVROTR128             : SchedWrite;  // 128-bit integer rotate
def WriteVROTR256             : SchedWrite;  // 256-bit integer rotate
def ReadVROTR128              : SchedRead;   // 128-bit integer rotate
def ReadVROTR256              : SchedRead;   // 256-bit integer rotate
//===----------------------------------------------------------------------===//
// VROTR.B  VROTR.H  VROTR.W  VROTR.D
//   Sched<[WriteVROTR128, ReadVROTR128, ReadVROTR128]>
// VROTRI.B  VROTRI.H  VROTRI.W  VROTRI.D
//   Sched<[WriteVROTR128, ReadVROTR128]>
// XVROTR.B  XVROTR.H  XVROTR.W  XVROTR.D
//   Sched<[WriteVROTR256, ReadVROTR256, ReadVROTR256]>
// XVROTRI.B  XVROTRI.H  XVROTRI.W  XVROTRI.D
//   Sched<[WriteVROTR256, ReadVROTR256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTR128           : SchedWrite;  // 128-bit shift with rounding
def WriteVSHIFTR256           : SchedWrite;  // 256-bit shift with rounding
def ReadVSHIFTR128            : SchedRead;   // 128-bit shift with rounding
def ReadVSHIFTR256            : SchedRead;   // 256-bit shift with rounding
//===----------------------------------------------------------------------===//
// VSRLR.B  VSRLR.H  VSRLR.W  VSRLR.D  VSRAR.B  VSRAR.H  VSRAR.W  VSRAR.D
//   Sched<[WriteVSHIFTR128, ReadVSHIFTR128, ReadVSHIFTR128]>
// VSRLRI.B  VSRLRI.H  VSRLRI.W  VSRLRI.D  VSRARI.B  VSRARI.H  VSRARI.W
// VSRARI.D
//   Sched<[WriteVSHIFTR128, ReadVSHIFTR128]>
// XVSRLR.B  XVSRLR.H  XVSRLR.W  XVSRLR.D  XVSRAR.B  XVSRAR.H  XVSRAR.W
// XVSRAR.D
//   Sched<[WriteVSHIFTR256, ReadVSHIFTR256, ReadVSHIFTR256]>
// XVSRLRI.B  XVSRLRI.H  XVSRLRI.W  XVSRLRI.D  XVSRARI.B  XVSRARI.H
// XVSRARI.W  XVSRARI.D
//   Sched<[WriteVSHIFTR256, ReadVSHIFTR256]>
//===----------------------------------------------------------------------===//

def WriteVSLLWIL128           : SchedWrite;  // 128-bit widening shift left
def WriteVSLLWIL256           : SchedWrite;  // 256-bit widening shift left
def ReadVSLLWIL128            : SchedRead;   // 128-bit widening shift left
def ReadVSLLWIL256            : SchedRead;   // 256-bit widening shift left
//===----------------------------------------------------------------------===//
// VSLLWIL.H.B  VSLLWIL.W.H  VSLLWIL.D.W  VSLLWIL.HU.BU  VSLLWIL.WU.HU
// VSLLWIL.DU.WU
//   Sched<[WriteVSLLWIL128, ReadVSLLWIL128]>
// XVSLLWIL.H.B  XVSLLWIL.W.H  XVSLLWIL.D.W  XVSLLWIL.HU.BU  XVSLLWIL.WU.HU
// XVSLLWIL.DU.WU
//   Sched<[WriteVSLLWIL256, ReadVSLLWIL256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTN128           : SchedWrite;  // 128-bit narrowing shift
def WriteVSHIFTN256           : SchedWrite;  // 256-bit narrowing shift
def ReadVSHIFTN128            : SchedRead;   // 128-bit narrowing shift
def ReadVSHIFTN256            : SchedRead;   // 256-bit narrowing shift
//===----------------------------------------------------------------------===//
// VSRLN.B.H  VSRLN.H.W  VSRLN.W.D  VSRAN.B.H  VSRAN.H.W  VSRAN.W.D
//   Sched<[WriteVSHIFTN128, ReadVSHIFTN128, ReadVSHIFTN128]>
// XVSRLN.B.H  XVSRLN.H.W  XVSRLN.W.D  XVSRAN.B.H  XVSRAN.H.W  XVSRAN.W.D
//   Sched<[WriteVSHIFTN256, ReadVSHIFTN256, ReadVSHIFTN256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTNI128          : SchedWrite;  // 128-bit narrowing shift by imm
def WriteVSHIFTNI256          : SchedWrite;  // 256-bit narrowing shift by imm
def ReadVSHIFTNI128           : SchedRead;   // 128-bit narrowing shift by imm
def ReadVSHIFTNI256           : SchedRead;   // 256-bit narrowing shift by imm
//===----------------------------------------------------------------------===//
// VSRLNI.B.H  VSRLNI.H.W  VSRLNI.W.D  VSRANI.B.H  VSRANI.H.W  VSRANI.W.D
//   Sched<[WriteVSHIFTNI128, ReadVSHIFTNI128, ReadVSHIFTNI128]>
// XVSRLNI.B.H  XVSRLNI.H.W  XVSRLNI.W.D  XVSRANI.B.H  XVSRANI.H.W
// XVSRANI.W.D
//   Sched<[WriteVSHIFTNI256, ReadVSHIFTNI256, ReadVSHIFTNI256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTNIQ128         : SchedWrite;  // 128-bit quadword narrow shf imm
def WriteVSHIFTNIQ256         : SchedWrite;  // 256-bit quadword narrow shf imm
def ReadVSHIFTNIQ128          : SchedRead;   // 128-bit quadword narrow shf imm
def ReadVSHIFTNIQ256          : SchedRead;   // 256-bit quadword narrow shf imm
//===----------------------------------------------------------------------===//
// VSRLNI.D.Q  VSRANI.D.Q
//   Sched<[WriteVSHIFTNIQ128, ReadVSHIFTNIQ128, ReadVSHIFTNIQ128]>
// XVSRLNI.D.Q  XVSRANI.D.Q
//   Sched<[WriteVSHIFTNIQ256, ReadVSHIFTNIQ256, ReadVSHIFTNIQ256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTRN128          : SchedWrite;  // 128-bit rounding narrow shift
def WriteVSHIFTRN256          : SchedWrite;  // 256-bit rounding narrow shift
def ReadVSHIFTRN128           : SchedRead;   // 128-bit rounding narrow shift
def ReadVSHIFTRN256           : SchedRead;   // 256-bit rounding narrow shift
//===----------------------------------------------------------------------===//
// VSRLRN.B.H  VSRLRN.H.W  VSRLRN.W.D  VSRARN.B.H  VSRARN.H.W  VSRARN.W.D
//   Sched<[WriteVSHIFTRN128, ReadVSHIFTRN128, ReadVSHIFTRN128]>
// XVSRLRN.B.H  XVSRLRN.H.W  XVSRLRN.W.D  XVSRARN.B.H  XVSRARN.H.W
// XVSRARN.W.D
//   Sched<[WriteVSHIFTRN256, ReadVSHIFTRN256, ReadVSHIFTRN256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTRNI128         : SchedWrite;  // 128-bit round narrow shf by imm
def WriteVSHIFTRNI256         : SchedWrite;  // 256-bit round narrow shf by imm
def ReadVSHIFTRNI128          : SchedRead;   // 128-bit round narrow shf by imm
def ReadVSHIFTRNI256          : SchedRead;   // 256-bit round narrow shf by imm
//===----------------------------------------------------------------------===//
// VSRLRNI.B.H  VSRLRNI.H.W  VSRLRNI.W.D  VSRARNI.B.H  VSRARNI.H.W
// VSRARNI.W.D
//   Sched<[WriteVSHIFTRNI128, ReadVSHIFTRNI128, ReadVSHIFTRNI128]>
// XVSRLRNI.B.H  XVSRLRNI.H.W  XVSRLRNI.W.D  XVSRARNI.B.H  XVSRARNI.H.W
// XVSRARNI.W.D
//   Sched<[WriteVSHIFTRNI256, ReadVSHIFTRNI256, ReadVSHIFTRNI256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTRNIQ128        : SchedWrite;  // 128-bit quadword rnd narrow imm
def WriteVSHIFTRNIQ256        : SchedWrite;  // 256-bit quadword rnd narrow imm
def ReadVSHIFTRNIQ128         : SchedRead;   // 128-bit quadword rnd narrow imm
def ReadVSHIFTRNIQ256         : SchedRead;   // 256-bit quadword rnd narrow imm
//===----------------------------------------------------------------------===//
// VSRLRNI.D.Q  VSRARNI.D.Q
//   Sched<[WriteVSHIFTRNIQ128, ReadVSHIFTRNIQ128, ReadVSHIFTRNIQ128]>
// XVSRLRNI.D.Q  XVSRARNI.D.Q
//   Sched<[WriteVSHIFTRNIQ256, ReadVSHIFTRNIQ256, ReadVSHIFTRNIQ256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTNS128          : SchedWrite;  // 128-bit sat narrowing shift
def WriteVSHIFTNS256          : SchedWrite;  // 256-bit sat narrowing shift
def ReadVSHIFTNS128           : SchedRead;   // 128-bit sat narrowing shift
def ReadVSHIFTNS256           : SchedRead;   // 256-bit sat narrowing shift
//===----------------------------------------------------------------------===//
// VSSRLN.B.H  VSSRLN.H.W  VSSRLN.W.D  VSSRAN.B.H  VSSRAN.H.W  VSSRAN.W.D
// VSSRLN.BU.H  VSSRLN.HU.W  VSSRLN.WU.D  VSSRAN.BU.H  VSSRAN.HU.W
// VSSRAN.WU.D
//   Sched<[WriteVSHIFTNS128, ReadVSHIFTNS128, ReadVSHIFTNS128]>
// XVSSRLN.B.H  XVSSRLN.H.W  XVSSRLN.W.D  XVSSRAN.B.H  XVSSRAN.H.W
// XVSSRAN.W.D  XVSSRLN.BU.H  XVSSRLN.HU.W  XVSSRLN.WU.D  XVSSRAN.BU.H
// XVSSRAN.HU.W  XVSSRAN.WU.D
//   Sched<[WriteVSHIFTNS256, ReadVSHIFTNS256, ReadVSHIFTNS256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTNSI128         : SchedWrite;  // 128-bit sat narrow shf by imm
def WriteVSHIFTNSI256         : SchedWrite;  // 256-bit sat narrow shf by imm
def ReadVSHIFTNSI128          : SchedRead;   // 128-bit sat narrow shf by imm
def ReadVSHIFTNSI256          : SchedRead;   // 256-bit sat narrow shf by imm
//===----------------------------------------------------------------------===//
// VSSRLNI.B.H  VSSRLNI.H.W  VSSRLNI.W.D  VSSRANI.B.H  VSSRANI.H.W
// VSSRANI.W.D  VSSRLNI.BU.H  VSSRLNI.HU.W  VSSRLNI.WU.D  VSSRANI.BU.H
// VSSRANI.HU.W  VSSRANI.WU.D
//   Sched<[WriteVSHIFTNSI128, ReadVSHIFTNSI128, ReadVSHIFTNSI128]>
// XVSSRLNI.B.H  XVSSRLNI.H.W  XVSSRLNI.W.D  XVSSRANI.B.H  XVSSRANI.H.W
// XVSSRANI.W.D  XVSSRLNI.BU.H  XVSSRLNI.HU.W  XVSSRLNI.WU.D  XVSSRANI.BU.H
// XVSSRANI.HU.W  XVSSRANI.WU.D
//   Sched<[WriteVSHIFTNSI256, ReadVSHIFTNSI256, ReadVSHIFTNSI256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTNSIQ128        : SchedWrite;  // 128-bit quadword sat narrow imm
def WriteVSHIFTNSIQ256        : SchedWrite;  // 256-bit quadword sat narrow imm
def ReadVSHIFTNSIQ128         : SchedRead;   // 128-bit quadword sat narrow imm
def ReadVSHIFTNSIQ256         : SchedRead;   // 256-bit quadword sat narrow imm
//===----------------------------------------------------------------------===//
// VSSRLNI.D.Q  VSSRANI.D.Q  VSSRLNI.DU.Q  VSSRANI.DU.Q
//   Sched<[WriteVSHIFTNSIQ128, ReadVSHIFTNSIQ128, ReadVSHIFTNSIQ128]>
// XVSSRLNI.D.Q  XVSSRANI.D.Q  XVSSRLNI.DU.Q  XVSSRANI.DU.Q
//   Sched<[WriteVSHIFTNSIQ256, ReadVSHIFTNSIQ256, ReadVSHIFTNSIQ256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTRNS128         : SchedWrite;  // 128-bit round sat narrow shift
def WriteVSHIFTRNS256         : SchedWrite;  // 256-bit round sat narrow shift
def ReadVSHIFTRNS128          : SchedRead;   // 128-bit round sat narrow shift
def ReadVSHIFTRNS256          : SchedRead;   // 256-bit round sat narrow shift
//===----------------------------------------------------------------------===//
// VSSRLRN.B.H  VSSRLRN.H.W  VSSRLRN.W.D  VSSRARN.B.H  VSSRARN.H.W
// VSSRARN.W.D  VSSRLRN.BU.H  VSSRLRN.HU.W  VSSRLRN.WU.D  VSSRARN.BU.H
// VSSRARN.HU.W  VSSRARN.WU.D
//   Sched<[WriteVSHIFTRNS128, ReadVSHIFTRNS128, ReadVSHIFTRNS128]>
// XVSSRLRN.B.H  XVSSRLRN.H.W  XVSSRLRN.W.D  XVSSRARN.B.H  XVSSRARN.H.W
// XVSSRARN.W.D  XVSSRLRN.BU.H  XVSSRLRN.HU.W  XVSSRLRN.WU.D  XVSSRARN.BU.H
// XVSSRARN.HU.W  XVSSRARN.WU.D
//   Sched<[WriteVSHIFTRNS256, ReadVSHIFTRNS256, ReadVSHIFTRNS256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTRNSI128        : SchedWrite;  // 128-bit rnd sat narrow shf imm
def WriteVSHIFTRNSI256        : SchedWrite;  // 256-bit rnd sat narrow shf imm
def ReadVSHIFTRNSI128         : SchedRead;   // 128-bit rnd sat narrow shf imm
def ReadVSHIFTRNSI256         : SchedRead;   // 256-bit rnd sat narrow shf imm
//===----------------------------------------------------------------------===//
// VSSRLRNI.B.H  VSSRLRNI.H.W  VSSRLRNI.W.D  VSSRARNI.B.H  VSSRARNI.H.W
// VSSRARNI.W.D  VSSRLRNI.BU.H  VSSRLRNI.HU.W  VSSRLRNI.WU.D  VSSRARNI.BU.H
// VSSRARNI.HU.W  VSSRARNI.WU.D
//   Sched<[WriteVSHIFTRNSI128, ReadVSHIFTRNSI128, ReadVSHIFTRNSI128]>
// XVSSRLRNI.B.H  XVSSRLRNI.H.W  XVSSRLRNI.W.D  XVSSRARNI.B.H  XVSSRARNI.H.W
// XVSSRARNI.W.D  XVSSRLRNI.BU.H  XVSSRLRNI.HU.W  XVSSRLRNI.WU.D
// XVSSRARNI.BU.H  XVSSRARNI.HU.W  XVSSRARNI.WU.D
//   Sched<[WriteVSHIFTRNSI256, ReadVSHIFTRNSI256, ReadVSHIFTRNSI256]>
//===----------------------------------------------------------------------===//

def WriteVSHIFTRNSIQ128       : SchedWrite;  // 128-bit quadword rnd sat narrow
def WriteVSHIFTRNSIQ256       : SchedWrite;  // 256-bit quadword rnd sat narrow
def ReadVSHIFTRNSIQ128        : SchedRead;   // 128-bit quadword rnd sat narrow
def ReadVSHIFTRNSIQ256        : SchedRead;   // 256-bit quadword rnd sat narrow
//===----------------------------------------------------------------------===//
// VSSRLRNI.D.Q  VSSRARNI.D.Q  VSSRLRNI.DU.Q  VSSRARNI.DU.Q
//   Sched<[WriteVSHIFTRNSIQ128, ReadVSHIFTRNSIQ128, ReadVSHIFTRNSIQ128]>
// XVSSRLRNI.D.Q  XVSSRARNI.D.Q  XVSSRLRNI.DU.Q  XVSSRARNI.DU.Q
//   Sched<[WriteVSHIFTRNSIQ256, ReadVSHIFTRNSIQ256, ReadVSHIFTRNSIQ256]>
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Vector shuffle, permute and element move
//===----------------------------------------------------------------------===//

def WriteVPACK128             : SchedWrite;  // 128-bit interleave / pack
def WriteVPACK256             : SchedWrite;  // 256-bit interleave / pack
def ReadVPACK128              : SchedRead;   // 128-bit interleave / pack
def ReadVPACK256              : SchedRead;   // 256-bit interleave / pack
//===----------------------------------------------------------------------===//
// VPACKEV.B  VPACKEV.H  VPACKEV.W  VPACKEV.D  VPACKOD.B  VPACKOD.H
// VPACKOD.W  VPACKOD.D  VPICKEV.B  VPICKEV.H  VPICKEV.W  VPICKEV.D
// VPICKOD.B  VPICKOD.H  VPICKOD.W  VPICKOD.D  VILVL.B  VILVL.H  VILVL.W
// VILVL.D  VILVH.B  VILVH.H  VILVH.W  VILVH.D
//   Sched<[WriteVPACK128, ReadVPACK128, ReadVPACK128]>
// XVPACKEV.B  XVPACKEV.H  XVPACKEV.W  XVPACKEV.D  XVPACKOD.B  XVPACKOD.H
// XVPACKOD.W  XVPACKOD.D  XVPICKEV.B  XVPICKEV.H  XVPICKEV.W  XVPICKEV.D
// XVPICKOD.B  XVPICKOD.H  XVPICKOD.W  XVPICKOD.D  XVILVL.B  XVILVL.H
// XVILVL.W  XVILVL.D  XVILVH.B  XVILVH.H  XVILVH.W  XVILVH.D
//   Sched<[WriteVPACK256, ReadVPACK256, ReadVPACK256]>
//===----------------------------------------------------------------------===//

def WriteVSHUF128             : SchedWrite;  // 128-bit shuffle by register
def WriteVSHUF256             : SchedWrite;  // 256-bit shuffle by register
def ReadVSHUF128              : SchedRead;   // 128-bit shuffle by register
def ReadVSHUF256              : SchedRead;   // 256-bit shuffle by register
//===----------------------------------------------------------------------===//
// VSHUF.B  VSHUF.H  VSHUF.W  VSHUF.D
//   Sched<[WriteVSHUF128, ReadVSHUF128, ReadVSHUF128, ReadVSHUF128]>
// XVSHUF.B  XVSHUF.H  XVSHUF.W  XVSHUF.D
//   Sched<[WriteVSHUF256, ReadVSHUF256, ReadVSHUF256, ReadVSHUF256]>
//===----------------------------------------------------------------------===//

def WriteVSHUF4I128           : SchedWrite;  // 128-bit shuffle by immediate
def WriteVSHUF4I256           : SchedWrite;  // 256-bit shuffle by immediate
def ReadVSHUF4I128            : SchedRead;   // 128-bit shuffle by immediate
def ReadVSHUF4I256            : SchedRead;   // 256-bit shuffle by immediate
//===----------------------------------------------------------------------===//
// VSHUF4I.B  VSHUF4I.H  VSHUF4I.W
//   Sched<[WriteVSHUF4I128, ReadVSHUF4I128]>
// VSHUF4I.D
//   Sched<[WriteVSHUF4I128, ReadVSHUF4I128, ReadVSHUF4I128]>
// XVSHUF4I.B  XVSHUF4I.H  XVSHUF4I.W
//   Sched<[WriteVSHUF4I256, ReadVSHUF4I256]>
// XVSHUF4I.D
//   Sched<[WriteVSHUF4I256, ReadVSHUF4I256, ReadVSHUF4I256]>
//===----------------------------------------------------------------------===//

def WriteVHSELI256            : SchedWrite;  // 256-bit half select by imm
def ReadVHSELI256             : SchedRead;   // 256-bit half select by imm
//===----------------------------------------------------------------------===//
// XVHSELI.D
//   Sched<[WriteVHSELI256, ReadVHSELI256]>
//===----------------------------------------------------------------------===//

def WriteVPERMI128            : SchedWrite;  // 128-bit lane permute by imm
def WriteVPERMI256            : SchedWrite;  // 256-bit lane permute by imm
def ReadVPERMI128             : SchedRead;   // 128-bit lane permute by imm
def ReadVPERMI256             : SchedRead;   // 256-bit lane permute by imm
//===----------------------------------------------------------------------===//
// VPERMI.W
//   Sched<[WriteVPERMI128, ReadVPERMI128, ReadVPERMI128]>
// XVPERMI.W
//   Sched<[WriteVPERMI256, ReadVPERMI256, ReadVPERMI256]>
//===----------------------------------------------------------------------===//

def WriteVPERM256             : SchedWrite;  // 256-bit cross-lane permute
def ReadVPERM256              : SchedRead;   // 256-bit cross-lane permute
//===----------------------------------------------------------------------===//
// XVPERM.W  XVPERMI.Q
//   Sched<[WriteVPERM256, ReadVPERM256, ReadVPERM256]>
// XVPERMI.D
//   Sched<[WriteVPERM256, ReadVPERM256]>
//===----------------------------------------------------------------------===//

def WriteVEXTRINS128          : SchedWrite;  // 128-bit element extract + insert
def WriteVEXTRINS256          : SchedWrite;  // 256-bit element extract + insert
def ReadVEXTRINS128           : SchedRead;   // 128-bit element extract + insert
def ReadVEXTRINS256           : SchedRead;   // 256-bit element extract + insert
//===----------------------------------------------------------------------===//
// VEXTRINS.D  VEXTRINS.W  VEXTRINS.H  VEXTRINS.B
//   Sched<[WriteVEXTRINS128, ReadVEXTRINS128, ReadVEXTRINS128]>
// XVEXTRINS.D  XVEXTRINS.W  XVEXTRINS.H  XVEXTRINS.B
//   Sched<[WriteVEXTRINS256, ReadVEXTRINS256, ReadVEXTRINS256]>
//===----------------------------------------------------------------------===//

def WriteVPICKVE256           : SchedWrite;  // 256-bit cross-lane elem pick
def ReadVPICKVE256            : SchedRead;   // 256-bit cross-lane elem pick
//===----------------------------------------------------------------------===//
// XVPICKVE.W  XVPICKVE.D
//   Sched<[WriteVPICKVE256, ReadVPICKVE256]>
//===----------------------------------------------------------------------===//

def WriteVINSVE0256           : SchedWrite;  // 256-bit cross-lane elem insert
def ReadVINSVE0256            : SchedRead;   // 256-bit cross-lane elem insert
//===----------------------------------------------------------------------===//
// XVINSVE0.W  XVINSVE0.D
//   Sched<[WriteVINSVE0256, ReadVINSVE0256, ReadVINSVE0256]>
//===----------------------------------------------------------------------===//

def WriteVREPLVE0256          : SchedWrite;  // 256-bit cross-lane broadcast
def ReadVREPLVE0256           : SchedRead;   // 256-bit cross-lane broadcast
//===----------------------------------------------------------------------===//
// XVREPLVE0.B  XVREPLVE0.H  XVREPLVE0.W  XVREPLVE0.D  XVREPLVE0.Q
//   Sched<[WriteVREPLVE0256, ReadVREPLVE0256]>
//===----------------------------------------------------------------------===//

def WriteVREPLVEI128          : SchedWrite;  // 128-bit elem broadcast by imm
def WriteVREPLVEI256          : SchedWrite;  // 256-bit elem broadcast by imm
def ReadVREPLVEI128           : SchedRead;   // 128-bit elem broadcast by imm
def ReadVREPLVEI256           : SchedRead;   // 256-bit elem broadcast by imm
//===----------------------------------------------------------------------===//
// VREPLVEI.B  VREPLVEI.H  VREPLVEI.W  VREPLVEI.D
//   Sched<[WriteVREPLVEI128, ReadVREPLVEI128]>
// XVREPL128VEI.B  XVREPL128VEI.H  XVREPL128VEI.W  XVREPL128VEI.D
//   Sched<[WriteVREPLVEI256, ReadVREPLVEI256]>
//===----------------------------------------------------------------------===//

def WriteVREPLVE128           : SchedWrite;  // 128-bit elem broadcast by reg
def WriteVREPLVE256           : SchedWrite;  // 256-bit elem broadcast by reg
def ReadVREPLVE128            : SchedRead;   // 128-bit elem broadcast by reg
def ReadVREPLVE256            : SchedRead;   // 256-bit elem broadcast by reg
//===----------------------------------------------------------------------===//
// VREPLVE.B  VREPLVE.H  VREPLVE.W  VREPLVE.D
//   Sched<[WriteVREPLVE128, ReadVREPLVE128, ReadVREPLVE128]>
// XVREPLVE.B  XVREPLVE.H  XVREPLVE.W  XVREPLVE.D
//   Sched<[WriteVREPLVE256, ReadVREPLVE256, ReadVREPLVE256]>
//===----------------------------------------------------------------------===//

def WriteVREPLGR2VR128        : SchedWrite;  // 128-bit B/H broadcast from GPR
def WriteVREPLGR2VR256        : SchedWrite;  // 256-bit B/H broadcast from GPR
def ReadVREPLGR2VR128         : SchedRead;   // 128-bit B/H broadcast from GPR
def ReadVREPLGR2VR256         : SchedRead;   // 256-bit B/H broadcast from GPR
//===----------------------------------------------------------------------===//
// VREPLGR2VR.B  VREPLGR2VR.H
//   Sched<[WriteVREPLGR2VR128, ReadVREPLGR2VR128]>
// XVREPLGR2VR.B  XVREPLGR2VR.H
//   Sched<[WriteVREPLGR2VR256, ReadVREPLGR2VR256]>
//===----------------------------------------------------------------------===//

def WriteVREPLGR2VRWD128      : SchedWrite;  // 128-bit W/D broadcast from GPR
def WriteVREPLGR2VRWD256      : SchedWrite;  // 256-bit W/D broadcast from GPR
def ReadVREPLGR2VRWD128       : SchedRead;   // 128-bit W/D broadcast from GPR
def ReadVREPLGR2VRWD256       : SchedRead;   // 256-bit W/D broadcast from GPR
//===----------------------------------------------------------------------===//
// VREPLGR2VR.W  VREPLGR2VR.D
//   Sched<[WriteVREPLGR2VRWD128, ReadVREPLGR2VRWD128]>
// XVREPLGR2VR.W  XVREPLGR2VR.D
//   Sched<[WriteVREPLGR2VRWD256, ReadVREPLGR2VRWD256]>
//===----------------------------------------------------------------------===//

def WriteVINSGR2VR128         : SchedWrite;  // 128-bit elem insert from GPR
def WriteVINSGR2VR256         : SchedWrite;  // 256-bit elem insert from GPR
def ReadVINSGR2VR128          : SchedRead;   // 128-bit elem insert from GPR
def ReadVINSGR2VR256          : SchedRead;   // 256-bit elem insert from GPR
//===----------------------------------------------------------------------===//
// VINSGR2VR.B  VINSGR2VR.H  VINSGR2VR.W  VINSGR2VR.D
//   Sched<[WriteVINSGR2VR128, ReadVINSGR2VR128, ReadVINSGR2VR128]>
// XVINSGR2VR.W  XVINSGR2VR.D
//   Sched<[WriteVINSGR2VR256, ReadVINSGR2VR256, ReadVINSGR2VR256]>
//===----------------------------------------------------------------------===//

def WriteVPICKVE2GR128        : SchedWrite;  // 128-bit elem extract to GPR
def WriteVPICKVE2GR256        : SchedWrite;  // 256-bit elem extract to GPR
def ReadVPICKVE2GR128         : SchedRead;   // 128-bit elem extract to GPR
def ReadVPICKVE2GR256         : SchedRead;   // 256-bit elem extract to GPR
//===----------------------------------------------------------------------===//
// VPICKVE2GR.B  VPICKVE2GR.H  VPICKVE2GR.W  VPICKVE2GR.D  VPICKVE2GR.BU
// VPICKVE2GR.HU  VPICKVE2GR.WU  VPICKVE2GR.DU
//   Sched<[WriteVPICKVE2GR128, ReadVPICKVE2GR128]>
// XVPICKVE2GR.W  XVPICKVE2GR.D  XVPICKVE2GR.WU  XVPICKVE2GR.DU
//   Sched<[WriteVPICKVE2GR256, ReadVPICKVE2GR256]>
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Vector floating-point arithmetic
//===----------------------------------------------------------------------===//

def WriteVFADD128             : SchedWrite;  // 128-bit FP add / sub
def WriteVFADD256             : SchedWrite;  // 256-bit FP add / sub
def ReadVFADD128              : SchedRead;   // 128-bit FP add / sub
def ReadVFADD256              : SchedRead;   // 256-bit FP add / sub
//===----------------------------------------------------------------------===//
// VFADD.S  VFADD.D  VFSUB.S  VFSUB.D
//   Sched<[WriteVFADD128, ReadVFADD128, ReadVFADD128]>
// XVFADD.S  XVFADD.D  XVFSUB.S  XVFSUB.D
//   Sched<[WriteVFADD256, ReadVFADD256, ReadVFADD256]>
//===----------------------------------------------------------------------===//

def WriteVFMUL128             : SchedWrite;  // 128-bit FP multiply
def WriteVFMUL256             : SchedWrite;  // 256-bit FP multiply
def ReadVFMUL128              : SchedRead;   // 128-bit FP multiply
def ReadVFMUL256              : SchedRead;   // 256-bit FP multiply
//===----------------------------------------------------------------------===//
// VFMUL.S  VFMUL.D
//   Sched<[WriteVFMUL128, ReadVFMUL128, ReadVFMUL128]>
// XVFMUL.S  XVFMUL.D
//   Sched<[WriteVFMUL256, ReadVFMUL256, ReadVFMUL256]>
//===----------------------------------------------------------------------===//

def WriteVFMADD128            : SchedWrite;  // 128-bit FP fused multiply-add
def WriteVFMADD256            : SchedWrite;  // 256-bit FP fused multiply-add
def ReadVFMADD128             : SchedRead;   // 128-bit FP fused multiply-add
def ReadVFMADD256             : SchedRead;   // 256-bit FP fused multiply-add
//===----------------------------------------------------------------------===//
// VFMADD.S  VFMADD.D  VFMSUB.S  VFMSUB.D  VFNMADD.S  VFNMADD.D  VFNMSUB.S
// VFNMSUB.D
//   Sched<[WriteVFMADD128, ReadVFMADD128, ReadVFMADD128, ReadVFMADD128]>
// XVFMADD.S  XVFMADD.D  XVFMSUB.S  XVFMSUB.D  XVFNMADD.S  XVFNMADD.D
// XVFNMSUB.S  XVFNMSUB.D
//   Sched<[WriteVFMADD256, ReadVFMADD256, ReadVFMADD256, ReadVFMADD256]>
//===----------------------------------------------------------------------===//

def WriteVFMINMAX128          : SchedWrite;  // 128-bit FP min / max
def WriteVFMINMAX256          : SchedWrite;  // 256-bit FP min / max
def ReadVFMINMAX128           : SchedRead;   // 128-bit FP min / max
def ReadVFMINMAX256           : SchedRead;   // 256-bit FP min / max
//===----------------------------------------------------------------------===//
// VFMAX.S  VFMAX.D  VFMIN.S  VFMIN.D  VFMAXA.S  VFMAXA.D  VFMINA.S
// VFMINA.D
//   Sched<[WriteVFMINMAX128, ReadVFMINMAX128, ReadVFMINMAX128]>
// XVFMAX.S  XVFMAX.D  XVFMIN.S  XVFMIN.D  XVFMAXA.S  XVFMAXA.D  XVFMINA.S
// XVFMINA.D
//   Sched<[WriteVFMINMAX256, ReadVFMINMAX256, ReadVFMINMAX256]>
//===----------------------------------------------------------------------===//

def WriteVFCLASS128           : SchedWrite;  // 128-bit FP class
def WriteVFCLASS256           : SchedWrite;  // 256-bit FP class
def ReadVFCLASS128            : SchedRead;   // 128-bit FP class
def ReadVFCLASS256            : SchedRead;   // 256-bit FP class
//===----------------------------------------------------------------------===//
// VFCLASS.S  VFCLASS.D
//   Sched<[WriteVFCLASS128, ReadVFCLASS128]>
// XVFCLASS.S  XVFCLASS.D
//   Sched<[WriteVFCLASS256, ReadVFCLASS256]>
//===----------------------------------------------------------------------===//

def WriteVFLOGB128            : SchedWrite;  // 128-bit FP exponent extract
def WriteVFLOGB256            : SchedWrite;  // 256-bit FP exponent extract
def ReadVFLOGB128             : SchedRead;   // 128-bit FP exponent extract
def ReadVFLOGB256             : SchedRead;   // 256-bit FP exponent extract
//===----------------------------------------------------------------------===//
// VFLOGB.S  VFLOGB.D
//   Sched<[WriteVFLOGB128, ReadVFLOGB128]>
// XVFLOGB.S  XVFLOGB.D
//   Sched<[WriteVFLOGB256, ReadVFLOGB256]>
//===----------------------------------------------------------------------===//

def WriteVFDIVS128            : SchedWrite;  // 128-bit FP single divide
def WriteVFDIVS256            : SchedWrite;  // 256-bit FP single divide
def ReadVFDIVS128             : SchedRead;   // 128-bit FP single divide
def ReadVFDIVS256             : SchedRead;   // 256-bit FP single divide
//===----------------------------------------------------------------------===//
// VFDIV.S
//   Sched<[WriteVFDIVS128, ReadVFDIVS128, ReadVFDIVS128]>
// XVFDIV.S
//   Sched<[WriteVFDIVS256, ReadVFDIVS256, ReadVFDIVS256]>
//===----------------------------------------------------------------------===//

def WriteVFDIVD128            : SchedWrite;  // 128-bit FP double divide
def WriteVFDIVD256            : SchedWrite;  // 256-bit FP double divide
def ReadVFDIVD128             : SchedRead;   // 128-bit FP double divide
def ReadVFDIVD256             : SchedRead;   // 256-bit FP double divide
//===----------------------------------------------------------------------===//
// VFDIV.D
//   Sched<[WriteVFDIVD128, ReadVFDIVD128, ReadVFDIVD128]>
// XVFDIV.D
//   Sched<[WriteVFDIVD256, ReadVFDIVD256, ReadVFDIVD256]>
//===----------------------------------------------------------------------===//

def WriteVFSQRTS128           : SchedWrite;  // 128-bit FP single sqrt
def WriteVFSQRTS256           : SchedWrite;  // 256-bit FP single sqrt
def ReadVFSQRTS128            : SchedRead;   // 128-bit FP single sqrt
def ReadVFSQRTS256            : SchedRead;   // 256-bit FP single sqrt
//===----------------------------------------------------------------------===//
// VFSQRT.S
//   Sched<[WriteVFSQRTS128, ReadVFSQRTS128]>
// XVFSQRT.S
//   Sched<[WriteVFSQRTS256, ReadVFSQRTS256]>
//===----------------------------------------------------------------------===//

def WriteVFSQRTD128           : SchedWrite;  // 128-bit FP double sqrt
def WriteVFSQRTD256           : SchedWrite;  // 256-bit FP double sqrt
def ReadVFSQRTD128            : SchedRead;   // 128-bit FP double sqrt
def ReadVFSQRTD256            : SchedRead;   // 256-bit FP double sqrt
//===----------------------------------------------------------------------===//
// VFSQRT.D
//   Sched<[WriteVFSQRTD128, ReadVFSQRTD128]>
// XVFSQRT.D
//   Sched<[WriteVFSQRTD256, ReadVFSQRTD256]>
//===----------------------------------------------------------------------===//

def WriteVFRECIPS128          : SchedWrite;  // 128-bit FP single recip
def WriteVFRECIPS256          : SchedWrite;  // 256-bit FP single recip
def ReadVFRECIPS128           : SchedRead;   // 128-bit FP single recip
def ReadVFRECIPS256           : SchedRead;   // 256-bit FP single recip
//===----------------------------------------------------------------------===//
// VFRECIP.S
//   Sched<[WriteVFRECIPS128, ReadVFRECIPS128]>
// XVFRECIP.S
//   Sched<[WriteVFRECIPS256, ReadVFRECIPS256]>
//===----------------------------------------------------------------------===//

def WriteVFRECIPD128          : SchedWrite;  // 128-bit FP double recip
def WriteVFRECIPD256          : SchedWrite;  // 256-bit FP double recip
def ReadVFRECIPD128           : SchedRead;   // 128-bit FP double recip
def ReadVFRECIPD256           : SchedRead;   // 256-bit FP double recip
//===----------------------------------------------------------------------===//
// VFRECIP.D
//   Sched<[WriteVFRECIPD128, ReadVFRECIPD128]>
// XVFRECIP.D
//   Sched<[WriteVFRECIPD256, ReadVFRECIPD256]>
//===----------------------------------------------------------------------===//

def WriteVFRSQRTS128          : SchedWrite;  // 128-bit FP single rsqrt
def WriteVFRSQRTS256          : SchedWrite;  // 256-bit FP single rsqrt
def ReadVFRSQRTS128           : SchedRead;   // 128-bit FP single rsqrt
def ReadVFRSQRTS256           : SchedRead;   // 256-bit FP single rsqrt
//===----------------------------------------------------------------------===//
// VFRSQRT.S
//   Sched<[WriteVFRSQRTS128, ReadVFRSQRTS128]>
// XVFRSQRT.S
//   Sched<[WriteVFRSQRTS256, ReadVFRSQRTS256]>
//===----------------------------------------------------------------------===//

def WriteVFRSQRTD128          : SchedWrite;  // 128-bit FP double rsqrt
def WriteVFRSQRTD256          : SchedWrite;  // 256-bit FP double rsqrt
def ReadVFRSQRTD128           : SchedRead;   // 128-bit FP double rsqrt
def ReadVFRSQRTD256           : SchedRead;   // 256-bit FP double rsqrt
//===----------------------------------------------------------------------===//
// VFRSQRT.D
//   Sched<[WriteVFRSQRTD128, ReadVFRSQRTD128]>
// XVFRSQRT.D
//   Sched<[WriteVFRSQRTD256, ReadVFRSQRTD256]>
//===----------------------------------------------------------------------===//

def WriteVFRECIPE128          : SchedWrite;  // 128-bit FP approximate recip
def WriteVFRECIPE256          : SchedWrite;  // 256-bit FP approximate recip
def ReadVFRECIPE128           : SchedRead;   // 128-bit FP approximate recip
def ReadVFRECIPE256           : SchedRead;   // 256-bit FP approximate recip
//===----------------------------------------------------------------------===//
// VFRECIPE.S  VFRECIPE.D
//   Sched<[WriteVFRECIPE128, ReadVFRECIPE128]>
// XVFRECIPE.S  XVFRECIPE.D
//   Sched<[WriteVFRECIPE256, ReadVFRECIPE256]>
//===----------------------------------------------------------------------===//

def WriteVFRSQRTE128          : SchedWrite;  // 128-bit FP approximate rsqrt
def WriteVFRSQRTE256          : SchedWrite;  // 256-bit FP approximate rsqrt
def ReadVFRSQRTE128           : SchedRead;   // 128-bit FP approximate rsqrt
def ReadVFRSQRTE256           : SchedRead;   // 256-bit FP approximate rsqrt
//===----------------------------------------------------------------------===//
// VFRSQRTE.S  VFRSQRTE.D
//   Sched<[WriteVFRSQRTE128, ReadVFRSQRTE128]>
// XVFRSQRTE.S  XVFRSQRTE.D
//   Sched<[WriteVFRSQRTE256, ReadVFRSQRTE256]>
//===----------------------------------------------------------------------===//

def WriteVFRINT128            : SchedWrite;  // 128-bit FP round to integer
def WriteVFRINT256            : SchedWrite;  // 256-bit FP round to integer
def ReadVFRINT128             : SchedRead;   // 128-bit FP round to integer
def ReadVFRINT256             : SchedRead;   // 256-bit FP round to integer
//===----------------------------------------------------------------------===//
// VFRINTRNE.S  VFRINTRNE.D  VFRINTRZ.S  VFRINTRZ.D  VFRINTRP.S  VFRINTRP.D
// VFRINTRM.S  VFRINTRM.D  VFRINT.S  VFRINT.D
//   Sched<[WriteVFRINT128, ReadVFRINT128]>
// XVFRINTRNE.S  XVFRINTRNE.D  XVFRINTRZ.S  XVFRINTRZ.D  XVFRINTRP.S
// XVFRINTRP.D  XVFRINTRM.S  XVFRINTRM.D  XVFRINT.S  XVFRINT.D
//   Sched<[WriteVFRINT256, ReadVFRINT256]>
//===----------------------------------------------------------------------===//

def WriteVFCVT128             : SchedWrite;  // 128-bit FP precision convert
def WriteVFCVT256             : SchedWrite;  // 256-bit FP precision convert
def ReadVFCVT128              : SchedRead;   // 128-bit FP precision convert
def ReadVFCVT256              : SchedRead;   // 256-bit FP precision convert
//===----------------------------------------------------------------------===//
// VFCVTL.S.H  VFCVTH.S.H  VFCVTL.D.S  VFCVTH.D.S
//   Sched<[WriteVFCVT128, ReadVFCVT128]>
// VFCVT.H.S  VFCVT.S.D
//   Sched<[WriteVFCVT128, ReadVFCVT128, ReadVFCVT128]>
// XVFCVTL.S.H  XVFCVTH.S.H  XVFCVTL.D.S  XVFCVTH.D.S
//   Sched<[WriteVFCVT256, ReadVFCVT256]>
// XVFCVT.H.S  XVFCVT.S.D
//   Sched<[WriteVFCVT256, ReadVFCVT256, ReadVFCVT256]>
//===----------------------------------------------------------------------===//

def WriteVFTINT128            : SchedWrite;  // 128-bit FP to integer convert
def WriteVFTINT256            : SchedWrite;  // 256-bit FP to integer convert
def ReadVFTINT128             : SchedRead;   // 128-bit FP to integer convert
def ReadVFTINT256             : SchedRead;   // 256-bit FP to integer convert
//===----------------------------------------------------------------------===//
// VFTINTRNE.W.S  VFTINTRNE.L.D  VFTINTRZ.W.S  VFTINTRZ.L.D  VFTINTRP.W.S
// VFTINTRP.L.D  VFTINTRM.W.S  VFTINTRM.L.D  VFTINT.W.S  VFTINT.L.D
// VFTINTRZ.WU.S  VFTINTRZ.LU.D  VFTINT.WU.S  VFTINT.LU.D
//   Sched<[WriteVFTINT128, ReadVFTINT128]>
// XVFTINTRNE.W.S  XVFTINTRNE.L.D  XVFTINTRZ.W.S  XVFTINTRZ.L.D
// XVFTINTRP.W.S  XVFTINTRP.L.D  XVFTINTRM.W.S  XVFTINTRM.L.D  XVFTINT.W.S
// XVFTINT.L.D  XVFTINTRZ.WU.S  XVFTINTRZ.LU.D  XVFTINT.WU.S  XVFTINT.LU.D
//   Sched<[WriteVFTINT256, ReadVFTINT256]>
//===----------------------------------------------------------------------===//

def WriteVFTINTX128           : SchedWrite;  // 128-bit FP to integer, resized
def WriteVFTINTX256           : SchedWrite;  // 256-bit FP to integer, resized
def ReadVFTINTX128            : SchedRead;   // 128-bit FP to integer, resized
def ReadVFTINTX256            : SchedRead;   // 256-bit FP to integer, resized
//===----------------------------------------------------------------------===//
// VFTINTRNE.W.D  VFTINTRZ.W.D  VFTINTRP.W.D  VFTINTRM.W.D  VFTINT.W.D
//   Sched<[WriteVFTINTX128, ReadVFTINTX128, ReadVFTINTX128]>
// VFTINTRNEL.L.S  VFTINTRNEH.L.S  VFTINTRZL.L.S  VFTINTRZH.L.S
// VFTINTRPL.L.S  VFTINTRPH.L.S  VFTINTRML.L.S  VFTINTRMH.L.S  VFTINTL.L.S
// VFTINTH.L.S
//   Sched<[WriteVFTINTX128, ReadVFTINTX128]>
// XVFTINTRNE.W.D  XVFTINTRZ.W.D  XVFTINTRP.W.D  XVFTINTRM.W.D  XVFTINT.W.D
//   Sched<[WriteVFTINTX256, ReadVFTINTX256, ReadVFTINTX256]>
// XVFTINTRNEL.L.S  XVFTINTRNEH.L.S  XVFTINTRZL.L.S  XVFTINTRZH.L.S
// XVFTINTRPL.L.S  XVFTINTRPH.L.S  XVFTINTRML.L.S  XVFTINTRMH.L.S
// XVFTINTL.L.S  XVFTINTH.L.S
//   Sched<[WriteVFTINTX256, ReadVFTINTX256]>
//===----------------------------------------------------------------------===//

def WriteVFFINT128            : SchedWrite;  // 128-bit integer to FP convert
def WriteVFFINT256            : SchedWrite;  // 256-bit integer to FP convert
def ReadVFFINT128             : SchedRead;   // 128-bit integer to FP convert
def ReadVFFINT256             : SchedRead;   // 256-bit integer to FP convert
//===----------------------------------------------------------------------===//
// VFFINT.S.W  VFFINT.D.L  VFFINT.S.WU  VFFINT.D.LU
//   Sched<[WriteVFFINT128, ReadVFFINT128]>
// XVFFINT.S.W  XVFFINT.D.L  XVFFINT.S.WU  XVFFINT.D.LU
//   Sched<[WriteVFFINT256, ReadVFFINT256]>
//===----------------------------------------------------------------------===//

def WriteVFFINTX128           : SchedWrite;  // 128-bit integer to FP, resized
def WriteVFFINTX256           : SchedWrite;  // 256-bit integer to FP, resized
def ReadVFFINTX128            : SchedRead;   // 128-bit integer to FP, resized
def ReadVFFINTX256            : SchedRead;   // 256-bit integer to FP, resized
//===----------------------------------------------------------------------===//
// VFFINTL.D.W  VFFINTH.D.W
//   Sched<[WriteVFFINTX128, ReadVFFINTX128]>
// VFFINT.S.L
//   Sched<[WriteVFFINTX128, ReadVFFINTX128, ReadVFFINTX128]>
// XVFFINTL.D.W  XVFFINTH.D.W
//   Sched<[WriteVFFINTX256, ReadVFFINTX256]>
// XVFFINT.S.L
//   Sched<[WriteVFFINTX256, ReadVFFINTX256, ReadVFFINTX256]>
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// Vector memory access
//===----------------------------------------------------------------------===//

def WriteVLD128               : SchedWrite;  // 128-bit load, reg+imm
def WriteVLD256               : SchedWrite;  // 256-bit load, reg+imm
//===----------------------------------------------------------------------===//
// VLD
//   Sched<[WriteVLD128, ReadVMemBase128]>
// XVLD
//   Sched<[WriteVLD256, ReadVMemBase256]>
//===----------------------------------------------------------------------===//

def WriteVLDX128              : SchedWrite;  // 128-bit load, reg+reg
def WriteVLDX256              : SchedWrite;  // 256-bit load, reg+reg
//===----------------------------------------------------------------------===//
// VLDX
//   Sched<[WriteVLDX128, ReadVMemBase128, ReadVMemOffset128]>
// XVLDX
//   Sched<[WriteVLDX256, ReadVMemBase256, ReadVMemOffset256]>
//===----------------------------------------------------------------------===//

def WriteVLDREPL128           : SchedWrite;  // 128-bit load and broadcast
def WriteVLDREPL256           : SchedWrite;  // 256-bit load and broadcast
//===----------------------------------------------------------------------===//
// VLDREPL.B  VLDREPL.H  VLDREPL.W  VLDREPL.D
//   Sched<[WriteVLDREPL128, ReadVMemBase128]>
// XVLDREPL.B  XVLDREPL.H  XVLDREPL.W  XVLDREPL.D
//   Sched<[WriteVLDREPL256, ReadVMemBase256]>
//===----------------------------------------------------------------------===//

def WriteVST128               : SchedWrite;  // 128-bit store, reg+imm
def WriteVST256               : SchedWrite;  // 256-bit store, reg+imm
//===----------------------------------------------------------------------===//
// VST
//   Sched<[WriteVST128, ReadVSTData128, ReadVMemBase128]>
// XVST
//   Sched<[WriteVST256, ReadVSTData256, ReadVMemBase256]>
//===----------------------------------------------------------------------===//

def WriteVSTX128              : SchedWrite;  // 128-bit store, reg+reg
def WriteVSTX256              : SchedWrite;  // 256-bit store, reg+reg
//===----------------------------------------------------------------------===//
// VSTX
//   Sched<[WriteVSTX128, ReadVSTData128, ReadVMemBase128, ReadVMemOffset128]>
// XVSTX
//   Sched<[WriteVSTX256, ReadVSTData256, ReadVMemBase256, ReadVMemOffset256]>
//===----------------------------------------------------------------------===//

def WriteVSTELM128            : SchedWrite;  // 128-bit element store
def WriteVSTELM256            : SchedWrite;  // 256-bit element store
//===----------------------------------------------------------------------===//
// VSTELM.B  VSTELM.H  VSTELM.W  VSTELM.D
//   Sched<[WriteVSTELM128, ReadVSTData128, ReadVMemBase128]>
// XVSTELM.B  XVSTELM.H  XVSTELM.W  XVSTELM.D
//   Sched<[WriteVSTELM256, ReadVSTData256, ReadVMemBase256]>
//===----------------------------------------------------------------------===//

def ReadVSTData128            : SchedRead;   // 128-bit store data
def ReadVMemBase128           : SchedRead;   // 128-bit address base
def ReadVMemOffset128         : SchedRead;   // 128-bit address offset
def ReadVSTData256            : SchedRead;   // 256-bit store data
def ReadVMemBase256           : SchedRead;   // 256-bit address base
def ReadVMemOffset256         : SchedRead;   // 256-bit address offset

