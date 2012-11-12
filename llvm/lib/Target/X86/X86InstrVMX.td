//===-- X86InstrVMX.td - VMX Instruction Set Extension -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the instructions that make up the Intel VMX instruction
// set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// VMX instructions

// 66 0F 38 80
def INVEPT32 : I<0x80, MRMSrcMem, (outs), (ins GR32:$src1, i128mem:$src2),
               "invept\t{$src2, $src1|$src1, $src2}", []>, OpSize, T8,
               Requires<[In32BitMode]>;
def INVEPT64 : I<0x80, MRMSrcMem, (outs), (ins GR64:$src1, i128mem:$src2),
               "invept\t{$src2, $src1|$src1, $src2}", []>, OpSize, T8,
               Requires<[In64BitMode]>;
// 66 0F 38 81
def INVVPID32 : I<0x81, MRMSrcMem, (outs), (ins GR32:$src1, i128mem:$src2),
                "invvpid\t{$src2, $src1|$src1, $src2}", []>, OpSize, T8,
                Requires<[In32BitMode]>;
def INVVPID64 : I<0x81, MRMSrcMem, (outs), (ins GR64:$src1, i128mem:$src2),
                "invvpid\t{$src2, $src1|$src1, $src2}", []>, OpSize, T8,
                Requires<[In64BitMode]>;
// 0F 01 C1
def VMCALL : I<0x01, MRM_C1, (outs), (ins), "vmcall", []>, TB;
def VMCLEARm : I<0xC7, MRM6m, (outs), (ins i64mem:$vmcs),
  "vmclear\t$vmcs", []>, OpSize, TB;
// OF 01 D4
def VMFUNC : I<0x01, MRM_D4, (outs), (ins), "vmfunc", []>, TB;
// 0F 01 C2
def VMLAUNCH : I<0x01, MRM_C2, (outs), (ins), "vmlaunch", []>, TB;
// 0F 01 C3
def VMRESUME : I<0x01, MRM_C3, (outs), (ins), "vmresume", []>, TB;
def VMPTRLDm : I<0xC7, MRM6m, (outs), (ins i64mem:$vmcs),
  "vmptrld\t$vmcs", []>, TB;
def VMPTRSTm : I<0xC7, MRM7m, (outs i64mem:$vmcs), (ins),
  "vmptrst\t$vmcs", []>, TB;
def VMREAD64rm : I<0x78, MRMDestMem, (outs i64mem:$dst), (ins GR64:$src),
  "vmread{q}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In64BitMode]>;
def VMREAD64rr : I<0x78, MRMDestReg, (outs GR64:$dst), (ins GR64:$src),
  "vmread{q}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In64BitMode]>;
def VMREAD32rm : I<0x78, MRMDestMem, (outs i32mem:$dst), (ins GR32:$src),
  "vmread{l}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In32BitMode]>;
def VMREAD32rr : I<0x78, MRMDestReg, (outs GR32:$dst), (ins GR32:$src),
  "vmread{l}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In32BitMode]>;
def VMWRITE64rm : I<0x79, MRMSrcMem, (outs GR64:$dst), (ins i64mem:$src),
  "vmwrite{q}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In64BitMode]>;
def VMWRITE64rr : I<0x79, MRMSrcReg, (outs GR64:$dst), (ins GR64:$src),
  "vmwrite{q}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In64BitMode]>;
def VMWRITE32rm : I<0x79, MRMSrcMem, (outs GR32:$dst), (ins i32mem:$src),
  "vmwrite{l}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In32BitMode]>;
def VMWRITE32rr : I<0x79, MRMSrcReg, (outs GR32:$dst), (ins GR32:$src),
  "vmwrite{l}\t{$src, $dst|$dst, $src}", []>, TB, Requires<[In32BitMode]>;
// 0F 01 C4
def VMXOFF : I<0x01, MRM_C4, (outs), (ins), "vmxoff", []>, TB;
def VMXON : I<0xC7, MRM6m, (outs), (ins i64mem:$vmxon),
  "vmxon\t$vmxon", []>, XS;

