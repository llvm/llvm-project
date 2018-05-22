//===-- MIMGInstructions.td - MIMG Instruction Defintions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class MIMG_Mask <string op, int channels> {
  string Op = op;
  int Channels = channels;
}

class MIMG_Atomic_Size <string op, bit is32Bit> {
  string Op = op;
  int AtomicSize = !if(is32Bit, 1, 2);
}

class mimg <bits<7> si, bits<7> vi = si> {
  field bits<7> SI = si;
  field bits<7> VI = vi;
}

class MIMG_Helper <dag outs, dag ins, string asm,
                   string dns=""> : MIMG<outs, ins, asm,[]> {
  let mayLoad = 1;
  let mayStore = 0;
  let hasPostISelHook = 1;
  let DecoderNamespace = dns;
  let isAsmParserOnly = !if(!eq(dns,""), 1, 0);
  let AsmMatchConverter = "cvtMIMG";
  let usesCustomInserter = 1;
  let SchedRW = [WriteVMEM];
}

class MIMG_NoSampler_Helper <bits<7> op, string asm,
                             RegisterClass dst_rc,
                             RegisterClass addr_rc,
                             bit d16_bit=0,
                             string dns=""> : MIMG_Helper <
  (outs dst_rc:$vdata),
  (ins addr_rc:$vaddr, SReg_256:$srsrc,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"#!if(d16_bit, " d16", ""),
  dns>, MIMGe<op> {
  let ssamp = 0;
  let D16 = d16;
}

multiclass MIMG_NoSampler_Src_Helper_Helper <bits<7> op, string asm,
                                             RegisterClass dst_rc,
                                             int channels, bit d16_bit,
                                             string suffix> {
  def NAME # _V1 # suffix : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32, d16_bit,
                                                   !if(!eq(channels, 1), "AMDGPU", "")>,
                            MIMG_Mask<asm#"_V1"#suffix, channels>;
  def NAME # _V2 # suffix : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64, d16_bit>,
                            MIMG_Mask<asm#"_V2"#suffix, channels>;
  def NAME # _V3 # suffix : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_96, d16_bit>,
                            MIMG_Mask<asm#"_V3"#suffix, channels>;
  def NAME # _V4 # suffix : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128, d16_bit>,
                            MIMG_Mask<asm#"_V4"#suffix, channels>;
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                      RegisterClass dst_rc,
                                      int channels> {
  defm NAME : MIMG_NoSampler_Src_Helper_Helper <op, asm, dst_rc, channels, 0, "">;

  let d16 = 1 in {
    let SubtargetPredicate = HasPackedD16VMem in {
      defm NAME : MIMG_NoSampler_Src_Helper_Helper <op, asm, dst_rc, channels, 1, "_D16">;
    } // End HasPackedD16VMem.

    let SubtargetPredicate = HasUnpackedD16VMem, DecoderNamespace = "GFX80_UNPACKED" in {
      defm NAME : MIMG_NoSampler_Src_Helper_Helper <op, asm, dst_rc, channels, 1, "_D16_gfx80">;
    } // End HasUnpackedD16VMem.
  } // End d16 = 1.
}

multiclass MIMG_NoSampler <bits<7> op, string asm> {
  defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 4>;
}

multiclass MIMG_PckNoSampler <bits<7> op, string asm> {
  defm NAME # _V1 : MIMG_NoSampler_Src_Helper_Helper <op, asm, VGPR_32, 1, 0, "">;
  defm NAME # _V2 : MIMG_NoSampler_Src_Helper_Helper <op, asm, VReg_64, 2, 0, "">;
  defm NAME # _V3 : MIMG_NoSampler_Src_Helper_Helper <op, asm, VReg_96, 3, 0, "">;
  defm NAME # _V4 : MIMG_NoSampler_Src_Helper_Helper <op, asm, VReg_128, 4, 0, "">;
}

class MIMG_Store_Helper <bits<7> op, string asm,
                         RegisterClass data_rc,
                         RegisterClass addr_rc,
                         bit d16_bit=0,
                         string dns = ""> : MIMG_Helper <
  (outs),
  (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"#!if(d16_bit, " d16", ""), dns>, MIMGe<op> {
  let ssamp = 0;
  let mayLoad = 0;
  let mayStore = 1;
  let hasSideEffects = 0;
  let hasPostISelHook = 0;
  let DisableWQM = 1;
  let D16 = d16;
}

multiclass MIMG_Store_Addr_Helper_Helper <bits<7> op, string asm,
                                  RegisterClass data_rc,
                                  int channels, bit d16_bit,
                                  string suffix> {
  def NAME # _V1 # suffix : MIMG_Store_Helper <op, asm, data_rc, VGPR_32, d16_bit,
                                               !if(!eq(channels, 1), "AMDGPU", "")>,
                            MIMG_Mask<asm#"_V1"#suffix, channels>;
  def NAME # _V2 # suffix : MIMG_Store_Helper <op, asm, data_rc, VReg_64, d16_bit>,
                            MIMG_Mask<asm#"_V2"#suffix, channels>;
  def NAME # _V3 # suffix : MIMG_Store_Helper <op, asm, data_rc, VReg_96, d16_bit>,
                            MIMG_Mask<asm#"_V3"#suffix, channels>;
  def NAME # _V4 # suffix : MIMG_Store_Helper <op, asm, data_rc, VReg_128, d16_bit>,
                            MIMG_Mask<asm#"_V4"#suffix, channels>;
}

multiclass MIMG_Store_Addr_Helper <bits<7> op, string asm,
                                  RegisterClass data_rc,
                                  int channels> {
  defm NAME : MIMG_Store_Addr_Helper_Helper <op, asm, data_rc, channels, 0, "">;

  let d16 = 1 in {
    let SubtargetPredicate = HasPackedD16VMem in {
      defm NAME : MIMG_Store_Addr_Helper_Helper <op, asm, data_rc, channels, 1, "_D16">;
    } // End HasPackedD16VMem.

    let SubtargetPredicate = HasUnpackedD16VMem, DecoderNamespace = "GFX80_UNPACKED" in {
      defm NAME : MIMG_Store_Addr_Helper_Helper <op, asm, data_rc, channels, 1, "_D16_gfx80">;
    } // End HasUnpackedD16VMem.
  } // End d16 = 1.
}

multiclass MIMG_Store <bits<7> op, string asm> {
  defm _V1 : MIMG_Store_Addr_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_Store_Addr_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_Store_Addr_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_Store_Addr_Helper <op, asm, VReg_128, 4>;
}

multiclass MIMG_PckStore <bits<7> op, string asm> {
  defm NAME # _V1 : MIMG_Store_Addr_Helper_Helper <op, asm, VGPR_32, 1, 0, "">;
  defm NAME # _V2 : MIMG_Store_Addr_Helper_Helper <op, asm, VReg_64, 2, 0, "">;
  defm NAME # _V3 : MIMG_Store_Addr_Helper_Helper <op, asm, VReg_96, 3, 0, "">;
  defm NAME # _V4 : MIMG_Store_Addr_Helper_Helper <op, asm, VReg_128, 4, 0, "">;
}

class MIMG_Atomic_Helper <string asm, RegisterClass data_rc,
                          RegisterClass addr_rc, string dns="",
                          bit enableDasm = 0> : MIMG_Helper <
    (outs data_rc:$vdst),
    (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
         dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
         r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
    asm#" $vdst, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da",
    !if(enableDasm, dns, "")> {
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1; // FIXME: Remove this
  let hasPostISelHook = 0;
  let DisableWQM = 1;
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";
}

class MIMG_Atomic_Real_si<mimg op, string name, string asm,
  RegisterClass data_rc, RegisterClass addr_rc, bit enableDasm> :
  MIMG_Atomic_Helper<asm, data_rc, addr_rc, "SICI", enableDasm>,
  SIMCInstr<name, SIEncodingFamily.SI>,
  MIMGe<op.SI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isSICI];
  let DisableDecoder = DisableSIDecoder;
}

class MIMG_Atomic_Real_vi<mimg op, string name, string asm,
  RegisterClass data_rc, RegisterClass addr_rc, bit enableDasm> :
  MIMG_Atomic_Helper<asm, data_rc, addr_rc, "VI", enableDasm>,
  SIMCInstr<name, SIEncodingFamily.VI>,
  MIMGe<op.VI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isVI];
  let DisableDecoder = DisableVIDecoder;
}

multiclass MIMG_Atomic_Helper_m <mimg op,
                                 string name,
                                 string asm,
                                 string key,
                                 RegisterClass data_rc,
                                 RegisterClass addr_rc,
                                 bit is32Bit,
                                 bit enableDasm = 0> {
  let isPseudo = 1, isCodeGenOnly = 1 in {
    def "" : MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
             SIMCInstr<name, SIEncodingFamily.NONE>;
  }

  let ssamp = 0 in {
    def _si : MIMG_Atomic_Real_si<op, name, asm, data_rc, addr_rc, enableDasm>,
              MIMG_Atomic_Size<key # "_si", is32Bit>;

    def _vi : MIMG_Atomic_Real_vi<op, name, asm, data_rc, addr_rc, enableDasm>,
              MIMG_Atomic_Size<key # "_vi", is32Bit>;
  }
}

multiclass MIMG_Atomic_Addr_Helper_m <mimg op,
                                      string name,
                                      string asm,
                                      RegisterClass data_rc,
                                      bit is32Bit,
                                      bit enableDasm = 0> {
  // _V* variants have different address size, but the size is not encoded.
  // So only one variant can be disassembled. V1 looks the safest to decode.
  defm _V1 : MIMG_Atomic_Helper_m <op, name # "_V1", asm, asm # "_V1", data_rc, VGPR_32, is32Bit, enableDasm>;
  defm _V2 : MIMG_Atomic_Helper_m <op, name # "_V2", asm, asm # "_V2", data_rc, VReg_64, is32Bit>;
  defm _V3 : MIMG_Atomic_Helper_m <op, name # "_V3", asm, asm # "_V3", data_rc, VReg_96, is32Bit>;
  defm _V4 : MIMG_Atomic_Helper_m <op, name # "_V4", asm, asm # "_V4", data_rc, VReg_128, is32Bit>;
}

multiclass MIMG_Atomic <mimg op, string asm,
                        RegisterClass data_rc_32 = VGPR_32,   // 32-bit atomics
                        RegisterClass data_rc_64 = VReg_64> { // 64-bit atomics
  // _V* variants have different dst size, but the size is encoded implicitly,
  // using dmask and tfe. Only 32-bit variant is registered with disassembler.
  // Other variants are reconstructed by disassembler using dmask and tfe.
  defm _V1 : MIMG_Atomic_Addr_Helper_m <op, asm # "_V1", asm, data_rc_32, 1, 1>;
  defm _V2 : MIMG_Atomic_Addr_Helper_m <op, asm # "_V2", asm, data_rc_64, 0>;
}

class MIMG_Sampler_Helper <bits<7> op, string asm,
                           RegisterClass dst_rc,
                           RegisterClass src_rc,
                           bit wqm,
                           bit d16_bit=0,
                           string dns=""> : MIMG_Helper <
  (outs dst_rc:$vdata),
  (ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da"#!if(d16_bit, " d16", ""),
  dns>, MIMGe<op> {
  let WQM = wqm;
  let D16 = d16;
}

multiclass MIMG_Sampler_Src_Helper_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, bit wqm,
                                    bit d16_bit, string suffix> {
    def _V1 # suffix : MIMG_Sampler_Helper <op, asm, dst_rc, VGPR_32, wqm, d16_bit,
                                   !if(!eq(channels, 1), "AMDGPU", "")>,
                       MIMG_Mask<asm#"_V1"#suffix, channels>;
    def _V2 # suffix : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_64, wqm, d16_bit>,
                       MIMG_Mask<asm#"_V2"#suffix, channels>;
    def _V3 # suffix : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_96, wqm, d16_bit>,
                       MIMG_Mask<asm#"_V3"#suffix, channels>;
    def _V4 # suffix : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_128, wqm, d16_bit>,
                       MIMG_Mask<asm#"_V4"#suffix, channels>;
    def _V8 # suffix : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_256, wqm, d16_bit>,
                       MIMG_Mask<asm#"_V8"#suffix, channels>;
    def _V16 # suffix : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_512, wqm, d16_bit>,
                        MIMG_Mask<asm#"_V16"#suffix, channels>;
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, bit wqm> {
  defm : MIMG_Sampler_Src_Helper_Helper <op, asm, dst_rc, channels, wqm, 0, "">;

  let d16 = 1 in {
    let SubtargetPredicate = HasPackedD16VMem in {
      defm : MIMG_Sampler_Src_Helper_Helper <op, asm, dst_rc, channels, wqm, 1, "_D16">;
    } // End HasPackedD16VMem.

    let SubtargetPredicate = HasUnpackedD16VMem, DecoderNamespace = "GFX80_UNPACKED" in {
      defm : MIMG_Sampler_Src_Helper_Helper <op, asm, dst_rc, channels, wqm, 1, "_D16_gfx80">;
    } // End HasUnpackedD16VMem.
  } // End d16 = 1.
}

multiclass MIMG_Sampler <bits<7> op, string asm, bit wqm=0> {
  defm _V1 : MIMG_Sampler_Src_Helper<op, asm, VGPR_32, 1, wqm>;
  defm _V2 : MIMG_Sampler_Src_Helper<op, asm, VReg_64, 2, wqm>;
  defm _V3 : MIMG_Sampler_Src_Helper<op, asm, VReg_96, 3, wqm>;
  defm _V4 : MIMG_Sampler_Src_Helper<op, asm, VReg_128, 4, wqm>;
}

multiclass MIMG_Sampler_WQM <bits<7> op, string asm> : MIMG_Sampler<op, asm, 1>;

class MIMG_Gather_Helper <bits<7> op, string asm,
                          RegisterClass dst_rc,
                          RegisterClass src_rc,
                          bit wqm,
                          bit d16_bit=0,
                          string dns=""> : MIMG <
  (outs dst_rc:$vdata),
  (ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da"#!if(d16_bit, " d16", ""),
  []>, MIMGe<op> {
  let mayLoad = 1;
  let mayStore = 0;

  // DMASK was repurposed for GATHER4. 4 components are always
  // returned and DMASK works like a swizzle - it selects
  // the component to fetch. The only useful DMASK values are
  // 1=red, 2=green, 4=blue, 8=alpha. (e.g. 1 returns
  // (red,red,red,red) etc.) The ISA document doesn't mention
  // this.
  // Therefore, disable all code which updates DMASK by setting this:
  let Gather4 = 1;
  let hasPostISelHook = 0;
  let WQM = wqm;
  let D16 = d16;

  let DecoderNamespace = dns;
  let isAsmParserOnly = !if(!eq(dns,""), 1, 0);
}


multiclass MIMG_Gather_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    bit wqm, bit d16_bit,
                                    string prefix,
                                    string suffix> {
  def prefix # _V1 # suffix : MIMG_Gather_Helper <op, asm, dst_rc, VGPR_32, wqm, d16_bit, "AMDGPU">;
  def prefix # _V2 # suffix : MIMG_Gather_Helper <op, asm, dst_rc, VReg_64, wqm, d16_bit>;
  def prefix # _V3 # suffix : MIMG_Gather_Helper <op, asm, dst_rc, VReg_96, wqm, d16_bit>;
  def prefix # _V4 # suffix : MIMG_Gather_Helper <op, asm, dst_rc, VReg_128, wqm, d16_bit>;
  def prefix # _V8 # suffix : MIMG_Gather_Helper <op, asm, dst_rc, VReg_256, wqm, d16_bit>;
  def prefix # _V16 # suffix : MIMG_Gather_Helper <op, asm, dst_rc, VReg_512, wqm, d16_bit>;
}

multiclass MIMG_Gather <bits<7> op, string asm, bit wqm=0> {
  defm : MIMG_Gather_Src_Helper<op, asm, VReg_128, wqm, 0, "_V4", "">;

  let d16 = 1 in {
    let AssemblerPredicate = HasPackedD16VMem in {
      defm : MIMG_Gather_Src_Helper<op, asm, VReg_64, wqm, 1, "_V2", "_D16">;
    } // End HasPackedD16VMem.

    let AssemblerPredicate = HasUnpackedD16VMem, DecoderNamespace = "GFX80_UNPACKED" in {
      defm : MIMG_Gather_Src_Helper<op, asm, VReg_128, wqm, 1, "_V4", "_D16_gfx80">;
    } // End HasUnpackedD16VMem.
  } // End d16 = 1.
}

multiclass MIMG_Gather_WQM <bits<7> op, string asm> : MIMG_Gather<op, asm, 1>;

//===----------------------------------------------------------------------===//
// MIMG Instructions
//===----------------------------------------------------------------------===//
let SubtargetPredicate = isGCN in {
defm IMAGE_LOAD : MIMG_NoSampler <0x00000000, "image_load">;
defm IMAGE_LOAD_MIP : MIMG_NoSampler <0x00000001, "image_load_mip">;
defm IMAGE_LOAD_PCK : MIMG_PckNoSampler <0x00000002, "image_load_pck">;
defm IMAGE_LOAD_PCK_SGN : MIMG_PckNoSampler <0x00000003, "image_load_pck_sgn">;
defm IMAGE_LOAD_MIP_PCK : MIMG_PckNoSampler <0x00000004, "image_load_mip_pck">;
defm IMAGE_LOAD_MIP_PCK_SGN : MIMG_PckNoSampler <0x00000005, "image_load_mip_pck_sgn">;
defm IMAGE_STORE : MIMG_Store <0x00000008, "image_store">;
defm IMAGE_STORE_MIP : MIMG_Store <0x00000009, "image_store_mip">;
defm IMAGE_STORE_PCK : MIMG_PckStore <0x0000000a, "image_store_pck">;
defm IMAGE_STORE_MIP_PCK : MIMG_PckStore <0x0000000b, "image_store_mip_pck">;

let mayLoad = 0, mayStore = 0 in {
defm IMAGE_GET_RESINFO : MIMG_NoSampler <0x0000000e, "image_get_resinfo">;
}

defm IMAGE_ATOMIC_SWAP : MIMG_Atomic <mimg<0x0f, 0x10>, "image_atomic_swap">;
defm IMAGE_ATOMIC_CMPSWAP : MIMG_Atomic <mimg<0x10, 0x11>, "image_atomic_cmpswap", VReg_64, VReg_128>;
defm IMAGE_ATOMIC_ADD : MIMG_Atomic <mimg<0x11, 0x12>, "image_atomic_add">;
defm IMAGE_ATOMIC_SUB : MIMG_Atomic <mimg<0x12, 0x13>, "image_atomic_sub">;
//def IMAGE_ATOMIC_RSUB : MIMG_NoPattern_ <"image_atomic_rsub", 0x00000013>; -- not on VI
defm IMAGE_ATOMIC_SMIN : MIMG_Atomic <mimg<0x14>, "image_atomic_smin">;
defm IMAGE_ATOMIC_UMIN : MIMG_Atomic <mimg<0x15>, "image_atomic_umin">;
defm IMAGE_ATOMIC_SMAX : MIMG_Atomic <mimg<0x16>, "image_atomic_smax">;
defm IMAGE_ATOMIC_UMAX : MIMG_Atomic <mimg<0x17>, "image_atomic_umax">;
defm IMAGE_ATOMIC_AND : MIMG_Atomic <mimg<0x18>, "image_atomic_and">;
defm IMAGE_ATOMIC_OR : MIMG_Atomic <mimg<0x19>, "image_atomic_or">;
defm IMAGE_ATOMIC_XOR : MIMG_Atomic <mimg<0x1a>, "image_atomic_xor">;
defm IMAGE_ATOMIC_INC : MIMG_Atomic <mimg<0x1b>, "image_atomic_inc">;
defm IMAGE_ATOMIC_DEC : MIMG_Atomic <mimg<0x1c>, "image_atomic_dec">;
//def IMAGE_ATOMIC_FCMPSWAP : MIMG_NoPattern_ <"image_atomic_fcmpswap", 0x0000001d>; -- not on VI
//def IMAGE_ATOMIC_FMIN : MIMG_NoPattern_ <"image_atomic_fmin", 0x0000001e>; -- not on VI
//def IMAGE_ATOMIC_FMAX : MIMG_NoPattern_ <"image_atomic_fmax", 0x0000001f>; -- not on VI
defm IMAGE_SAMPLE           : MIMG_Sampler_WQM <0x00000020, "image_sample">;
defm IMAGE_SAMPLE_CL        : MIMG_Sampler_WQM <0x00000021, "image_sample_cl">;
defm IMAGE_SAMPLE_D         : MIMG_Sampler <0x00000022, "image_sample_d">;
defm IMAGE_SAMPLE_D_CL      : MIMG_Sampler <0x00000023, "image_sample_d_cl">;
defm IMAGE_SAMPLE_L         : MIMG_Sampler <0x00000024, "image_sample_l">;
defm IMAGE_SAMPLE_B         : MIMG_Sampler_WQM <0x00000025, "image_sample_b">;
defm IMAGE_SAMPLE_B_CL      : MIMG_Sampler_WQM <0x00000026, "image_sample_b_cl">;
defm IMAGE_SAMPLE_LZ        : MIMG_Sampler <0x00000027, "image_sample_lz">;
defm IMAGE_SAMPLE_C         : MIMG_Sampler_WQM <0x00000028, "image_sample_c">;
defm IMAGE_SAMPLE_C_CL      : MIMG_Sampler_WQM <0x00000029, "image_sample_c_cl">;
defm IMAGE_SAMPLE_C_D       : MIMG_Sampler <0x0000002a, "image_sample_c_d">;
defm IMAGE_SAMPLE_C_D_CL    : MIMG_Sampler <0x0000002b, "image_sample_c_d_cl">;
defm IMAGE_SAMPLE_C_L       : MIMG_Sampler <0x0000002c, "image_sample_c_l">;
defm IMAGE_SAMPLE_C_B       : MIMG_Sampler_WQM <0x0000002d, "image_sample_c_b">;
defm IMAGE_SAMPLE_C_B_CL    : MIMG_Sampler_WQM <0x0000002e, "image_sample_c_b_cl">;
defm IMAGE_SAMPLE_C_LZ      : MIMG_Sampler <0x0000002f, "image_sample_c_lz">;
defm IMAGE_SAMPLE_O         : MIMG_Sampler_WQM <0x00000030, "image_sample_o">;
defm IMAGE_SAMPLE_CL_O      : MIMG_Sampler_WQM <0x00000031, "image_sample_cl_o">;
defm IMAGE_SAMPLE_D_O       : MIMG_Sampler <0x00000032, "image_sample_d_o">;
defm IMAGE_SAMPLE_D_CL_O    : MIMG_Sampler <0x00000033, "image_sample_d_cl_o">;
defm IMAGE_SAMPLE_L_O       : MIMG_Sampler <0x00000034, "image_sample_l_o">;
defm IMAGE_SAMPLE_B_O       : MIMG_Sampler_WQM <0x00000035, "image_sample_b_o">;
defm IMAGE_SAMPLE_B_CL_O    : MIMG_Sampler_WQM <0x00000036, "image_sample_b_cl_o">;
defm IMAGE_SAMPLE_LZ_O      : MIMG_Sampler <0x00000037, "image_sample_lz_o">;
defm IMAGE_SAMPLE_C_O       : MIMG_Sampler_WQM <0x00000038, "image_sample_c_o">;
defm IMAGE_SAMPLE_C_CL_O    : MIMG_Sampler_WQM <0x00000039, "image_sample_c_cl_o">;
defm IMAGE_SAMPLE_C_D_O     : MIMG_Sampler <0x0000003a, "image_sample_c_d_o">;
defm IMAGE_SAMPLE_C_D_CL_O  : MIMG_Sampler <0x0000003b, "image_sample_c_d_cl_o">;
defm IMAGE_SAMPLE_C_L_O     : MIMG_Sampler <0x0000003c, "image_sample_c_l_o">;
defm IMAGE_SAMPLE_C_B_O     : MIMG_Sampler_WQM <0x0000003d, "image_sample_c_b_o">;
defm IMAGE_SAMPLE_C_B_CL_O  : MIMG_Sampler_WQM <0x0000003e, "image_sample_c_b_cl_o">;
defm IMAGE_SAMPLE_C_LZ_O    : MIMG_Sampler <0x0000003f, "image_sample_c_lz_o">;
defm IMAGE_GATHER4          : MIMG_Gather_WQM <0x00000040, "image_gather4">;
defm IMAGE_GATHER4_CL       : MIMG_Gather_WQM <0x00000041, "image_gather4_cl">;
defm IMAGE_GATHER4_L        : MIMG_Gather <0x00000044, "image_gather4_l">;
defm IMAGE_GATHER4_B        : MIMG_Gather_WQM <0x00000045, "image_gather4_b">;
defm IMAGE_GATHER4_B_CL     : MIMG_Gather_WQM <0x00000046, "image_gather4_b_cl">;
defm IMAGE_GATHER4_LZ       : MIMG_Gather <0x00000047, "image_gather4_lz">;
defm IMAGE_GATHER4_C        : MIMG_Gather_WQM <0x00000048, "image_gather4_c">;
defm IMAGE_GATHER4_C_CL     : MIMG_Gather_WQM <0x00000049, "image_gather4_c_cl">;
defm IMAGE_GATHER4_C_L      : MIMG_Gather <0x0000004c, "image_gather4_c_l">;
defm IMAGE_GATHER4_C_B      : MIMG_Gather_WQM <0x0000004d, "image_gather4_c_b">;
defm IMAGE_GATHER4_C_B_CL   : MIMG_Gather_WQM <0x0000004e, "image_gather4_c_b_cl">;
defm IMAGE_GATHER4_C_LZ     : MIMG_Gather <0x0000004f, "image_gather4_c_lz">;
defm IMAGE_GATHER4_O        : MIMG_Gather_WQM <0x00000050, "image_gather4_o">;
defm IMAGE_GATHER4_CL_O     : MIMG_Gather_WQM <0x00000051, "image_gather4_cl_o">;
defm IMAGE_GATHER4_L_O      : MIMG_Gather <0x00000054, "image_gather4_l_o">;
defm IMAGE_GATHER4_B_O      : MIMG_Gather_WQM <0x00000055, "image_gather4_b_o">;
defm IMAGE_GATHER4_B_CL_O   : MIMG_Gather <0x00000056, "image_gather4_b_cl_o">;
defm IMAGE_GATHER4_LZ_O     : MIMG_Gather <0x00000057, "image_gather4_lz_o">;
defm IMAGE_GATHER4_C_O      : MIMG_Gather_WQM <0x00000058, "image_gather4_c_o">;
defm IMAGE_GATHER4_C_CL_O   : MIMG_Gather_WQM <0x00000059, "image_gather4_c_cl_o">;
defm IMAGE_GATHER4_C_L_O    : MIMG_Gather <0x0000005c, "image_gather4_c_l_o">;
defm IMAGE_GATHER4_C_B_O    : MIMG_Gather_WQM <0x0000005d, "image_gather4_c_b_o">;
defm IMAGE_GATHER4_C_B_CL_O : MIMG_Gather_WQM <0x0000005e, "image_gather4_c_b_cl_o">;
defm IMAGE_GATHER4_C_LZ_O   : MIMG_Gather <0x0000005f, "image_gather4_c_lz_o">;

let mayLoad = 0, mayStore = 0 in {
defm IMAGE_GET_LOD          : MIMG_Sampler_WQM <0x00000060, "image_get_lod">;
}

defm IMAGE_SAMPLE_CD        : MIMG_Sampler <0x00000068, "image_sample_cd">;
defm IMAGE_SAMPLE_CD_CL     : MIMG_Sampler <0x00000069, "image_sample_cd_cl">;
defm IMAGE_SAMPLE_C_CD      : MIMG_Sampler <0x0000006a, "image_sample_c_cd">;
defm IMAGE_SAMPLE_C_CD_CL   : MIMG_Sampler <0x0000006b, "image_sample_c_cd_cl">;
defm IMAGE_SAMPLE_CD_O      : MIMG_Sampler <0x0000006c, "image_sample_cd_o">;
defm IMAGE_SAMPLE_CD_CL_O   : MIMG_Sampler <0x0000006d, "image_sample_cd_cl_o">;
defm IMAGE_SAMPLE_C_CD_O    : MIMG_Sampler <0x0000006e, "image_sample_c_cd_o">;
defm IMAGE_SAMPLE_C_CD_CL_O : MIMG_Sampler <0x0000006f, "image_sample_c_cd_cl_o">;
//def IMAGE_RSRC256 : MIMG_NoPattern_RSRC256 <"image_rsrc256", 0x0000007e>;
//def IMAGE_SAMPLER : MIMG_NoPattern_ <"image_sampler", 0x0000007f>;
}

/********** ============================== **********/
/********** Dimension-aware image patterns **********/
/********** ============================== **********/

class getDwordsType<int dwords> {
  int NumDwords = dwords;
  string suffix = !if(!lt(dwords, 1), ?,
                  !if(!eq(dwords, 1), "_V1",
                  !if(!eq(dwords, 2), "_V2",
                  !if(!le(dwords, 4), "_V4",
                  !if(!le(dwords, 8), "_V8",
                  !if(!le(dwords, 16), "_V16", ?))))));
  ValueType VT = !if(!lt(dwords, 1), ?,
                 !if(!eq(dwords, 1), f32,
                 !if(!eq(dwords, 2), v2f32,
                 !if(!le(dwords, 4), v4f32,
                 !if(!le(dwords, 8), v8f32,
                 !if(!le(dwords, 16), v16f32, ?))))));
  RegisterClass VReg = !if(!lt(dwords, 1), ?,
                       !if(!eq(dwords, 1), VGPR_32,
                       !if(!eq(dwords, 2), VReg_64,
                       !if(!le(dwords, 4), VReg_128,
                       !if(!le(dwords, 8), VReg_256,
                       !if(!le(dwords, 16), VReg_512, ?))))));
}

class makeRegSequence_Fold<int i, dag d> {
  int idx = i;
  dag lhs = d;
}

// Generate a dag node which returns a vector register of class RC into which
// the source operands given by names have been inserted (assuming that each
// name corresponds to an operand whose size is equal to a subregister).
class makeRegSequence<ValueType vt, RegisterClass RC, list<string> names> {
  dag ret =
    !if(!eq(!size(names), 1),
        !dag(COPY_TO_REGCLASS, [?, RC], [names[0], ?]),
        !foldl(makeRegSequence_Fold<0, (vt (IMPLICIT_DEF))>, names, f, name,
               makeRegSequence_Fold<
                 !add(f.idx, 1),
                 !con((INSERT_SUBREG f.lhs),
                      !dag(INSERT_SUBREG, [?,    !cast<SubRegIndex>("sub"#f.idx)],
                                          [name, ?]))>).lhs);
}

class ImageDimPattern<AMDGPUImageDimIntrinsic I,
                      string dop, ValueType dty,
                      string suffix = ""> : GCNPat<(undef), (undef)> {
  list<AMDGPUArg> AddrArgs = I.P.AddrDefaultArgs;
  getDwordsType AddrDwords = getDwordsType<!size(AddrArgs)>;

  Instruction MI =
    !cast<Instruction>(!strconcat("IMAGE_", I.P.OpMod, dop, AddrDwords.suffix, suffix));

  // DAG fragment to match data arguments (vdata for store/atomic, dmask
  // for non-atomic).
  dag MatchDataDag =
    !con(!dag(I, !foreach(arg, I.P.DataArgs, dty),
                 !foreach(arg, I.P.DataArgs, arg.Name)),
         !if(I.P.IsAtomic, (I), (I i32:$dmask)));

  // DAG fragment to match vaddr arguments.
  dag MatchAddrDag = !dag(I, !foreach(arg, AddrArgs, arg.Type.VT),
                             !foreach(arg, AddrArgs, arg.Name));

  // DAG fragment to match sampler resource and unorm arguments.
  dag MatchSamplerDag = !if(I.P.IsSample, (I v4i32:$sampler, i1:$unorm), (I));

  // DAG node that generates the MI vdata for store/atomic
  getDwordsType DataDwords = getDwordsType<!size(I.P.DataArgs)>;
  dag GenDataDag =
    !if(I.P.IsAtomic, (MI makeRegSequence<DataDwords.VT, DataDwords.VReg,
                                          !foreach(arg, I.P.DataArgs, arg.Name)>.ret),
    !if(!size(I.P.DataArgs), (MI $vdata), (MI)));

  // DAG node that generates the MI vaddr
  dag GenAddrDag = makeRegSequence<AddrDwords.VT, AddrDwords.VReg,
                                   !foreach(arg, AddrArgs, arg.Name)>.ret;
  // DAG fragments that generate various inline flags
  dag GenDmask =
    !if(I.P.IsAtomic, (MI !add(!shl(1, DataDwords.NumDwords), -1)),
                      (MI (as_i32imm $dmask)));
  dag GenGLC =
    !if(I.P.IsAtomic, (MI 1),
                      (MI (bitextract_imm<0> $cachepolicy)));

  dag MatchIntrinsic = !con(MatchDataDag,
                            MatchAddrDag,
                            (I v8i32:$rsrc),
                            MatchSamplerDag,
                            (I 0/*texfailctrl*/,
                               i32:$cachepolicy));
  let PatternToMatch =
    !if(!size(I.RetTypes), (dty MatchIntrinsic), MatchIntrinsic);

  bit IsCmpSwap = !and(I.P.IsAtomic, !eq(!size(I.P.DataArgs), 2));
  dag ImageInstruction =
    !con(GenDataDag,
         (MI GenAddrDag),
         (MI $rsrc),
         !if(I.P.IsSample, (MI $sampler), (MI)),
         GenDmask,
         !if(I.P.IsSample, (MI (as_i1imm $unorm)), (MI 1)),
         GenGLC,
         (MI (bitextract_imm<1> $cachepolicy),
             0, /* r128 */
             0, /* tfe */
             0 /*(as_i1imm $lwe)*/,
             { I.P.Dim.DA }));
  let ResultInstrs = [
    !if(IsCmpSwap, (EXTRACT_SUBREG ImageInstruction, sub0), ImageInstruction)
  ];
}

foreach intr = !listconcat(AMDGPUImageDimIntrinsics,
                           AMDGPUImageDimGetResInfoIntrinsics) in {
  def intr#_pat_v1 : ImageDimPattern<intr, "_V1", f32>;
  def intr#_pat_v2 : ImageDimPattern<intr, "_V2", v2f32>;
  def intr#_pat_v4 : ImageDimPattern<intr, "_V4", v4f32>;
}

// v2f16 and v4f16 are used as data types to signal that D16 should be used.
// However, they are not (always) legal types, and the SelectionDAG requires us
// to legalize them before running any patterns. So we legalize them by
// converting to an int type of equal size and using an internal 'd16helper'
// intrinsic instead which signifies both the use of D16 and actually allows
// this integer-based return type.
multiclass ImageDimD16Helper<AMDGPUImageDimIntrinsic I,
                             AMDGPUImageDimIntrinsic d16helper> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    def _unpacked_v1 : ImageDimPattern<I, "_V1", f16, "_D16_gfx80">;
    def _unpacked_v2 : ImageDimPattern<d16helper, "_V2", v2i32, "_D16_gfx80">;
    def _unpacked_v4 : ImageDimPattern<d16helper, "_V4", v4i32, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    def _packed_v1 : ImageDimPattern<I, "_V1", f16, "_D16">;
    def _packed_v2 : ImageDimPattern<I, "_V1", v2f16, "_D16">;
    def _packed_v4 : ImageDimPattern<d16helper, "_V2", v2i32, "_D16">;
  } // End HasPackedD16VMem.
}

foreach intr = AMDGPUImageDimIntrinsics in {
  def intr#_d16helper_profile : AMDGPUDimProfileCopy<intr.P> {
    let RetTypes = !foreach(ty, intr.P.RetTypes, llvm_any_ty);
    let DataArgs = !foreach(arg, intr.P.DataArgs, AMDGPUArg<llvm_any_ty, arg.Name>);
  }

  let TargetPrefix = "SI", isTarget = 1 in
  def int_SI_image_d16helper_ # intr.P.OpMod # intr.P.Dim.Name :
      AMDGPUImageDimIntrinsic<!cast<AMDGPUDimProfile>(intr#"_d16helper_profile"),
                              intr.IntrProperties, intr.Properties>;

  defm intr#_d16 :
      ImageDimD16Helper<
          intr, !cast<AMDGPUImageDimIntrinsic>(
                    "int_SI_image_d16helper_" # intr.P.OpMod # intr.P.Dim.Name)>;
}

foreach intr = AMDGPUImageDimGatherIntrinsics in {
  def intr#_pat3 : ImageDimPattern<intr, "_V4", v4f32>;

  def intr#_d16helper_profile : AMDGPUDimProfileCopy<intr.P> {
    let RetTypes = !foreach(ty, intr.P.RetTypes, llvm_any_ty);
    let DataArgs = !foreach(arg, intr.P.DataArgs, AMDGPUArg<llvm_any_ty, arg.Name>);
  }

  let TargetPrefix = "SI", isTarget = 1 in
  def int_SI_image_d16helper_ # intr.P.OpMod # intr.P.Dim.Name :
      AMDGPUImageDimIntrinsic<!cast<AMDGPUDimProfile>(intr#"_d16helper_profile"),
                              intr.IntrProperties, intr.Properties>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    def intr#_unpacked_v4 :
        ImageDimPattern<!cast<AMDGPUImageDimIntrinsic>(
                            "int_SI_image_d16helper_" # intr.P.OpMod # intr.P.Dim.Name),
                        "_V4", v4i32, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    def intr#_packed_v4 :
        ImageDimPattern<!cast<AMDGPUImageDimIntrinsic>(
                            "int_SI_image_d16helper_" # intr.P.OpMod # intr.P.Dim.Name),
                        "_V2", v2i32, "_D16">;
  } // End HasPackedD16VMem.
}

foreach intr = AMDGPUImageDimAtomicIntrinsics in {
  def intr#_pat1 : ImageDimPattern<intr, "_V1", i32>;
}

/********** ======================= **********/
/********** Image sampling patterns **********/
/********** ======================= **********/

// ImageSample for amdgcn
// TODO:
// 1. Handle v4i32 rsrc type (Register Class for the instruction to be SReg_128).
// 2. Add A16 support when we pass address of half type.
multiclass ImageSamplePattern<SDPatternOperator name, MIMG opcode, ValueType dt, ValueType vt>  {
  def : GCNPat<
    (dt (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, i32:$dmask, i1:$unorm, i1:$glc,
        i1:$slc, i1:$lwe, i1:$da)),
    (opcode $addr, $rsrc, $sampler,
          (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $slc),
          0, 0, (as_i1imm $lwe), (as_i1imm $da))
    >;
}

multiclass ImageSampleDataPatterns<SDPatternOperator name, string opcode, ValueType dt, string suffix = ""> {
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V1 # suffix), dt, f32>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V2 # suffix), dt, v2f32>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V4 # suffix), dt, v4f32>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V8 # suffix), dt, v8f32>;
  defm : ImageSamplePattern<name, !cast<MIMG>(opcode # _V16 # suffix), dt, v16f32>;
}

// ImageSample patterns.
multiclass ImageSamplePatterns<SDPatternOperator name, string opcode> {
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f32>;
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v2f32>;
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16">;
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), v2f16, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageSample alternative patterns for illegal vector half Types.
multiclass ImageSampleAltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16_gfx80">;
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4i32, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16">;
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageGather4 patterns.
multiclass ImageGather4Patterns<SDPatternOperator name, string opcode> {
  defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;
}

// ImageGather4 alternative patterns for illegal vector half Types.
multiclass ImageGather4AltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V4), v4i32, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageSampleDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageLoad for amdgcn.
multiclass ImageLoadPattern<SDPatternOperator name, MIMG opcode, ValueType dt, ValueType vt> {
  def : GCNPat <
    (dt (name vt:$addr, v8i32:$rsrc, i32:$dmask, i1:$glc, i1:$slc, i1:$lwe,
                i1:$da)),
    (opcode $addr, $rsrc,
          (as_i32imm $dmask), 1, (as_i1imm $glc), (as_i1imm $slc),
          0, 0, (as_i1imm $lwe), (as_i1imm $da))
  >;
}

multiclass ImageLoadDataPatterns<SDPatternOperator name, string opcode, ValueType dt, string suffix = ""> {
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V1 # suffix), dt, i32>;
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V2 # suffix), dt, v2i32>;
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V4 # suffix), dt, v4i32>;
}

// ImageLoad patterns.
// TODO: support v3f32.
multiclass ImageLoadPatterns<SDPatternOperator name, string opcode> {
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f32>;
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v2f32>;
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16">;
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), v2f16, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageLoad alternative patterns for illegal vector half Types.
multiclass ImageLoadAltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16_gfx80">;
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V4), v4i32,  "_D16_gfx80">;
  } // End HasUnPackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16">;
    defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageStore for amdgcn.
multiclass ImageStorePattern<SDPatternOperator name, MIMG opcode, ValueType dt, ValueType vt> {
  def : GCNPat <
    (name dt:$data, vt:$addr, v8i32:$rsrc, i32:$dmask, i1:$glc, i1:$slc,
          i1:$lwe, i1:$da),
    (opcode $data, $addr, $rsrc,
          (as_i32imm $dmask), 1, (as_i1imm $glc), (as_i1imm $slc),
          0, 0, (as_i1imm $lwe), (as_i1imm $da))
  >;
}

multiclass ImageStoreDataPatterns<SDPatternOperator name, string opcode, ValueType dt, string suffix = ""> {
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V1 # suffix), dt, i32>;
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V2 # suffix), dt, v2i32>;
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V4 # suffix), dt, v4i32>;
}

// ImageStore patterns.
// TODO: support v3f32.
multiclass ImageStorePatterns<SDPatternOperator name, string opcode> {
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f32>;
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2f32>;
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;

  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f16, "_D16">;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), v2f16, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageStore alternative patterns.
multiclass ImageStoreAltPatterns<SDPatternOperator name, string opcode> {
  let SubtargetPredicate = HasUnpackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16_gfx80">;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V4), v4i32, "_D16_gfx80">;
  } // End HasUnpackedD16VMem.

  let SubtargetPredicate = HasPackedD16VMem in {
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), i32, "_D16">;
    defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2i32, "_D16">;
  } // End HasPackedD16VMem.
}

// ImageAtomic for amdgcn.
class ImageAtomicPattern<SDPatternOperator name, MIMG opcode, ValueType vt> : GCNPat <
  (name i32:$vdata, vt:$addr, v8i32:$rsrc, imm:$r128, imm:$da, imm:$slc),
  (opcode $vdata, $addr, $rsrc, 1, 1, 1, (as_i1imm $slc), (as_i1imm $r128), 0, 0, (as_i1imm $da))
>;

// ImageAtomic patterns.
multiclass ImageAtomicPatterns<SDPatternOperator name, string opcode> {
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1_V1), i32>;
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1_V2), v2i32>;
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1_V4), v4i32>;
}

// ImageAtomicCmpSwap for amdgcn.
class ImageAtomicCmpSwapPattern<MIMG opcode, ValueType vt> : GCNPat <
  (int_amdgcn_image_atomic_cmpswap i32:$vsrc, i32:$vcmp, vt:$addr, v8i32:$rsrc,
                                   imm:$r128, imm:$da, imm:$slc),
  (EXTRACT_SUBREG
    (opcode (REG_SEQUENCE VReg_64, $vsrc, sub0, $vcmp, sub1),
            $addr, $rsrc, 3, 1, 1, (as_i1imm $slc), (as_i1imm $r128), 0, 0, (as_i1imm $da)),
    sub0)
>;

// ======= amdgcn Image Intrinsics ==============

// Image load.
defm : ImageLoadPatterns<int_amdgcn_image_load, "IMAGE_LOAD">;
defm : ImageLoadPatterns<int_amdgcn_image_load_mip, "IMAGE_LOAD_MIP">;
defm : ImageLoadPatterns<int_amdgcn_image_getresinfo, "IMAGE_GET_RESINFO">;
defm : ImageLoadAltPatterns<SIImage_load, "IMAGE_LOAD">;
defm : ImageLoadAltPatterns<SIImage_load_mip, "IMAGE_LOAD_MIP">;

// Image store.
defm : ImageStorePatterns<int_amdgcn_image_store, "IMAGE_STORE">;
defm : ImageStorePatterns<int_amdgcn_image_store_mip, "IMAGE_STORE_MIP">;
defm : ImageStoreAltPatterns<SIImage_store, "IMAGE_STORE">;
defm : ImageStoreAltPatterns<SIImage_store_mip, "IMAGE_STORE_MIP">;

// Basic sample.
defm : ImageSamplePatterns<int_amdgcn_image_sample,           "IMAGE_SAMPLE">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d,         "IMAGE_SAMPLE_D">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_l,         "IMAGE_SAMPLE_L">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b,         "IMAGE_SAMPLE_B">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison.
defm : ImageSamplePatterns<int_amdgcn_image_sample_c,         "IMAGE_SAMPLE_C">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets.
defm : ImageSamplePatterns<int_amdgcn_image_sample_o,         "IMAGE_SAMPLE_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets.
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : ImageSamplePatterns<int_amdgcn_image_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Basic gather4.
defm : ImageGather4Patterns<int_amdgcn_image_gather4,           "IMAGE_GATHER4">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_cl,        "IMAGE_GATHER4_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_l,         "IMAGE_GATHER4_L">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b,         "IMAGE_GATHER4_B">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b_cl,      "IMAGE_GATHER4_B_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_lz,        "IMAGE_GATHER4_LZ">;

// Gather4 with comparison.
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c,         "IMAGE_GATHER4_C">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_cl,      "IMAGE_GATHER4_C_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_l,       "IMAGE_GATHER4_C_L">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b,       "IMAGE_GATHER4_C_B">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b_cl,    "IMAGE_GATHER4_C_B_CL">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_lz,      "IMAGE_GATHER4_C_LZ">;

// Gather4 with offsets.
defm : ImageGather4Patterns<int_amdgcn_image_gather4_o,         "IMAGE_GATHER4_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_cl_o,      "IMAGE_GATHER4_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_l_o,       "IMAGE_GATHER4_L_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b_o,       "IMAGE_GATHER4_B_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_b_cl_o,    "IMAGE_GATHER4_B_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_lz_o,      "IMAGE_GATHER4_LZ_O">;

// Gather4 with comparison and offsets.
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_o,       "IMAGE_GATHER4_C_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_cl_o,    "IMAGE_GATHER4_C_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_l_o,     "IMAGE_GATHER4_C_L_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b_o,     "IMAGE_GATHER4_C_B_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_b_cl_o,  "IMAGE_GATHER4_C_B_CL_O">;
defm : ImageGather4Patterns<int_amdgcn_image_gather4_c_lz_o,    "IMAGE_GATHER4_C_LZ_O">;

// Basic sample alternative.
defm : ImageSampleAltPatterns<SIImage_sample,           "IMAGE_SAMPLE">;
defm : ImageSampleAltPatterns<SIImage_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_d,         "IMAGE_SAMPLE_D">;
defm : ImageSampleAltPatterns<SIImage_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_l,         "IMAGE_SAMPLE_L">;
defm : ImageSampleAltPatterns<SIImage_sample_b,         "IMAGE_SAMPLE_B">;
defm : ImageSampleAltPatterns<SIImage_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : ImageSampleAltPatterns<SIImage_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : ImageSampleAltPatterns<SIImage_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison alternative.
defm : ImageSampleAltPatterns<SIImage_sample_c,         "IMAGE_SAMPLE_C">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : ImageSampleAltPatterns<SIImage_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets alternative.
defm : ImageSampleAltPatterns<SIImage_sample_o,         "IMAGE_SAMPLE_O">;
defm : ImageSampleAltPatterns<SIImage_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : ImageSampleAltPatterns<SIImage_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : ImageSampleAltPatterns<SIImage_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : ImageSampleAltPatterns<SIImage_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : ImageSampleAltPatterns<SIImage_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : ImageSampleAltPatterns<SIImage_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets alternative.
defm : ImageSampleAltPatterns<SIImage_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : ImageSampleAltPatterns<SIImage_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Basic gather4 alternative.
defm : ImageGather4AltPatterns<SIImage_gather4,           "IMAGE_GATHER4">;
defm : ImageGather4AltPatterns<SIImage_gather4_cl,        "IMAGE_GATHER4_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_l,         "IMAGE_GATHER4_L">;
defm : ImageGather4AltPatterns<SIImage_gather4_b,         "IMAGE_GATHER4_B">;
defm : ImageGather4AltPatterns<SIImage_gather4_b_cl,      "IMAGE_GATHER4_B_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_lz,        "IMAGE_GATHER4_LZ">;

// Gather4 with comparison alternative.
defm : ImageGather4AltPatterns<SIImage_gather4_c,         "IMAGE_GATHER4_C">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_cl,      "IMAGE_GATHER4_C_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_l,       "IMAGE_GATHER4_C_L">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b,       "IMAGE_GATHER4_C_B">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b_cl,    "IMAGE_GATHER4_C_B_CL">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_lz,      "IMAGE_GATHER4_C_LZ">;

// Gather4 with offsets alternative.
defm : ImageGather4AltPatterns<SIImage_gather4_o,         "IMAGE_GATHER4_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_cl_o,      "IMAGE_GATHER4_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_l_o,       "IMAGE_GATHER4_L_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_b_o,       "IMAGE_GATHER4_B_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_b_cl_o,    "IMAGE_GATHER4_B_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_lz_o,      "IMAGE_GATHER4_LZ_O">;

// Gather4 with comparison and offsets alternative.
defm : ImageGather4AltPatterns<SIImage_gather4_c_o,       "IMAGE_GATHER4_C_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_cl_o,    "IMAGE_GATHER4_C_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_l_o,     "IMAGE_GATHER4_C_L_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b_o,     "IMAGE_GATHER4_C_B_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_b_cl_o,  "IMAGE_GATHER4_C_B_CL_O">;
defm : ImageGather4AltPatterns<SIImage_gather4_c_lz_o,    "IMAGE_GATHER4_C_LZ_O">;

defm : ImageSamplePatterns<int_amdgcn_image_getlod, "IMAGE_GET_LOD">;

// Image atomics
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_swap, "IMAGE_ATOMIC_SWAP">;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1_V1, i32>;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1_V2, v2i32>;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1_V4, v4i32>;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_add, "IMAGE_ATOMIC_ADD">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_sub, "IMAGE_ATOMIC_SUB">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_smin, "IMAGE_ATOMIC_SMIN">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_umin, "IMAGE_ATOMIC_UMIN">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_smax, "IMAGE_ATOMIC_SMAX">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_umax, "IMAGE_ATOMIC_UMAX">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_and, "IMAGE_ATOMIC_AND">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_or, "IMAGE_ATOMIC_OR">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_xor, "IMAGE_ATOMIC_XOR">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_inc, "IMAGE_ATOMIC_INC">;
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_dec, "IMAGE_ATOMIC_DEC">;

/* SIsample for simple 1D texture lookup */
def : GCNPat <
  (SIsample i32:$addr, v8i32:$rsrc, v4i32:$sampler, imm),
  (IMAGE_SAMPLE_V4_V1 $addr, $rsrc, $sampler, 0xf, 0, 0, 0, 0, 0, 0, 0)
>;

class SamplePattern<SDNode name, MIMG opcode, ValueType vt> : GCNPat <
    (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, imm),
    (opcode $addr, $rsrc, $sampler, 0xf, 0, 0, 0, 0, 0, 0, 0)
>;

class SampleRectPattern<SDNode name, MIMG opcode, ValueType vt> : GCNPat <
    (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, TEX_RECT),
    (opcode $addr, $rsrc, $sampler, 0xf, 1, 0, 0, 0, 0, 0, 0)
>;

class SampleArrayPattern<SDNode name, MIMG opcode, ValueType vt> : GCNPat <
    (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, TEX_ARRAY),
    (opcode $addr, $rsrc, $sampler, 0xf, 0, 0, 0, 0, 0, 0, 1)
>;

class SampleShadowPattern<SDNode name, MIMG opcode,
                          ValueType vt> : GCNPat <
    (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, TEX_SHADOW),
    (opcode $addr, $rsrc, $sampler, 0xf, 0, 0, 0, 0, 0, 0, 0)
>;

class SampleShadowArrayPattern<SDNode name, MIMG opcode,
                               ValueType vt> : GCNPat <
    (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, TEX_SHADOW_ARRAY),
    (opcode $addr, $rsrc, $sampler, 0xf, 0, 0, 0, 0, 0, 0, 1)
>;

/* SIsample* for texture lookups consuming more address parameters */
multiclass SamplePatterns<MIMG sample, MIMG sample_c, MIMG sample_l,
                          MIMG sample_c_l, MIMG sample_b, MIMG sample_c_b,
MIMG sample_d, MIMG sample_c_d, ValueType addr_type> {
  def : SamplePattern <SIsample, sample, addr_type>;
  def : SampleRectPattern <SIsample, sample, addr_type>;
  def : SampleArrayPattern <SIsample, sample, addr_type>;
  def : SampleShadowPattern <SIsample, sample_c, addr_type>;
  def : SampleShadowArrayPattern <SIsample, sample_c, addr_type>;

  def : SamplePattern <SIsamplel, sample_l, addr_type>;
  def : SampleArrayPattern <SIsamplel, sample_l, addr_type>;
  def : SampleShadowPattern <SIsamplel, sample_c_l, addr_type>;
  def : SampleShadowArrayPattern <SIsamplel, sample_c_l, addr_type>;

  def : SamplePattern <SIsampleb, sample_b, addr_type>;
  def : SampleArrayPattern <SIsampleb, sample_b, addr_type>;
  def : SampleShadowPattern <SIsampleb, sample_c_b, addr_type>;
  def : SampleShadowArrayPattern <SIsampleb, sample_c_b, addr_type>;

  def : SamplePattern <SIsampled, sample_d, addr_type>;
  def : SampleArrayPattern <SIsampled, sample_d, addr_type>;
  def : SampleShadowPattern <SIsampled, sample_c_d, addr_type>;
  def : SampleShadowArrayPattern <SIsampled, sample_c_d, addr_type>;
}

defm : SamplePatterns<IMAGE_SAMPLE_V4_V2, IMAGE_SAMPLE_C_V4_V2,
                      IMAGE_SAMPLE_L_V4_V2, IMAGE_SAMPLE_C_L_V4_V2,
                      IMAGE_SAMPLE_B_V4_V2, IMAGE_SAMPLE_C_B_V4_V2,
                      IMAGE_SAMPLE_D_V4_V2, IMAGE_SAMPLE_C_D_V4_V2,
                      v2i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V4, IMAGE_SAMPLE_C_V4_V4,
                      IMAGE_SAMPLE_L_V4_V4, IMAGE_SAMPLE_C_L_V4_V4,
                      IMAGE_SAMPLE_B_V4_V4, IMAGE_SAMPLE_C_B_V4_V4,
                      IMAGE_SAMPLE_D_V4_V4, IMAGE_SAMPLE_C_D_V4_V4,
                      v4i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V8, IMAGE_SAMPLE_C_V4_V8,
                      IMAGE_SAMPLE_L_V4_V8, IMAGE_SAMPLE_C_L_V4_V8,
                      IMAGE_SAMPLE_B_V4_V8, IMAGE_SAMPLE_C_B_V4_V8,
                      IMAGE_SAMPLE_D_V4_V8, IMAGE_SAMPLE_C_D_V4_V8,
                      v8i32>;
defm : SamplePatterns<IMAGE_SAMPLE_V4_V16, IMAGE_SAMPLE_C_V4_V16,
                      IMAGE_SAMPLE_L_V4_V16, IMAGE_SAMPLE_C_L_V4_V16,
                      IMAGE_SAMPLE_B_V4_V16, IMAGE_SAMPLE_C_B_V4_V16,
                      IMAGE_SAMPLE_D_V4_V16, IMAGE_SAMPLE_C_D_V4_V16,
                      v16i32>;
