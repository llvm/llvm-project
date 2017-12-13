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
                             string dns=""> : MIMG_Helper <
  (outs dst_rc:$vdata),
  (ins addr_rc:$vaddr, SReg_256:$srsrc,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da",
  dns>, MIMGe<op> {
  let ssamp = 0;
}

multiclass MIMG_NoSampler_Src_Helper <bits<7> op, string asm,
                                      RegisterClass dst_rc,
                                      int channels> {
  def _V1 : MIMG_NoSampler_Helper <op, asm, dst_rc, VGPR_32,
                                   !if(!eq(channels, 1), "AMDGPU", "")>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_NoSampler_Helper <op, asm, dst_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
}

multiclass MIMG_NoSampler <bits<7> op, string asm> {
  defm _V1 : MIMG_NoSampler_Src_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_NoSampler_Src_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_NoSampler_Src_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_NoSampler_Src_Helper <op, asm, VReg_128, 4>;
}

class MIMG_Store_Helper <bits<7> op, string asm,
                         RegisterClass data_rc,
                         RegisterClass addr_rc,
                         string dns = ""> : MIMG_Helper <
  (outs),
  (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da", dns>, MIMGe<op> {
  let ssamp = 0;
  let mayLoad = 1; // TableGen requires this for matching with the intrinsics
  let mayStore = 1;
  let hasSideEffects = 1;
  let hasPostISelHook = 0;
  let DisableWQM = 1;
}

multiclass MIMG_Store_Addr_Helper <bits<7> op, string asm,
                                  RegisterClass data_rc,
                                  int channels> {
  def _V1 : MIMG_Store_Helper <op, asm, data_rc, VGPR_32,
                               !if(!eq(channels, 1), "AMDGPU", "")>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Store_Helper <op, asm, data_rc, VReg_64>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Store_Helper <op, asm, data_rc, VReg_128>,
            MIMG_Mask<asm#"_V4", channels>;
}

multiclass MIMG_Store <bits<7> op, string asm> {
  defm _V1 : MIMG_Store_Addr_Helper <op, asm, VGPR_32, 1>;
  defm _V2 : MIMG_Store_Addr_Helper <op, asm, VReg_64, 2>;
  defm _V3 : MIMG_Store_Addr_Helper <op, asm, VReg_96, 3>;
  defm _V4 : MIMG_Store_Addr_Helper <op, asm, VReg_128, 4>;
}

class MIMG_Atomic_Helper <string asm, RegisterClass data_rc,
                          RegisterClass addr_rc> : MIMG_Helper <
    (outs data_rc:$vdst),
    (ins data_rc:$vdata, addr_rc:$vaddr, SReg_256:$srsrc,
         dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
         r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
    asm#" $vdst, $vaddr, $srsrc$dmask$unorm$glc$slc$r128$tfe$lwe$da"
  > {
  let mayStore = 1;
  let hasSideEffects = 1;
  let hasPostISelHook = 0;
  let DisableWQM = 1;
  let Constraints = "$vdst = $vdata";
  let AsmMatchConverter = "cvtMIMGAtomic";
}

class MIMG_Atomic_Real_si<mimg op, string name, string asm,
  RegisterClass data_rc, RegisterClass addr_rc> :
  MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
  SIMCInstr<name, SIEncodingFamily.SI>,
  MIMGe<op.SI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isSICI];
  let DecoderNamespace = "SICI";
  let DisableDecoder = DisableSIDecoder;
}

class MIMG_Atomic_Real_vi<mimg op, string name, string asm,
  RegisterClass data_rc, RegisterClass addr_rc> :
  MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
  SIMCInstr<name, SIEncodingFamily.VI>,
  MIMGe<op.VI> {
  let isCodeGenOnly = 0;
  let AssemblerPredicates = [isVI];
  let DecoderNamespace = "VI";
  let DisableDecoder = DisableVIDecoder;
}

multiclass MIMG_Atomic_Helper_m <mimg op, string name, string asm,
                                 RegisterClass data_rc, RegisterClass addr_rc> {
  let isPseudo = 1, isCodeGenOnly = 1 in {
    def "" : MIMG_Atomic_Helper<asm, data_rc, addr_rc>,
             SIMCInstr<name, SIEncodingFamily.NONE>;
  }

  let ssamp = 0 in {
    def _si : MIMG_Atomic_Real_si<op, name, asm, data_rc, addr_rc>;

    def _vi : MIMG_Atomic_Real_vi<op, name, asm, data_rc, addr_rc>;
  }
}

multiclass MIMG_Atomic <mimg op, string asm, RegisterClass data_rc = VGPR_32> {
  defm _V1 : MIMG_Atomic_Helper_m <op, asm # "_V1", asm, data_rc, VGPR_32>;
  defm _V2 : MIMG_Atomic_Helper_m <op, asm # "_V2", asm, data_rc, VReg_64>;
  defm _V4 : MIMG_Atomic_Helper_m <op, asm # "_V3", asm, data_rc, VReg_128>;
}

class MIMG_Sampler_Helper <bits<7> op, string asm,
                           RegisterClass dst_rc,
                           RegisterClass src_rc,
                           bit wqm,
                           string dns=""> : MIMG_Helper <
  (outs dst_rc:$vdata),
  (ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da",
  dns>, MIMGe<op> {
  let WQM = wqm;
}

multiclass MIMG_Sampler_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, bit wqm> {
  def _V1 : MIMG_Sampler_Helper <op, asm, dst_rc, VGPR_32, wqm,
                                 !if(!eq(channels, 1), "AMDGPU", "")>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_64, wqm>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_128, wqm>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_256, wqm>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Sampler_Helper <op, asm, dst_rc, VReg_512, wqm>,
            MIMG_Mask<asm#"_V16", channels>;
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
                          RegisterClass src_rc, bit wqm> : MIMG <
  (outs dst_rc:$vdata),
  (ins src_rc:$vaddr, SReg_256:$srsrc, SReg_128:$ssamp,
       dmask:$dmask, unorm:$unorm, GLC:$glc, slc:$slc,
       r128:$r128, tfe:$tfe, lwe:$lwe, da:$da),
  asm#" $vdata, $vaddr, $srsrc, $ssamp$dmask$unorm$glc$slc$r128$tfe$lwe$da",
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

  let isAsmParserOnly = 1; // TBD: fix it later
}

multiclass MIMG_Gather_Src_Helper <bits<7> op, string asm,
                                    RegisterClass dst_rc,
                                    int channels, bit wqm> {
  def _V1 : MIMG_Gather_Helper <op, asm, dst_rc, VGPR_32, wqm>,
            MIMG_Mask<asm#"_V1", channels>;
  def _V2 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_64, wqm>,
            MIMG_Mask<asm#"_V2", channels>;
  def _V4 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_128, wqm>,
            MIMG_Mask<asm#"_V4", channels>;
  def _V8 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_256, wqm>,
            MIMG_Mask<asm#"_V8", channels>;
  def _V16 : MIMG_Gather_Helper <op, asm, dst_rc, VReg_512, wqm>,
            MIMG_Mask<asm#"_V16", channels>;
}

multiclass MIMG_Gather <bits<7> op, string asm, bit wqm=0> {
  defm _V1 : MIMG_Gather_Src_Helper<op, asm, VGPR_32, 1, wqm>;
  defm _V2 : MIMG_Gather_Src_Helper<op, asm, VReg_64, 2, wqm>;
  defm _V3 : MIMG_Gather_Src_Helper<op, asm, VReg_96, 3, wqm>;
  defm _V4 : MIMG_Gather_Src_Helper<op, asm, VReg_128, 4, wqm>;
}

multiclass MIMG_Gather_WQM <bits<7> op, string asm> : MIMG_Gather<op, asm, 1>;

//===----------------------------------------------------------------------===//
// MIMG Instructions
//===----------------------------------------------------------------------===//
let SubtargetPredicate = isGCN in {
defm IMAGE_LOAD : MIMG_NoSampler <0x00000000, "image_load">;
defm IMAGE_LOAD_MIP : MIMG_NoSampler <0x00000001, "image_load_mip">;
//def IMAGE_LOAD_PCK : MIMG_NoPattern_ <"image_load_pck", 0x00000002>;
//def IMAGE_LOAD_PCK_SGN : MIMG_NoPattern_ <"image_load_pck_sgn", 0x00000003>;
//def IMAGE_LOAD_MIP_PCK : MIMG_NoPattern_ <"image_load_mip_pck", 0x00000004>;
//def IMAGE_LOAD_MIP_PCK_SGN : MIMG_NoPattern_ <"image_load_mip_pck_sgn", 0x00000005>;
defm IMAGE_STORE : MIMG_Store <0x00000008, "image_store">;
defm IMAGE_STORE_MIP : MIMG_Store <0x00000009, "image_store_mip">;
//def IMAGE_STORE_PCK : MIMG_NoPattern_ <"image_store_pck", 0x0000000a>;
//def IMAGE_STORE_MIP_PCK : MIMG_NoPattern_ <"image_store_mip_pck", 0x0000000b>;

let mayLoad = 0, mayStore = 0 in {
defm IMAGE_GET_RESINFO : MIMG_NoSampler <0x0000000e, "image_get_resinfo">;
}

defm IMAGE_ATOMIC_SWAP : MIMG_Atomic <mimg<0x0f, 0x10>, "image_atomic_swap">;
defm IMAGE_ATOMIC_CMPSWAP : MIMG_Atomic <mimg<0x10, 0x11>, "image_atomic_cmpswap", VReg_64>;
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

/********** ======================= **********/
/********** Image sampling patterns **********/
/********** ======================= **********/

// Image + sampler
class SampleRawPattern<SDPatternOperator name, MIMG opcode, ValueType vt> : GCNPat <
  (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, i32:$dmask, i32:$unorm,
        i32:$r128, i32:$da, i32:$glc, i32:$slc, i32:$tfe, i32:$lwe),
  (opcode $addr, $rsrc, $sampler,
          (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $slc),
          (as_i1imm $r128), (as_i1imm $tfe), (as_i1imm $lwe), (as_i1imm $da))
>;

multiclass SampleRawPatterns<SDPatternOperator name, string opcode> {
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V1), i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V2), v2i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V4), v4i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V8), v8i32>;
  def : SampleRawPattern<name, !cast<MIMG>(opcode # _V4_V16), v16i32>;
}

// Image + sampler for amdgcn
// TODO:
// 1. Handle half data type like v4f16, and add D16 bit support;
// 2. Handle v4i32 rsrc type (Register Class for the instruction to be SReg_128).
// 3. Add A16 support when we pass address of half type.
multiclass AMDGCNSamplePattern<SDPatternOperator name, MIMG opcode, ValueType dt, ValueType vt>  {
  def : GCNPat<
    (dt (name vt:$addr, v8i32:$rsrc, v4i32:$sampler, i32:$dmask, i1:$unorm, i1:$glc,
        i1:$slc, i1:$lwe, i1:$da)),
    (opcode $addr, $rsrc, $sampler,
          (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $slc),
          0, 0, (as_i1imm $lwe), (as_i1imm $da))
    >;
}

multiclass AMDGCNSampleDataPatterns<SDPatternOperator name, string opcode, ValueType dt> {
  defm : AMDGCNSamplePattern<name, !cast<MIMG>(opcode # _V1), dt, f32>;
  defm : AMDGCNSamplePattern<name, !cast<MIMG>(opcode # _V2), dt, v2f32>;
  defm : AMDGCNSamplePattern<name, !cast<MIMG>(opcode # _V4), dt, v4f32>;
  defm : AMDGCNSamplePattern<name, !cast<MIMG>(opcode # _V8), dt, v8f32>;
  defm : AMDGCNSamplePattern<name, !cast<MIMG>(opcode # _V16), dt, v16f32>;
}

// TODO: support v3f32.
multiclass AMDGCNSamplePatterns<SDPatternOperator name, string opcode> {
  defm : AMDGCNSampleDataPatterns<name, !cast<string>(opcode # _V1), f32>;
  defm : AMDGCNSampleDataPatterns<name, !cast<string>(opcode # _V2), v2f32>;
  defm : AMDGCNSampleDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;
}

// Image only
class ImagePattern<SDPatternOperator name, MIMG opcode, ValueType vt> : GCNPat <
  (name vt:$addr, v8i32:$rsrc, imm:$dmask, imm:$unorm,
        imm:$r128, imm:$da, imm:$glc, imm:$slc, imm:$tfe, imm:$lwe),
  (opcode $addr, $rsrc,
          (as_i32imm $dmask), (as_i1imm $unorm), (as_i1imm $glc), (as_i1imm $slc),
          (as_i1imm $r128), (as_i1imm $tfe), (as_i1imm $lwe), (as_i1imm $da))
>;

multiclass ImagePatterns<SDPatternOperator name, string opcode> {
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V1), i32>;
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V2), v2i32>;
  def : ImagePattern<name, !cast<MIMG>(opcode # _V4_V4), v4i32>;
}

multiclass ImageLoadPattern<SDPatternOperator name, MIMG opcode, ValueType dt, ValueType vt> {
  def : GCNPat <
    (dt (name vt:$addr, v8i32:$rsrc, i32:$dmask, i1:$glc, i1:$slc, i1:$lwe,
                i1:$da)),
    (opcode $addr, $rsrc,
          (as_i32imm $dmask), 1, (as_i1imm $glc), (as_i1imm $slc),
          0, 0, (as_i1imm $lwe), (as_i1imm $da))
  >;
}

multiclass ImageLoadDataPatterns<SDPatternOperator name, string opcode, ValueType dt> {
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V1), dt, i32>;
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V2), dt, v2i32>;
  defm : ImageLoadPattern<name, !cast<MIMG>(opcode # _V4), dt, v4i32>;
}

// TODO: support v3f32.
multiclass ImageLoadPatterns<SDPatternOperator name, string opcode> {
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V1), f32>;
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V2), v2f32>;
  defm : ImageLoadDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;
}

multiclass ImageStorePattern<SDPatternOperator name, MIMG opcode, ValueType dt, ValueType vt> {
  def : GCNPat <
    (name dt:$data, vt:$addr, v8i32:$rsrc, i32:$dmask, i1:$glc, i1:$slc,
          i1:$lwe, i1:$da),
    (opcode $data, $addr, $rsrc,
          (as_i32imm $dmask), 1, (as_i1imm $glc), (as_i1imm $slc),
          0, 0, (as_i1imm $lwe), (as_i1imm $da))
  >;
}

multiclass ImageStoreDataPatterns<SDPatternOperator name, string opcode, ValueType dt> {
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V1), dt, i32>;
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V2), dt, v2i32>;
  defm : ImageStorePattern<name, !cast<MIMG>(opcode # _V4), dt, v4i32>;
}

// TODO: support v3f32.
multiclass ImageStorePatterns<SDPatternOperator name, string opcode> {
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V1), f32>;
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V2), v2f32>;
  defm : ImageStoreDataPatterns<name, !cast<string>(opcode # _V4), v4f32>;
}

class ImageAtomicPattern<SDPatternOperator name, MIMG opcode, ValueType vt> : GCNPat <
  (name i32:$vdata, vt:$addr, v8i32:$rsrc, imm:$r128, imm:$da, imm:$slc),
  (opcode $vdata, $addr, $rsrc, 1, 1, 1, (as_i1imm $slc), (as_i1imm $r128), 0, 0, (as_i1imm $da))
>;

multiclass ImageAtomicPatterns<SDPatternOperator name, string opcode> {
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V1), i32>;
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V2), v2i32>;
  def : ImageAtomicPattern<name, !cast<MIMG>(opcode # _V4), v4i32>;
}

class ImageAtomicCmpSwapPattern<MIMG opcode, ValueType vt> : GCNPat <
  (int_amdgcn_image_atomic_cmpswap i32:$vsrc, i32:$vcmp, vt:$addr, v8i32:$rsrc,
                                   imm:$r128, imm:$da, imm:$slc),
  (EXTRACT_SUBREG
    (opcode (REG_SEQUENCE VReg_64, $vsrc, sub0, $vcmp, sub1),
            $addr, $rsrc, 3, 1, 1, (as_i1imm $slc), (as_i1imm $r128), 0, 0, (as_i1imm $da)),
    sub0)
>;

// ======= amdgcn Image Intrinsics ==============

// Image load
defm : ImageLoadPatterns<int_amdgcn_image_load, "IMAGE_LOAD">;
defm : ImageLoadPatterns<int_amdgcn_image_load_mip, "IMAGE_LOAD_MIP">;
defm : ImageLoadPatterns<int_amdgcn_image_getresinfo, "IMAGE_GET_RESINFO">;

// Image store
defm : ImageStorePatterns<int_amdgcn_image_store, "IMAGE_STORE">;
defm : ImageStorePatterns<int_amdgcn_image_store_mip, "IMAGE_STORE_MIP">;

// Basic sample
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample,           "IMAGE_SAMPLE">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_cl,        "IMAGE_SAMPLE_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_d,         "IMAGE_SAMPLE_D">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_d_cl,      "IMAGE_SAMPLE_D_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_l,         "IMAGE_SAMPLE_L">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_b,         "IMAGE_SAMPLE_B">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_b_cl,      "IMAGE_SAMPLE_B_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_lz,        "IMAGE_SAMPLE_LZ">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_cd,        "IMAGE_SAMPLE_CD">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_cd_cl,     "IMAGE_SAMPLE_CD_CL">;

// Sample with comparison
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c,         "IMAGE_SAMPLE_C">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_cl,      "IMAGE_SAMPLE_C_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_d,       "IMAGE_SAMPLE_C_D">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_d_cl,    "IMAGE_SAMPLE_C_D_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_l,       "IMAGE_SAMPLE_C_L">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_b,       "IMAGE_SAMPLE_C_B">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_b_cl,    "IMAGE_SAMPLE_C_B_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_lz,      "IMAGE_SAMPLE_C_LZ">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_cd,      "IMAGE_SAMPLE_C_CD">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_cd_cl,   "IMAGE_SAMPLE_C_CD_CL">;

// Sample with offsets
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_o,         "IMAGE_SAMPLE_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_cl_o,      "IMAGE_SAMPLE_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_d_o,       "IMAGE_SAMPLE_D_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_d_cl_o,    "IMAGE_SAMPLE_D_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_l_o,       "IMAGE_SAMPLE_L_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_b_o,       "IMAGE_SAMPLE_B_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_b_cl_o,    "IMAGE_SAMPLE_B_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_lz_o,      "IMAGE_SAMPLE_LZ_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_cd_o,      "IMAGE_SAMPLE_CD_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_cd_cl_o,   "IMAGE_SAMPLE_CD_CL_O">;

// Sample with comparison and offsets
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_o,       "IMAGE_SAMPLE_C_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_cl_o,    "IMAGE_SAMPLE_C_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_d_o,     "IMAGE_SAMPLE_C_D_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_d_cl_o,  "IMAGE_SAMPLE_C_D_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_l_o,     "IMAGE_SAMPLE_C_L_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_b_o,     "IMAGE_SAMPLE_C_B_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_b_cl_o,  "IMAGE_SAMPLE_C_B_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_lz_o,    "IMAGE_SAMPLE_C_LZ_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_cd_o,    "IMAGE_SAMPLE_C_CD_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_sample_c_cd_cl_o, "IMAGE_SAMPLE_C_CD_CL_O">;

// Gather opcodes
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4,           "IMAGE_GATHER4">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_cl,        "IMAGE_GATHER4_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_l,         "IMAGE_GATHER4_L">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_b,         "IMAGE_GATHER4_B">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_b_cl,      "IMAGE_GATHER4_B_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_lz,        "IMAGE_GATHER4_LZ">;

defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c,         "IMAGE_GATHER4_C">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_cl,      "IMAGE_GATHER4_C_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_l,       "IMAGE_GATHER4_C_L">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_b,       "IMAGE_GATHER4_C_B">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_b_cl,    "IMAGE_GATHER4_C_B_CL">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_lz,      "IMAGE_GATHER4_C_LZ">;

defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_o,         "IMAGE_GATHER4_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_cl_o,      "IMAGE_GATHER4_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_l_o,       "IMAGE_GATHER4_L_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_b_o,       "IMAGE_GATHER4_B_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_b_cl_o,    "IMAGE_GATHER4_B_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_lz_o,      "IMAGE_GATHER4_LZ_O">;

defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_o,       "IMAGE_GATHER4_C_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_cl_o,    "IMAGE_GATHER4_C_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_l_o,     "IMAGE_GATHER4_C_L_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_b_o,     "IMAGE_GATHER4_C_B_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_b_cl_o,  "IMAGE_GATHER4_C_B_CL_O">;
defm : AMDGCNSamplePatterns<int_amdgcn_image_gather4_c_lz_o,    "IMAGE_GATHER4_C_LZ_O">;

defm : AMDGCNSamplePatterns<int_amdgcn_image_getlod, "IMAGE_GET_LOD">;

// Image atomics
defm : ImageAtomicPatterns<int_amdgcn_image_atomic_swap, "IMAGE_ATOMIC_SWAP">;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V1, i32>;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V2, v2i32>;
def : ImageAtomicCmpSwapPattern<IMAGE_ATOMIC_CMPSWAP_V4, v4i32>;
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
