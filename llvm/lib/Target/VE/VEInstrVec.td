//===----------------------------------------------------------------------===//
// Vector Instructions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instructions
//
// Define all vector instructions defined in SX-Aurora TSUBASA Architecture
// Guide here.  As those mnemonics, we use mnemonics defined in Vector Engine
// Assembly Language Reference Manual.
//
// Some instructions can update existing data by following instructions
// sequence.
//
//   lea %s0, 256
//   lea %s1, 128
//   lvl %s0
//   vbrd %v0, 2 # v0 = { 2, 2, 2, ..., 2, 2, 2 }
//   lvl %s1
//   vbrd %v0, 3 # v0 = { 3, 3, 3, ..., 3, 2, 2, 2, ..., 2, 2, 2 }
//
// In order to represent above with a virtual register, we defines instructions
// with an additional base register and `_v` suffiex in mnemonic.
//
//   lea t0, 256
//   lea t1, 128
//   lea t0
//   vbrd tv0, 2
//   lvl t1
//   vbrd_v tv1, 2, tv0
//
// We also have some instructions uses VL register with an pseudo VL value
// with following suffixes in mnemonic.
//
//   l: have an additional I32 register to represent the VL value.
//   L: have an additional VL register to represent the VL value.
//===----------------------------------------------------------------------===//

//-----------------------------------------------------------------------------
// Section 8.9 - Vector Load/Store and Move Instructions
//-----------------------------------------------------------------------------

// Multiclass for VLD instructions
let mayLoad = 1, hasSideEffects = 0, Uses = [VL] in
multiclass VLDbm<string opcStr, bits<8>opc, RegisterClass RC, dag dag_in,
                 string disEnc = ""> {
  let DisableEncoding = disEnc in
  def "" : RVM<opc, (outs RC:$vx), dag_in,
               !strconcat(opcStr, " $vx, $sy, $sz")>;
  let Constraints = "$vx = $base", DisableEncoding = disEnc#"$base",
      isCodeGenOnly = 1 in
  def _v : RVM<opc, (outs RC:$vx), !con(dag_in, (ins RC:$base)),
               !strconcat(opcStr, " $vx, $sy, $sz")>;
}
multiclass VLDlm<string opcStr, bits<8>opc, RegisterClass RC, dag dag_in> {
  defm "" : VLDbm<opcStr, opc, RC, dag_in>;
  let isCodeGenOnly = 1, VE_VLInUse = 1 in {
    defm l : VLDbm<opcStr, opc, RC, !con(dag_in, (ins I32:$vl)), "$vl,">;
    defm L : VLDbm<opcStr, opc, RC, !con(dag_in, (ins VLS:$vl)), "$vl,">;
  }
}
let VE_VLIndex = 3 in
multiclass VLDtgm<string opcStr, bits<8>opc, RegisterClass RC> {
  defm rr : VLDlm<opcStr, opc, RC, (ins I64:$sy, I64:$sz)>;
  let cy = 0 in
  defm ir : VLDlm<opcStr, opc, RC, (ins simm7:$sy, I64:$sz)>;
  let cz = 0 in
  defm rz : VLDlm<opcStr, opc, RC, (ins I64:$sy, zero:$sz)>;
  let cy = 0, cz = 0 in
  defm iz : VLDlm<opcStr, opc, RC, (ins simm7:$sy, zero:$sz)>;
}
multiclass VLDm<string opcStr, bits<8>opc, RegisterClass RC> {
  let vc = 1 in defm "" : VLDtgm<opcStr, opc, RC>;
  let vc = 0 in defm NC : VLDtgm<opcStr#".nc", opc, RC>;
}

// Section 8.9.1 - VLD (Vector Load)
defm VLD : VLDm<"vld", 0x81, V64>;

// Section 8.9.2 - VLDU (Vector Load Upper)
defm VLDU : VLDm<"vldu", 0x82, V64>;

// Section 8.9.3 - VLDL (Vector Load Lower)
defm VLDLSX : VLDm<"vldl.sx", 0x83, V64>;
let cx = 1 in defm VLDLZX : VLDm<"vldl.zx", 0x83, V64>;

// Section 8.9.4 - VLD2D (Vector Load 2D)
defm VLD2D : VLDm<"vld2d", 0xc1, V64>;

// Section 8.9.5 - VLDU2D (Vector Load Upper 2D)
defm VLDU2D : VLDm<"vldu2d", 0xc2, V64>;

// Section 8.9.6 - VLDL2D (Vector Load Lower 2D)
defm VLDL2DSX : VLDm<"vldl2d.sx", 0xc3, V64>;
let cx = 1 in defm VLDL2DZX : VLDm<"vldl2d.zx", 0xc3, V64>;

// Multiclass for VST instructions
let mayStore = 1, hasSideEffects = 0, Uses = [VL] in
multiclass VSTbm<string opcStr, string argStr, bits<8>opc, dag dag_in> {
  def "" : RVM<opc, (outs), dag_in, !strconcat(opcStr, argStr)>;
  let DisableEncoding = "$vl", isCodeGenOnly = 1, VE_VLInUse = 1 in {
    def l : RVM<opc, (outs), !con(dag_in, (ins I32:$vl)),
                !strconcat(opcStr, argStr)>;
    def L : RVM<opc, (outs), !con(dag_in, (ins VLS:$vl)),
                !strconcat(opcStr, argStr)>;
  }
}
multiclass VSTmm<string opcStr, bits<8>opc, dag dag_in> {
  defm "" : VSTbm<opcStr, " $vx, $sy, $sz", opc, dag_in>;
  let m = ?, VE_VLWithMask = 1 in
  defm m : VSTbm<opcStr, " $vx, $sy, $sz, $m", opc, !con(dag_in, (ins VM:$m))>;
}
let VE_VLIndex = 3 in
multiclass VSTtgm<string opcStr, bits<8>opc, RegisterClass RC> {
  defm rrv : VSTmm<opcStr, opc, (ins I64:$sy, I64:$sz, RC:$vx)>;
  let cy = 0 in
  defm irv : VSTmm<opcStr, opc, (ins simm7:$sy, I64:$sz, RC:$vx)>;
  let cz = 0 in
  defm rzv : VSTmm<opcStr, opc, (ins I64:$sy, zero:$sz, RC:$vx)>;
  let cy = 0, cz = 0 in
  defm izv : VSTmm<opcStr, opc, (ins simm7:$sy, zero:$sz, RC:$vx)>;
}
multiclass VSTm<string opcStr, bits<8>opc, RegisterClass RC> {
  let vc = 1, cx = 0 in defm "" : VSTtgm<opcStr, opc, RC>;
  let vc = 0, cx = 0 in defm NC : VSTtgm<opcStr#".nc", opc, RC>;
  let vc = 1, cx = 1 in defm OT : VSTtgm<opcStr#".ot", opc, RC>;
  let vc = 0, cx = 1 in defm NCOT : VSTtgm<opcStr#".nc.ot", opc, RC>;
}

// Section 8.9.7 - VST (Vector Store)
defm VST : VSTm<"vst", 0x91, V64>;

// Section 8.9.8 - VST (Vector Store Upper)
defm VSTU : VSTm<"vstu", 0x92, V64>;

// Section 8.9.9 - VSTL (Vector Store Lower)
defm VSTL : VSTm<"vstl", 0x93, V64>;

// Section 8.9.10 - VST2D (Vector Store 2D)
defm VST2D : VSTm<"vst2d", 0xd1, V64>;

// Section 8.9.11 - VSTU2D (Vector Store Upper 2D)
defm VSTU2D : VSTm<"vstu2d", 0xd2, V64>;

// Section 8.9.12 - VSTL2D (Vector Store Lower 2D)
defm VSTL2D : VSTm<"vstl2d", 0xd3, V64>;

// Multiclass for VGT instructions
let mayLoad = 1, hasSideEffects = 0, Uses = [VL] in
multiclass VGTbm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                 dag dag_in, string disEnc = ""> {
  let DisableEncoding = disEnc in
  def "" : RVM<opc, (outs RC:$vx), dag_in,
               !strconcat(opcStr, " $vx, ", argStr)>;
  let Constraints = "$vx = $base", DisableEncoding = disEnc#"$base",
      isCodeGenOnly = 1 in
  def _v : RVM<opc, (outs RC:$vx), !con(dag_in, (ins RC:$base)),
               !strconcat(opcStr, " $vx, ", argStr)>;
}
multiclass VGTlm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                 dag dag_in> {
  defm "" : VGTbm<opcStr, argStr, opc, RC, dag_in>;
  let isCodeGenOnly = 1, VE_VLInUse = 1 in {
    defm l : VGTbm<opcStr, argStr, opc, RC, !con(dag_in, (ins I32:$vl)),
                   "$vl,">;
    defm L : VGTbm<opcStr, argStr, opc, RC, !con(dag_in, (ins VLS:$vl)),
                   "$vl,">;
  }
}
multiclass VGTmm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                 dag dag_in> {
  defm "" : VGTlm<opcStr, argStr, opc, RC, dag_in>;
  let m = ?, VE_VLWithMask = 1 in
  defm m : VGTlm<opcStr, argStr#", $m", opc, RC, !con(dag_in, (ins VM:$m))>;
}
let VE_VLIndex = 4 in
multiclass VGTlhm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                  dag dag_in> {
  defm rr : VGTmm<opcStr, argStr#", $sy, $sz", opc, RC,
                  !con(dag_in, (ins I64:$sy, I64:$sz))>;
  let cy = 0 in
  defm ir : VGTmm<opcStr, argStr#", $sy, $sz", opc, RC,
                  !con(dag_in, (ins simm7:$sy, I64:$sz))>;
  let cz = 0 in
  defm rz : VGTmm<opcStr, argStr#", $sy, $sz", opc, RC,
                  !con(dag_in, (ins I64:$sy, zero:$sz))>;
  let cy = 0, cz = 0 in
  defm iz : VGTmm<opcStr, argStr#", $sy, $sz", opc, RC,
                  !con(dag_in, (ins simm7:$sy, zero:$sz))>;
}
multiclass VGTtgm<string opcStr, bits<8>opc, RegisterClass RC> {
  let vy = ? in defm v : VGTlhm<opcStr, "$vy", opc, RC, (ins V64:$vy)>;
  let cs = 1, sw = ? in defm s : VGTlhm<opcStr, "$sw", opc, RC, (ins I64:$sw)>;
}
multiclass VGTm<string opcStr, bits<8>opc, RegisterClass RC> {
  let vc = 1 in defm "" : VGTtgm<opcStr, opc, RC>;
  let vc = 0 in defm NC : VGTtgm<opcStr#".nc", opc, RC>;
}

// Section 8.9.13 - VGT (Vector Gather)
defm VGT : VGTm<"vgt", 0xa1, V64>;

// Section 8.9.14 - VGTU (Vector Gather Upper)
defm VGTU : VGTm<"vgtu", 0xa2, V64>;

// Section 8.9.15 - VGTL (Vector Gather Lower)
defm VGTLSX : VGTm<"vgtl.sx", 0xa3, V64>;
let cx = 1 in defm VGTLZX : VGTm<"vgtl.zx", 0xa3, V64>;
def : MnemonicAlias<"vgtl", "vgtl.zx">;
def : MnemonicAlias<"vgtl.nc", "vgtl.zx.nc">;

// Multiclass for VSC instructions
let mayStore = 1, hasSideEffects = 0, Uses = [VL] in
multiclass VSCbm<string opcStr, string argStr, bits<8>opc, dag dag_in> {
  def "" : RVM<opc, (outs), dag_in, !strconcat(opcStr, argStr)>;
  let DisableEncoding = "$vl", isCodeGenOnly = 1, VE_VLInUse = 1 in {
    def l : RVM<opc, (outs), !con(dag_in, (ins I32:$vl)),
                !strconcat(opcStr, argStr)>;
    def L : RVM<opc, (outs), !con(dag_in, (ins VLS:$vl)),
                !strconcat(opcStr, argStr)>;
  }
}
multiclass VSCmm<string opcStr, string argStr, bits<8>opc, dag dag_in> {
  defm "" : VSCbm<opcStr, argStr, opc, dag_in>;
  let m = ?, VE_VLWithMask = 1 in
  defm m : VSCbm<opcStr, argStr#", $m", opc, !con(dag_in, (ins VM:$m))>;
}
let VE_VLIndex = 4 in
multiclass VSClhm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                  dag dag_in> {
  defm rrv : VSCmm<opcStr, " $vx, "#argStr#", $sy, $sz", opc,
                   !con(dag_in, (ins I64:$sy, I64:$sz, RC:$vx))>;
  let cy = 0 in
  defm irv : VSCmm<opcStr, " $vx, "#argStr#", $sy, $sz", opc,
                   !con(dag_in, (ins simm7:$sy, I64:$sz, RC:$vx))>;
  let cz = 0 in
  defm rzv : VSCmm<opcStr, " $vx, "#argStr#", $sy, $sz", opc,
                   !con(dag_in, (ins I64:$sy, zero:$sz, RC:$vx))>;
  let cy = 0, cz = 0 in
  defm izv : VSCmm<opcStr, " $vx, "#argStr#", $sy, $sz", opc,
                   !con(dag_in, (ins simm7:$sy, zero:$sz, RC:$vx))>;
}
multiclass VSCtgm<string opcStr, bits<8>opc, RegisterClass RC> {
  let vy = ? in defm v : VSClhm<opcStr, "$vy", opc, RC, (ins V64:$vy)>;
  let cs = 1, sw = ? in defm s : VSClhm<opcStr, "$sw", opc, RC, (ins I64:$sw)>;
}
multiclass VSCm<string opcStr, bits<8>opc, RegisterClass RC> {
  let vc = 1, cx = 0 in defm "" : VSCtgm<opcStr, opc, RC>;
  let vc = 0, cx = 0 in defm NC : VSCtgm<opcStr#".nc", opc, RC>;
  let vc = 1, cx = 1 in defm OT : VSCtgm<opcStr#".ot", opc, RC>;
  let vc = 0, cx = 1 in defm NCOT : VSCtgm<opcStr#".nc.ot", opc, RC>;
}

// Section 8.9.16 - VSC (Vector Scatter)
defm VSC : VSCm<"vsc", 0xb1, V64>;

// Section 8.9.17 - VSCU (Vector Scatter Upper)
defm VSCU : VSCm<"vscu", 0xb2, V64>;

// Section 8.9.18 - VSCL (Vector Scatter Lower)
defm VSCL : VSCm<"vscl", 0xb3, V64>;

// Section 8.9.19 - PFCHV (Prefetch Vector)
let Uses = [VL] in
multiclass PFCHVbm<string opcStr, string argStr, bits<8>opc, dag dag_in> {
  def "" : RVM<opc, (outs), dag_in, !strconcat(opcStr, argStr)>;
  let DisableEncoding = "$vl", isCodeGenOnly = 1, VE_VLInUse = 1 in {
    def l : RVM<opc, (outs), !con(dag_in, (ins I32:$vl)),
                !strconcat(opcStr, argStr)>;
    def L : RVM<opc, (outs), !con(dag_in, (ins VLS:$vl)),
                !strconcat(opcStr, argStr)>;
  }
}
let VE_VLIndex = 2 in
multiclass PFCHVm<string opcStr, bits<8>opc> {
  defm rr : PFCHVbm<opcStr, " $sy, $sz", opc, (ins I64:$sy, I64:$sz)>;
  let cy = 0 in
  defm ir : PFCHVbm<opcStr, " $sy, $sz", opc, (ins simm7:$sy, I64:$sz)>;
  let cz = 0 in
  defm rz : PFCHVbm<opcStr, " $sy, $sz", opc, (ins I64:$sy, zero:$sz)>;
  let cy = 0, cz = 0 in
  defm iz : PFCHVbm<opcStr, " $sy, $sz", opc, (ins simm7:$sy, zero:$sz)>;
}
let vc = 1, vx = 0 in defm PFCHV : PFCHVm<"pfchv", 0x80>;
let vc = 0, vx = 0 in defm PFCHVNC : PFCHVm<"pfchv.nc", 0x80>;

// Section 8.9.20 - LSV (Load S to V)
let sx = 0, vx = ?, hasSideEffects = 0 in
multiclass LSVbm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                 dag dag_in> {
  def "" : RR<opc, (outs RC:$vx), dag_in, !strconcat(opcStr, " ${vx}", argStr)>;
  let Constraints = "$vx = $base", DisableEncoding = "$base",
      isCodeGenOnly = 1 in
  def _v : RR<opc, (outs RC:$vx), !con(dag_in, (ins RC:$base)),
               !strconcat(opcStr, " ${vx}", argStr)>;
}
multiclass LSVm<string opcStr, bits<8>opc, RegisterClass RC> {
  defm rr : LSVbm<opcStr, "(${sy}), $sz", opc, RC, (ins I64:$sy, I64:$sz)>;
  let cy = 0 in
  defm ir : LSVbm<opcStr, "(${sy}), $sz", opc, RC, (ins uimm7:$sy, I64:$sz)>;
  let cz = 0 in
  defm rm : LSVbm<opcStr, "(${sy}), $sz", opc, RC, (ins I64:$sy, mimm:$sz)>;
  let cy = 0, cz = 0 in
  defm im : LSVbm<opcStr, "(${sy}), $sz", opc, RC, (ins uimm7:$sy, mimm:$sz)>;
}
defm LSV : LSVm<"lsv", 0x8e, V64>;

// Section 8.9.21 - LVS (Load V to S)
let cz = 0, sz = 0, vx = ?, hasSideEffects = 0 in
multiclass LVSm<string opcStr, bits<8>opc, RegisterClass RC> {
  def vr : RR<opc, (outs I64:$sx), (ins RC:$vx, I64:$sy),
              opcStr#" $sx, ${vx}(${sy})">;
  let cy = 0 in
  def vi : RR<opc, (outs I64:$sx), (ins RC:$vx, uimm7:$sy),
              opcStr#" $sx, ${vx}(${sy})">;
}
defm LVS : LVSm<"lvs", 0x9e, V64>;

// Section 8.9.22 - LVM (Load VM)
let sx = 0, vx = ?, hasSideEffects = 0 in
multiclass LVMbm<string opcStr, string argStr, bits<8>opc, RegisterClass RCM,
                 dag dag_in> {
  def "" : RR<opc, (outs RCM:$vx), dag_in,
              !strconcat(opcStr, " $vx, ", argStr)>;
  let Constraints = "$vx = $base", DisableEncoding = "$base",
      isCodeGenOnly = 1 in {
    def _m : RR<opc, (outs RCM:$vx), !con(dag_in, (ins RCM:$base)),
                !strconcat(opcStr, " $vx, ", argStr)>;
  }
}
multiclass LVMom<string opcStr, bits<8>opc, RegisterClass RCM> {
  defm rr : LVMbm<opcStr, "$sy, $sz", opc, RCM, (ins I64:$sy, I64:$sz)>;
  let cy = 0 in
  defm ir : LVMbm<opcStr, "$sy, $sz", opc, RCM, (ins uimm2:$sy, I64:$sz)>;
  let cz = 0 in
  defm rm : LVMbm<opcStr, "$sy, $sz", opc, RCM, (ins I64:$sy, mimm:$sz)>;
  let cy = 0, cz = 0 in
  defm im : LVMbm<opcStr, "$sy, $sz", opc, RCM, (ins uimm2:$sy, mimm:$sz)>;
}
multiclass LVMm<string opcStr, bits<8>opc, RegisterClass RCM> {
  defm "" : LVMom<opcStr, opc, RCM>;
}
defm LVM : LVMm<"lvm", 0xb7, VM>;

// Section 8.9.23 - SVM (Save VM)
let cz = 0, sz = 0, vz = ?, hasSideEffects = 0 in
multiclass SVMm<string opcStr, bits<8>opc, RegisterClass RCM> {
  def mr : RR<opc, (outs I64:$sx), (ins RCM:$vz, I64:$sy),
              opcStr#" $sx, $vz, $sy">;
  let cy = 0 in
  def mi : RR<opc, (outs I64:$sx), (ins RCM:$vz, uimm2:$sy),
              opcStr#" $sx, $vz, $sy">;
}
defm SVM : SVMm<"svm", 0xa7, VM>;

// Section 8.9.24 - VBRD (Vector Broadcast)
let vx = ?, hasSideEffects = 0, Uses = [VL] in
multiclass VBRDbm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                  dag dag_in, string disEnc = ""> {
  let DisableEncoding = disEnc in
  def "" : RV<opc, (outs RC:$vx), dag_in,
              !strconcat(opcStr, " $vx, ", argStr)>;
  let Constraints = "$vx = $base", DisableEncoding = disEnc#"$base",
      isCodeGenOnly = 1 in
  def _v : RV<opc, (outs RC:$vx), !con(dag_in, (ins RC:$base)),
              !strconcat(opcStr, " $vx, ", argStr)>;
}
multiclass VBRDlm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                  dag dag_in> {
  defm "" : VBRDbm<opcStr, argStr, opc, RC, dag_in>;
  let isCodeGenOnly = 1, VE_VLInUse = 1 in {
    defm l : VBRDbm<opcStr, argStr, opc, RC, !con(dag_in, (ins I32:$vl)),
                   "$vl,">;
    defm L : VBRDbm<opcStr, argStr, opc, RC, !con(dag_in, (ins VLS:$vl)),
                   "$vl,">;
  }
}
multiclass VBRDmm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                  RegisterClass RCM, dag dag_in> {
  defm "" : VBRDlm<opcStr, argStr, opc, RC, dag_in>;
  let m = ?, VE_VLWithMask = 1 in
  defm m : VBRDlm<opcStr, argStr#", $m", opc, RC, !con(dag_in, (ins RCM:$m))>;
}
let VE_VLIndex = 2 in
multiclass VBRDm<string opcStr, bits<8>opc, RegisterClass VRC, RegisterClass RC,
                 RegisterClass RCM> {
  defm r : VBRDmm<opcStr, "$sy", opc, VRC, RCM, (ins RC:$sy)>;
  let cy = 0 in
  defm i : VBRDmm<opcStr, "$sy", opc, VRC, RCM, (ins simm7:$sy)>;
}
let cx = 0, cx2 = 0 in
defm VBRD : VBRDm<"vbrd", 0x8c, V64, I64, VM>;
let cx = 0, cx2 = 1 in
defm VBRDL : VBRDm<"vbrdl", 0x8c, V64, I32, VM>;
let cx = 1, cx2 = 0 in
defm VBRDU : VBRDm<"vbrdu", 0x8c, V64, F32, VM>;
let cx = 1, cx2 = 1 in
defm PVBRD : VBRDm<"pvbrd", 0x8c, V64, I64, VM512>;

// Section 8.9.25 - VMV (Vector Move)
let vx = ?, vz = ?, hasSideEffects = 0, Uses = [VL] in
multiclass VMVbm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                 dag dag_in, string disEnc = ""> {
  let DisableEncoding = disEnc in
  def "" : RV<opc, (outs RC:$vx), dag_in,
              !strconcat(opcStr, " $vx, ", argStr)>;
  let Constraints = "$vx = $base", DisableEncoding = disEnc#"$base",
      isCodeGenOnly = 1 in
  def _v : RV<opc, (outs RC:$vx), !con(dag_in, (ins RC:$base)),
              !strconcat(opcStr, " $vx, ", argStr)>;
}
multiclass VMVlm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                 dag dag_in> {
  defm "" : VMVbm<opcStr, argStr, opc, RC, dag_in>;
  let isCodeGenOnly = 1, VE_VLInUse = 1 in {
    defm l : VMVbm<opcStr, argStr, opc, RC, !con(dag_in, (ins I32:$vl)),
                   "$vl,">;
    defm L : VMVbm<opcStr, argStr, opc, RC, !con(dag_in, (ins VLS:$vl)),
                   "$vl,">;
  }
}
multiclass VMVmm<string opcStr, bits<8>opc, RegisterClass RC,
                 RegisterClass RCM, dag dag_in> {
  defm "" : VMVlm<opcStr, "$sy, $vz", opc, RC, dag_in>;
  let m = ?, VE_VLWithMask = 1 in
  defm m : VMVlm<opcStr, "$sy, $vz, $m", opc, RC, !con(dag_in, (ins RCM:$m))>;
}
let VE_VLIndex = 3 in
multiclass VMVm<string opcStr, bits<8>opc, RegisterClass RC,
                RegisterClass RCM> {
  defm rv : VMVmm<opcStr, opc, RC, RCM, (ins I64:$sy, RC:$vz)>;
  let cy = 0 in
  defm iv : VMVmm<opcStr, opc, RC, RCM, (ins uimm7:$sy, RC:$vz)>;
}
defm VMV : VMVm<"vmv", 0x9c, V64, VM>;

//-----------------------------------------------------------------------------
// Section 8.10 - Vector Fixed-Point Arithmetic Instructions
//-----------------------------------------------------------------------------

// Multiclass for generic vector calculation
let vx = ?, hasSideEffects = 0, Uses = [VL] in
multiclass RVbm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                dag dag_in, string disEnc = ""> {
  let DisableEncoding = disEnc in
  def "" : RV<opc, (outs RC:$vx), dag_in,
              !strconcat(opcStr, " $vx", argStr)>;
  let Constraints = "$vx = $base", DisableEncoding = disEnc#"$base",
      isCodeGenOnly = 1 in
  def _v : RV<opc, (outs RC:$vx), !con(dag_in, (ins RC:$base)),
              !strconcat(opcStr, " $vx", argStr)>;
}
multiclass RVlm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                dag dag_in> {
  defm "" : RVbm<opcStr, argStr, opc, RC, dag_in>;
  let isCodeGenOnly = 1, VE_VLInUse = 1 in {
    defm l : RVbm<opcStr, argStr, opc, RC, !con(dag_in, (ins I32:$vl)),
                  "$vl,">;
    defm L : RVbm<opcStr, argStr, opc, RC, !con(dag_in, (ins VLS:$vl)),
                  "$vl,">;
  }
}
multiclass RVmm<string opcStr, string argStr, bits<8>opc, RegisterClass RC,
                RegisterClass RCM, dag dag_in> {
  defm "" : RVlm<opcStr, argStr, opc, RC, dag_in>;
  let m = ?, VE_VLWithMask = 1 in
  defm m : RVlm<opcStr, argStr#", $m", opc, RC, !con(dag_in, (ins RCM:$m))>;
}
// Generic RV multiclass with 2 arguments.
//   e.g. VADD, VSUB, VMPY, and etc.
let VE_VLIndex = 3 in
multiclass RVm<string opcStr, bits<8>opc, RegisterClass VRC, RegisterClass RC,
               RegisterClass RCM, Operand SIMM = simm7> {
  let cy = 0, sy = 0, vy = ?, vz = ? in
  defm vv : RVmm<opcStr, ", $vy, $vz", opc, VRC, RCM, (ins VRC:$vy, VRC:$vz)>;
  let cs = 1, vz = ? in
  defm rv : RVmm<opcStr, ", $sy, $vz", opc, VRC, RCM, (ins RC:$sy, VRC:$vz)>;
  let cs = 1, cy = 0, vz = ? in
  defm iv : RVmm<opcStr, ", $sy, $vz", opc, VRC, RCM, (ins SIMM:$sy, VRC:$vz)>;
}
// Special RV multiclass with 2 arguments using cs2.
//   e.g. VDIV, VDVS, and VDVX.
let VE_VLIndex = 3 in
multiclass RVDIVm<string opcStr, bits<8>opc, RegisterClass VRC,
                  RegisterClass RC, RegisterClass RCM, Operand SIMM = simm7> {
  let cy = 0, sy = 0, vy = ?, vz = ? in
  defm vv : RVmm<opcStr, ", $vy, $vz", opc, VRC, RCM, (ins VRC:$vy, VRC:$vz)>;
  let cs2 = 1, vy = ? in
  defm vr : RVmm<opcStr, ", $vy, $sy", opc, VRC, RCM, (ins VRC:$vy, RC:$sy)>;
  let cs2 = 1, cy = 0, vy = ? in
  defm vi : RVmm<opcStr, ", $vy, $sy", opc, VRC, RCM, (ins VRC:$vy, SIMM:$sy)>;
  let cs = 1, vz = ? in
  defm rv : RVmm<opcStr, ", $sy, $vz", opc, VRC, RCM, (ins RC:$sy, VRC:$vz)>;
  let cs = 1, cy = 0, vz = ? in
  defm iv : RVmm<opcStr, ", $sy, $vz", opc, VRC, RCM, (ins SIMM:$sy, VRC:$vz)>;
}

// Section 8.10.1 - VADD (Vector Add)
let cx = 0, cx2 = 0 in
defm VADDUL : RVm<"vaddu.l", 0xc8, V64, I64, VM>;
let cx = 0, cx2 = 1 in {
  defm PVADDULO : RVm<"pvaddu.lo", 0xc8, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VADDUW : RVm<"vaddu.w", 0xc8, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVADDUUP : RVm<"pvaddu.up", 0xc8, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVADDU : RVm<"pvaddu", 0xc8, V64, I64, VM512>;
def : MnemonicAlias<"vaddu.w", "pvaddu.lo">;

// Section 8.10.2 - VADS (Vector Add Single)
let cx = 0, cx2 = 0 in
defm VADDSWSX : RVm<"vadds.w.sx", 0xca, V64, I32, VM>;
let cx = 0, cx2 = 1 in {
  defm PVADDSLO : RVm<"pvadds.lo", 0xca, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VADDSWZX : RVm<"vadds.w.zx", 0xca, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVADDSUP : RVm<"pvadds.up", 0xca, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVADDS : RVm<"pvadds", 0xca, V64, I64, VM512>;
def : MnemonicAlias<"pvadds.lo.sx", "vadds.w.sx">;
def : MnemonicAlias<"vadds.w.zx", "pvadds.lo">;
def : MnemonicAlias<"vadds.w", "pvadds.lo">;
def : MnemonicAlias<"pvadds.lo.zx", "pvadds.lo">;

// Section 8.10.3 - VADX (Vector Add)
defm VADDSL : RVm<"vadds.l", 0x8b, V64, I64, VM>;

// Section 8.10.4 - VSUB (Vector Subtract)
let cx = 0, cx2 = 0 in
defm VSUBUL : RVm<"vsubu.l", 0xd8, V64, I64, VM>;
let cx = 0, cx2 = 1 in {
  defm PVSUBULO : RVm<"pvsubu.lo", 0xd8, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VSUBUW : RVm<"vsubu.w", 0xd8, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVSUBUUP : RVm<"pvsubu.up", 0xd8, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVSUBU : RVm<"pvsubu", 0xd8, V64, I64, VM512>;
def : MnemonicAlias<"vsubu.w", "pvsubu.lo">;

// Section 8.10.5 - VSBS (Vector Subtract Single)
let cx = 0, cx2 = 0 in
defm VSUBSWSX : RVm<"vsubs.w.sx", 0xda, V64, I32, VM>;
let cx = 0, cx2 = 1 in {
  defm PVSUBSLO : RVm<"pvsubs.lo", 0xda, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VSUBSWZX : RVm<"vsubs.w.zx", 0xda, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVSUBSUP : RVm<"pvsubs.up", 0xda, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVSUBS : RVm<"pvsubs", 0xda, V64, I64, VM512>;
def : MnemonicAlias<"pvsubs.lo.sx", "vsubs.w.sx">;
def : MnemonicAlias<"vsubs.w.zx", "pvsubs.lo">;
def : MnemonicAlias<"vsubs.w", "pvsubs.lo">;
def : MnemonicAlias<"pvsubs.lo.zx", "pvsubs.lo">;

// Section 8.10.6 - VSBX (Vector Subtract)
defm VSUBSL : RVm<"vsubs.l", 0x9b, V64, I64, VM>;

// Section 8.10.7 - VMPY (Vector Multiply)
let cx2 = 0 in
defm VMULUL : RVm<"vmulu.l", 0xc9, V64, I64, VM>;
let cx2 = 1 in
defm VMULUW : RVm<"vmulu.w", 0xc9, V64, I32, VM>;

// Section 8.10.8 - VMPS (Vector Multiply Single)
let cx2 = 0 in
defm VMULSWSX : RVm<"vmuls.w.sx", 0xcb, V64, I32, VM>;
let cx2 = 1 in
defm VMULSWZX : RVm<"vmuls.w.zx", 0xcb, V64, I32, VM>;
def : MnemonicAlias<"vmuls.w", "vmuls.w.zx">;

// Section 8.10.9 - VMPX (Vector Multiply)
defm VMULSL : RVm<"vmuls.l", 0xdb, V64, I64, VM>;

// Section 8.10.10 - VMPD (Vector Multiply)
defm VMULSLW : RVm<"vmuls.l.w", 0xd9, V64, I32, VM>;

// Section 8.10.11 - VDIV (Vector Divide)
let cx2 = 0 in
defm VDIVUL : RVDIVm<"vdivu.l", 0xe9, V64, I64, VM>;
let cx2 = 1 in
defm VDIVUW : RVDIVm<"vdivu.w", 0xe9, V64, I32, VM>;

// Section 8.10.12 - VDVS (Vector Divide Single)
let cx2 = 0 in
defm VDIVSWSX : RVDIVm<"vdivs.w.sx", 0xeb, V64, I32, VM>;
let cx2 = 1 in
defm VDIVSWZX : RVDIVm<"vdivs.w.zx", 0xeb, V64, I32, VM>;
def : MnemonicAlias<"vdivs.w", "vdivs.w.zx">;

// Section 8.10.13 - VDVX (Vector Divide)
defm VDIVSL : RVDIVm<"vdivs.l", 0xfb, V64, I64, VM>;

// Section 8.10.14 - VCMP (Vector Compare)
let cx = 0, cx2 = 0 in
defm VCMPUL : RVm<"vcmpu.l", 0xb9, V64, I64, VM>;
let cx = 0, cx2 = 1 in {
  defm PVCMPULO : RVm<"pvcmpu.lo", 0xb9, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VCMPUW : RVm<"vcmpu.w", 0xb9, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVCMPUUP : RVm<"pvcmpu.up", 0xb9, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVCMPU : RVm<"pvcmpu", 0xb9, V64, I64, VM512>;
def : MnemonicAlias<"vcmpu.w", "pvcmpu.lo">;

// Section 8.10.15 - VCPS (Vector Compare Single)
let cx = 0, cx2 = 0 in
defm VCMPSWSX : RVm<"vcmps.w.sx", 0xfa, V64, I32, VM>;
let cx = 0, cx2 = 1 in {
  defm PVCMPSLO : RVm<"pvcmps.lo", 0xfa, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VCMPSWZX : RVm<"vcmps.w.zx", 0xfa, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVCMPSUP : RVm<"pvcmps.up", 0xfa, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVCMPS : RVm<"pvcmps", 0xfa, V64, I64, VM512>;
def : MnemonicAlias<"pvcmps.lo.sx", "vcmps.w.sx">;
def : MnemonicAlias<"vcmps.w.zx", "pvcmps.lo">;
def : MnemonicAlias<"vcmps.w", "pvcmps.lo">;
def : MnemonicAlias<"pvcmps.lo.zx", "pvcmps.lo">;

// Section 8.10.16 - VCPX (Vector Compare)
defm VCMPSL : RVm<"vcmps.l", 0xba, V64, I64, VM>;

// Section 8.10.17 - VCMS (Vector Compare and Select Maximum/Minimum Single)
let cx = 0, cx2 = 0 in
defm VMAXSWSX : RVm<"vmaxs.w.sx", 0x8a, V64, I32, VM>;
let cx = 0, cx2 = 1 in {
  defm PVMAXSLO : RVm<"pvmaxs.lo", 0x8a, V64, I32, VM>;
  let isCodeGenOnly = 1 in
  defm VMAXSWZX : RVm<"vmaxs.w.zx", 0x8a, V64, I32, VM>;
}
let cx = 1, cx2 = 0 in
defm PVMAXSUP : RVm<"pvmaxs.up", 0x8a, V64, I64, VM>;
let cx = 1, cx2 = 1 in
defm PVMAXS : RVm<"pvmaxs", 0x8a, V64, I64, VM512>;
let cs2 = 1 in {
  let cx = 0, cx2 = 0 in
  defm VMINSWSX : RVm<"vmins.w.sx", 0x8a, V64, I32, VM>;
  let cx = 0, cx2 = 1 in {
    defm PVMINSLO : RVm<"pvmins.lo", 0x8a, V64, I32, VM>;
    let isCodeGenOnly = 1 in
    defm VMINSWZX : RVm<"vmins.w.zx", 0x8a, V64, I32, VM>;
  }
  let cx = 1, cx2 = 0 in
  defm PVMINSUP : RVm<"pvmins.up", 0x8a, V64, I64, VM>;
  let cx = 1, cx2 = 1 in
  defm PVMINS : RVm<"pvmins", 0x8a, V64, I64, VM512>;
}
def : MnemonicAlias<"pvmaxs.lo.sx", "vmaxs.w.sx">;
def : MnemonicAlias<"vmaxs.w.zx", "pvmaxs.lo">;
def : MnemonicAlias<"vmaxs.w", "pvmaxs.lo">;
def : MnemonicAlias<"pvmaxs.lo.zx", "pvmaxs.lo">;
def : MnemonicAlias<"pvmins.lo.sx", "vmins.w.sx">;
def : MnemonicAlias<"vmins.w.zx", "pvmins.lo">;
def : MnemonicAlias<"vmins.w", "pvmins.lo">;
def : MnemonicAlias<"pvmins.lo.zx", "pvmins.lo">;

// Section 8.10.18 - VCMX (Vector Compare and Select Maximum/Minimum)
defm VMAXSL : RVm<"vmaxs.l", 0x9a, V64, I64, VM>;
let cs2 = 1 in
defm VMINSL : RVm<"vmins.l", 0x9a, V64, I64, VM>;
