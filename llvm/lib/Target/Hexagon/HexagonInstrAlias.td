//==- HexagonInstrAlias.td - Hexagon Instruction Aliases ---*- tablegen -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//                     Hexagon Instruction Mappings
//===----------------------------------------------------------------------===//


def : InstAlias<"memb({GP}+#$addr) = $Nt.new",
                (S2_storerbnewgp u16_0Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memh({GP}+#$addr) = $Nt.new",
                (S2_storerhnewgp u16_1Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memw({GP}+#$addr) = $Nt.new",
                (S2_storerinewgp u16_2Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memb({GP}+#$addr) = $Nt",
                (S2_storerbgp u16_0Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memh({GP}+#$addr) = $Nt",
                (S2_storerhgp u16_1Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memh({GP}+#$addr) = $Nt.h",
                (S2_storerfgp u16_1Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memw({GP}+#$addr) = $Nt",
                (S2_storerigp u16_2Imm:$addr, IntRegs:$Nt)>;
def : InstAlias<"memd({GP}+#$addr) = $Nt",
                (S2_storerdgp u16_3Imm:$addr, DoubleRegs:$Nt)>;

def : InstAlias<"$Nt = memb({GP}+#$addr)",
                (L2_loadrbgp IntRegs:$Nt, u16_0Imm:$addr)>;
def : InstAlias<"$Nt = memub({GP}+#$addr)",
                (L2_loadrubgp IntRegs:$Nt, u16_0Imm:$addr)>;
def : InstAlias<"$Nt = memh({GP}+#$addr)",
                (L2_loadrhgp IntRegs:$Nt, u16_1Imm:$addr)>;
def : InstAlias<"$Nt = memuh({GP}+#$addr)",
                (L2_loadruhgp IntRegs:$Nt, u16_1Imm:$addr)>;
def : InstAlias<"$Nt = memw({GP}+#$addr)",
                (L2_loadrigp IntRegs:$Nt, u16_2Imm:$addr)>;
def : InstAlias<"$Nt = memd({GP}+#$addr)",
                (L2_loadrdgp DoubleRegs:$Nt, u16_3Imm:$addr)>;

// Alias of: memXX($Rs+#XX) = $Rt to memXX($Rs) = $Rt
def : InstAlias<"memb($Rs) = $Rt",
      (S2_storerb_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memh($Rs) = $Rt",
      (S2_storerh_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memh($Rs) = $Rt.h",
      (S2_storerf_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memw($Rs) = $Rt",
      (S2_storeri_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memb($Rs) = $Rt.new",
      (S2_storerbnew_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memh($Rs) = $Rt.new",
      (S2_storerhnew_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memw($Rs) = $Rt.new",
      (S2_storerinew_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"memb($Rs) = #$S8",
      (S4_storeirb_io IntRegs:$Rs, 0, s8Ext:$S8), 0>;

def : InstAlias<"memh($Rs) = #$S8",
      (S4_storeirh_io IntRegs:$Rs, 0, s8Ext:$S8), 0>;

def : InstAlias<"memw($Rs) = #$S8",
      (S4_storeiri_io IntRegs:$Rs, 0, s8Ext:$S8), 0>;

def : InstAlias<"memd($Rs) = $Rtt",
      (S2_storerd_io IntRegs:$Rs, 0, DoubleRegs:$Rtt), 0>;

def : InstAlias<"memb($Rs) = setbit(#$U5)",
      (L4_ior_memopb_io IntRegs:$Rs, 0, u5Imm:$U5), 0>;

def : InstAlias<"memh($Rs) = setbit(#$U5)",
      (L4_ior_memoph_io IntRegs:$Rs, 0, u5Imm:$U5), 0>;

def : InstAlias<"memw($Rs) = setbit(#$U5)",
      (L4_ior_memopw_io IntRegs:$Rs, 0, u5Imm:$U5), 0>;

def : InstAlias<"memb($Rs) = clrbit(#$U5)",
      (L4_iand_memopb_io IntRegs:$Rs, 0, u5Imm:$U5), 0>;

def : InstAlias<"memh($Rs) = clrbit(#$U5)",
      (L4_iand_memoph_io IntRegs:$Rs, 0, u5Imm:$U5), 0>;

def : InstAlias<"memw($Rs) = clrbit(#$U5)",
      (L4_iand_memopw_io IntRegs:$Rs, 0, u5Imm:$U5), 0>;

// Alias of: $Rd = memXX($Rs+#XX) to $Rd = memXX($Rs)
def : InstAlias<"$Rd = memb($Rs)",
      (L2_loadrb_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rd = memub($Rs)",
      (L2_loadrub_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rd = memh($Rs)",
      (L2_loadrh_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rd = memuh($Rs)",
      (L2_loadruh_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rd = memw($Rs)",
      (L2_loadri_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rdd = memd($Rs)",
      (L2_loadrd_io DoubleRegs:$Rdd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rd = memubh($Rs)",
      (L2_loadbzw2_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rdd = memubh($Rs)",
      (L2_loadbzw4_io DoubleRegs:$Rdd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rd = membh($Rs)",
      (L2_loadbsw2_io IntRegs:$Rd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rdd = membh($Rs)",
      (L2_loadbsw4_io DoubleRegs:$Rdd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rdd = memb_fifo($Rs)",
      (L2_loadalignb_io DoubleRegs:$Rdd, IntRegs:$Rs, 0), 0>;

def : InstAlias<"$Rdd = memh_fifo($Rs)",
      (L2_loadalignh_io DoubleRegs:$Rdd, IntRegs:$Rs, 0), 0>;

// Alias of: if ($Pt) $Rd = memXX($Rs + #$u6_X)
//       to: if ($Pt) $Rd = memXX($Rs)
def : InstAlias<"if ($Pt) $Rd = memb($Rs)",
      (L2_ploadrbt_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt) $Rd = memub($Rs)",
      (L2_ploadrubt_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt) $Rd = memh($Rs)",
      (L2_ploadrht_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt) $Rd = memuh($Rs)",
      (L2_ploadruht_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt) $Rd = memw($Rs)",
      (L2_ploadrit_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt) $Rdd = memd($Rs)",
      (L2_ploadrdt_io DoubleRegs:$Rdd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

// Alias of: if ($Pt) memXX($Rs + #$u6_X) = $Rt
//       to: if ($Pt) memXX($Rs) = $Rt
def : InstAlias<"if ($Pt) memb($Rs) = $Rt",
      (S2_pstorerbt_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memh($Rs) = $Rt",
      (S2_pstorerht_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memh($Rs) = $Rt.h",
      (S2_pstorerft_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memw($Rs) = $Rt",
      (S2_pstorerit_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memd($Rs) = $Rtt",
      (S2_pstorerdt_io PredRegs:$Pt, IntRegs:$Rs, 0, DoubleRegs:$Rtt), 0>;

def : InstAlias<"if ($Pt) memb($Rs) = $Rt.new",
      (S2_pstorerbnewt_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memh($Rs) = $Rt.new",
      (S2_pstorerhnewt_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memw($Rs) = $Rt.new",
      (S2_pstorerinewt_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt.new) memb($Rs) = $Rt.new",
      (S4_pstorerbnewtnew_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt.new) memh($Rs) = $Rt.new",
      (S4_pstorerhnewtnew_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt.new) memw($Rs) = $Rt.new",
      (S4_pstorerinewtnew_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;


// Alias of: if (!$Pt) $Rd = memXX($Rs + #$u6_X)
//       to: if (!$Pt) $Rd = memXX($Rs)
def : InstAlias<"if (!$Pt) $Rd = memb($Rs)",
      (L2_ploadrbf_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt) $Rd = memub($Rs)",
      (L2_ploadrubf_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt) $Rd = memh($Rs)",
      (L2_ploadrhf_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt) $Rd = memuh($Rs)",
      (L2_ploadruhf_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt) $Rd = memw($Rs)",
      (L2_ploadrif_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt) $Rdd = memd($Rs)",
      (L2_ploadrdf_io DoubleRegs:$Rdd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

// Alias of: if (!$Pt) memXX($Rs + #$u6_X) = $Rt
//       to: if (!$Pt) memXX($Rs) = $Rt
def : InstAlias<"if (!$Pt) memb($Rs) = $Rt",
      (S2_pstorerbf_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt) memh($Rs) = $Rt",
      (S2_pstorerhf_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt) memh($Rs) = $Rt.h",
      (S2_pstorerff_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt) memw($Rs) = $Rt",
      (S2_pstorerif_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt) memd($Rs) = $Rtt",
      (S2_pstorerdf_io PredRegs:$Pt, IntRegs:$Rs, 0, DoubleRegs:$Rtt), 0>;

def : InstAlias<"if (!$Pt) memb($Rs) = $Rt.new",
      (S2_pstorerbnewf_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt) memh($Rs) = $Rt.new",
      (S2_pstorerhnewf_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt) memw($Rs) = $Rt.new",
      (S2_pstorerinewf_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt.new) memb($Rs) = $Rt.new",
      (S4_pstorerbnewfnew_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt.new) memh($Rs) = $Rt.new",
      (S4_pstorerhnewfnew_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pt.new) memw($Rs) = $Rt.new",
      (S4_pstorerinewfnew_io PredRegs:$Pt, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pt) memb($Rs) = #$S6",
      (S4_storeirbt_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if ($Pt) memh($Rs) = #$S6",
      (S4_storeirht_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if ($Pt) memw($Rs) = #$S6",
      (S4_storeirit_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if ($Pt.new) memb($Rs) = #$S6",
      (S4_storeirbtnew_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if ($Pt.new) memh($Rs) = #$S6",
      (S4_storeirhtnew_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if ($Pt.new) memw($Rs) = #$S6",
      (S4_storeiritnew_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if (!$Pt) memb($Rs) = #$S6",
      (S4_storeirbf_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if (!$Pt) memh($Rs) = #$S6",
      (S4_storeirhf_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if (!$Pt) memw($Rs) = #$S6",
      (S4_storeirif_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if (!$Pt.new) memb($Rs) = #$S6",
      (S4_storeirbfnew_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if (!$Pt.new) memh($Rs) = #$S6",
      (S4_storeirhfnew_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

def : InstAlias<"if (!$Pt.new) memw($Rs) = #$S6",
      (S4_storeirifnew_io PredRegs:$Pt, IntRegs:$Rs, 0, s6Ext:$S6), 0>;

// Alias of: memXX($Rs + $u6_X) |= $Rt, also &=, +=, -=
//       to: memXX($Rs) |= $Rt
def : InstAlias<"memb($Rs) &= $Rt",
      (L4_and_memopb_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memb($Rs) |= $Rt",
      (L4_or_memopb_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memb($Rs) += $Rt",
      (L4_add_memopb_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memb($Rs) -= $Rt",
      (L4_sub_memopb_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memb($Rs) += #$U5",
      (L4_iadd_memopb_io IntRegs:$Rs, 0, u5Imm:$U5), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memb($Rs) -= #$U5",
      (L4_isub_memopb_io IntRegs:$Rs, 0, u5Imm:$U5), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memh($Rs) &= $Rt",
      (L4_and_memoph_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memh($Rs) |= $Rt",
      (L4_or_memoph_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memh($Rs) += $Rt",
      (L4_add_memoph_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memh($Rs) -= $Rt",
      (L4_sub_memoph_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memh($Rs) += #$U5",
      (L4_iadd_memoph_io IntRegs:$Rs, 0, u5Imm:$U5), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memh($Rs) -= #$U5",
      (L4_isub_memoph_io IntRegs:$Rs, 0, u5Imm:$U5), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memw($Rs) &= $Rt",
      (L4_and_memopw_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memw($Rs) |= $Rt",
      (L4_or_memopw_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memw($Rs) += $Rt",
      (L4_add_memopw_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memw($Rs) -= $Rt",
      (L4_sub_memopw_io IntRegs:$Rs, 0, IntRegs:$Rt), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memw($Rs) += #$U5",
      (L4_iadd_memopw_io IntRegs:$Rs, 0, u5Imm:$U5), 0>,
      Requires<[UseMEMOP]>;

def : InstAlias<"memw($Rs) -= #$U5",
      (L4_isub_memopw_io IntRegs:$Rs, 0, u5Imm:$U5), 0>,
      Requires<[UseMEMOP]>;

//
// Alias of: if ($Pv.new) memX($Rs) = $Rt
//       to: if (p3.new) memX(r17 + #0) = $Rt
def : InstAlias<"if ($Pv.new) memb($Rs) = $Rt",
      (S4_pstorerbtnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pv.new) memh($Rs) = $Rt",
      (S4_pstorerhtnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pv.new) memh($Rs) = $Rt.h",
      (S4_pstorerftnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pv.new) memw($Rs) = $Rt",
      (S4_pstoreritnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if ($Pv.new) memd($Rs) = $Rtt",
      (S4_pstorerdtnew_io
       PredRegs:$Pv, IntRegs:$Rs, 0, DoubleRegs:$Rtt), 0>;

def : InstAlias<"if (!$Pv.new) memb($Rs) = $Rt",
      (S4_pstorerbfnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pv.new) memh($Rs) = $Rt",
      (S4_pstorerhfnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pv.new) memh($Rs) = $Rt.h",
      (S4_pstorerffnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pv.new) memw($Rs) = $Rt",
      (S4_pstorerifnew_io PredRegs:$Pv, IntRegs:$Rs, 0, IntRegs:$Rt), 0>;

def : InstAlias<"if (!$Pv.new) memd($Rs) = $Rtt",
      (S4_pstorerdfnew_io
       PredRegs:$Pv, IntRegs:$Rs, 0, DoubleRegs:$Rtt), 0>;

//
// Alias of: if ($Pt.new) $Rd = memub($Rs) -- And if (!$Pt.new) ...
//       to: if ($Pt.new) $Rd = memub($Rs + #$u6_0)
def : InstAlias<"if ($Pt.new) $Rd = memub($Rs)",
      (L2_ploadrubtnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt.new) $Rd = memb($Rs)",
      (L2_ploadrbtnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt.new) $Rd = memh($Rs)",
      (L2_ploadrhtnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt.new) $Rd = memuh($Rs)",
      (L2_ploadruhtnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt.new) $Rd = memw($Rs)",
      (L2_ploadritnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if ($Pt.new) $Rdd = memd($Rs)",
      (L2_ploadrdtnew_io DoubleRegs:$Rdd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt.new) $Rd = memub($Rs)",
      (L2_ploadrubfnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt.new) $Rd = memb($Rs)",
      (L2_ploadrbfnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt.new) $Rd = memh($Rs)",
      (L2_ploadrhfnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt.new) $Rd = memuh($Rs)",
      (L2_ploadruhfnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt.new) $Rd = memw($Rs)",
      (L2_ploadrifnew_io IntRegs:$Rd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"if (!$Pt.new) $Rdd = memd($Rs)",
      (L2_ploadrdfnew_io DoubleRegs:$Rdd, PredRegs:$Pt, IntRegs:$Rs, 0), 0>;

def : InstAlias<"dcfetch($Rs)",
      (Y2_dcfetchbo IntRegs:$Rs, 0), 0>;

// Alias of some insn mappings, others must be handled by the parser
def : InstAlias<"$Pd=cmp.lt($Rs, $Rt)",
      (C2_cmpgt PredRegs:$Pd, IntRegs:$Rt, IntRegs:$Rs), 0>;
def : InstAlias<"$Pd=cmp.ltu($Rs, $Rt)",
      (C2_cmpgtu PredRegs:$Pd, IntRegs:$Rt, IntRegs:$Rs), 0>;

// Rd=neg(Rs) is aliased to Rd=sub(#0,Rs)
def : InstAlias<"$Rd = neg($Rs)",
      (A2_subri IntRegs:$Rd, 0, IntRegs:$Rs), 0>;

def : InstAlias<"m0 = $Rs", (A2_tfrrcr C6, IntRegs:$Rs)>;
def : InstAlias<"$Rd = m0", (A2_tfrcrr IntRegs:$Rd, C6)>;
def : InstAlias<"m1 = $Rs", (A2_tfrrcr C7, IntRegs:$Rs)>;
def : InstAlias<"$Rd = m1", (A2_tfrcrr IntRegs:$Rd, C7)>;

def : InstAlias<"$Pd = $Ps",
      (C2_or PredRegs:$Pd, PredRegs:$Ps, PredRegs:$Ps), 0>;

def : InstAlias<"$Rdd = vaddb($Rss, $Rtt)",
      (A2_vaddub DoubleRegs:$Rdd, DoubleRegs:$Rss, DoubleRegs:$Rtt), 1>;

def : InstAlias<"$Rdd = vsubb($Rss,$Rtt)",
      (A2_vsubub DoubleRegs:$Rdd, DoubleRegs:$Rss, DoubleRegs:$Rtt), 0>;

def : InstAlias<"$Rd = mpyui($Rs,$Rt)",
      (M2_mpyi IntRegs:$Rd, IntRegs:$Rs, IntRegs:$Rt), 0>;

// Assembler mapped insns: cmp.lt(a,b) -> cmp.gt(b,a)
def : InstAlias<"$Pd=cmp.lt($Rs, $Rt)",
      (C2_cmpgt PredRegs:$Pd, IntRegs:$Rt, IntRegs:$Rs), 0>;
def : InstAlias<"$Pd=cmp.ltu($Rs, $Rt)",
      (C2_cmpgtu PredRegs:$Pd, IntRegs:$Rt, IntRegs:$Rs), 0>;

// maps if (!Pu) jumpr Rs -> if (!Pu) jumpr:nt Rs
def : InstAlias<"if (!$Pu) jumpr $Rs",
      (J2_jumprf PredRegs:$Pu, IntRegs:$Rs)>,
      Requires<[HasV60T]>;

// maps if (Pu) jumpr Rs -> if (Pu) jumpr:nt Rs
def : InstAlias<"if ($Pu) jumpr $Rs",
      (J2_jumprt PredRegs:$Pu, IntRegs:$Rs)>,
      Requires<[HasV60T]>;

// maps if (!Pu) jump $r15_2 -> if (!Pu) jump:nt $r15_2
def : InstAlias<"if (!$Pu) jump $r15_2",
      (J2_jumpf PredRegs:$Pu, brtarget:$r15_2)>,
      Requires<[HasV60T]>;

// maps if (Pu) jump $r15_2 -> if (Pu) jump:nt $r15_2
def : InstAlias<"if ($Pu) jump $r15_2",
     (J2_jumpt PredRegs:$Pu, brtarget:$r15_2)>,
     Requires<[HasV60T]>;

def : InstAlias<"if ($src) jump $r15_2",
      (J2_jumpt PredRegs:$src, brtarget:$r15_2), 0>;

def : InstAlias<"if (!$src) jump $r15_2",
      (J2_jumpf PredRegs:$src, brtarget:$r15_2), 0>;

def : InstAlias<"if ($src1) jumpr $src2",
      (J2_jumprt PredRegs:$src1, IntRegs:$src2), 0>;

def : InstAlias<"if (!$src1) jumpr $src2",
      (J2_jumprf PredRegs:$src1, IntRegs:$src2), 0>;

// maps Vdd = Vss to Vdd = V6_vassignp(Vss)
def : InstAlias<"$Vdd = $Vss",
      (V6_vassignp VecDblRegs:$Vdd, VecDblRegs:$Vss)>,
      Requires<[HasV60T]>;

// maps Vd = #0 to Vd = vxor(Vd, Vd)
def : InstAlias<"$Vd = #0",
      (V6_vxor VectorRegs:$Vd, VectorRegs:$Vd, VectorRegs:$Vd)>,
      Requires<[HasV60T]>;

// maps Vdd  = #0 to Vdd = vsub(Vdd, Vdd)
def : InstAlias<"$Vdd = #0",
      (V6_vsubw_dv VecDblRegs:$Vdd, VecDblRegs:$Vdd, VecDblRegs:$Vdd)>,
      Requires<[HasV60T]>;

// maps   "$Qd = vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd = vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd = vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd &= vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd &= vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd &= vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh_and VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd |= vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd |= vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd |= vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh_or VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd ^= vcmp.eq($Vu.uh, $Vv.uh)" -> "$Qd ^= vcmp.eq($Vu.h, $Vv.h)"
def : InstAlias<"$Qd ^= vcmp.eq($Vu.uh, $Vv.uh)",
      (V6_veqh_xor VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd = vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd = vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd = vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqw VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd &= vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd &= vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd &= vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqw_and VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd |= vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd |= vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd |= vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqh_or VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd ^= vcmp.eq($Vu.uw, $Vv.uw)" -> "$Qd ^= vcmp.eq($Vu.w, $Vv.w)"
def : InstAlias<"$Qd ^= vcmp.eq($Vu.uw, $Vv.uw)",
      (V6_veqw_xor VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd = vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd = vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd = vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd &= vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd &= vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd &= vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb_and VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd |= vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd |= vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd |= vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb_or VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Qd ^= vcmp.eq($Vu.ub, $Vv.ub)" -> "$Qd ^= vcmp.eq($Vu.b, $Vv.b)"
def : InstAlias<"$Qd ^= vcmp.eq($Vu.ub, $Vv.ub)",
      (V6_veqb_xor VecPredRegs:$Qd, VectorRegs:$Vu, VectorRegs:$Vv)>,
      Requires<[HasV60T]>;

// maps   "$Rd.w = vextract($Vu, $Rs)" -> "$Rd = vextract($Vu, $Rs)"
def : InstAlias<"$Rd.w = vextract($Vu, $Rs)",
      (V6_extractw IntRegs:$Rd, VectorRegs:$Vu, IntRegs:$Rs)>,
      Requires<[HasV60T]>;

// Mapping from vtrans2x2(Vy32,Vx32,Rt32) to vshuff(Vy32,Vx32,Rt32)
def : InstAlias<"vtrans2x2($Vy, $Vx, $Rt)",
      (V6_vshuff VectorRegs:$Vy, VectorRegs:$Vx, IntRegs:$Rt)>,
      Requires<[HasV60T]>;

def : InstAlias<"$Vt=vmem($Rs)",
      (V6_vL32b_ai VectorRegs:$Vt, IntRegs:$Rs, 0)>,
      Requires<[HasV60T]>;

def : InstAlias<"$Vt=vmem($Rs):nt",
      (V6_vL32b_nt_ai VectorRegs:$Vt, IntRegs:$Rs, 0)>,
      Requires<[HasV60T]>;

def : InstAlias<"vmem($Rs)=$Vt",
      (V6_vS32b_ai IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"vmem($Rs):nt=$Vt",
      (V6_vS32b_nt_ai IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"vmem($Rs)=$Vt.new",
      (V6_vS32b_new_ai IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"vmem($Rs):nt=$Vt.new",
      (V6_vS32b_nt_new_ai IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if ($Qv) vmem($Rs)=$Vt",
      (V6_vS32b_qpred_ai VecPredRegs:$Qv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if (!$Qv) vmem($Rs)=$Vt",
      (V6_vS32b_nqpred_ai VecPredRegs:$Qv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if ($Qv) vmem($Rs):nt=$Vt",
      (V6_vS32b_nt_qpred_ai VecPredRegs:$Qv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if (!$Qv) vmem($Rs):nt=$Vt",
      (V6_vS32b_nt_nqpred_ai VecPredRegs:$Qv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if ($Pv) vmem($Rs)=$Vt",
      (V6_vS32b_pred_ai PredRegs:$Pv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if (!$Pv) vmem($Rs)=$Vt",
      (V6_vS32b_npred_ai PredRegs:$Pv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if ($Pv) vmem($Rs):nt=$Vt",
      (V6_vS32b_nt_pred_ai PredRegs:$Pv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if (!$Pv) vmem($Rs):nt=$Vt",
      (V6_vS32b_nt_npred_ai PredRegs:$Pv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"$Vt=vmemu($Rs)",
      (V6_vL32Ub_ai VectorRegs:$Vt, IntRegs:$Rs, 0)>,
      Requires<[HasV60T]>;

def : InstAlias<"vmemu($Rs)=$Vt",
      (V6_vS32Ub_ai IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if ($Pv) vmemu($Rs)=$Vt",
      (V6_vS32Ub_pred_ai PredRegs:$Pv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;

def : InstAlias<"if (!$Pv) vmemu($Rs)=$Vt",
      (V6_vS32Ub_npred_ai PredRegs:$Pv, IntRegs:$Rs, 0, VectorRegs:$Vt)>,
      Requires<[HasV60T]>;


