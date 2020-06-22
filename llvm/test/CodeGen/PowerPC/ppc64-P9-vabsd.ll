; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-PWR8 -implicit-check-not vabsdu
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-PWR7 -implicit-check-not vmaxsd

define <4 x i32> @simple_absv_32(<4 x i32> %a) local_unnamed_addr {
entry:
  %sub.i = sub <4 x i32> zeroinitializer, %a
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %a, <4 x i32> %sub.i)
  ret <4 x i32> %0
; CHECK-LABEL: simple_absv_32
; CHECK-NOT:  vxor 
; CHECK-NOT:  vabsduw
; CHECK:      vnegw v[[REG:[0-9]+]], v2
; CHECK-NEXT: vmaxsw v2, v2, v[[REG]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_32
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuwm
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
; CHECK-PWR7-LABEL: simple_absv_32
; CHECK-PWR7: xxlxor
; CHECK-PWR7: vsubuwm
; CHECK-PWR7: vmaxsw
; CHECK-PWR7: blr
}

define <4 x i32> @simple_absv_32_swap(<4 x i32> %a) local_unnamed_addr {
entry:
  %sub.i = sub <4 x i32> zeroinitializer, %a
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %sub.i, <4 x i32> %a)
  ret <4 x i32> %0
; CHECK-LABEL: simple_absv_32_swap
; CHECK-NOT:  vxor 
; CHECK-NOT:  vabsduw
; CHECK:      vnegw  v[[REG:[0-9]+]], v2
; CHECK-NEXT: vmaxsw v2, v2, v[[REG]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_32_swap
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuwm
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
}

define <8 x i16> @simple_absv_16(<8 x i16> %a) local_unnamed_addr {
entry:
  %sub.i = sub <8 x i16> zeroinitializer, %a
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %a, <8 x i16> %sub.i)
  ret <8 x i16> %0
; CHECK-LABEL: simple_absv_16
; CHECK-NOT:  mtvsrws
; CHECK-NOT:  vabsduh
; CHECK:      xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-NEXT: vsubuhm v[[REG:[0-9]+]], v[[ZERO]], v2
; CHECK-NEXT: vmaxsh v2, v2, v[[REG]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_16
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuhm
; CHECK-PWR8: vmaxsh
; CHECK-PWR8: blr
; CHECK-PWR7-LABEL: simple_absv_16
; CHECK-PWR7: xxlxor
; CHECK-PWR7: vsubuhm
; CHECK-PWR7: vmaxsh
; CHECK-PWR7: blr
}

define <16 x i8> @simple_absv_8(<16 x i8> %a) local_unnamed_addr {
entry:
  %sub.i = sub <16 x i8> zeroinitializer, %a
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %a, <16 x i8> %sub.i)
  ret <16 x i8> %0
; CHECK-LABEL: simple_absv_8
; CHECK-NOT:  xxspltib
; CHECK-NOT:  vabsdub
; CHECK:      xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-NEXT: vsububm v[[REG:[0-9]+]], v[[ZERO]], v2
; CHECK-NEXT: vmaxsb v2, v2, v[[REG]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_8
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsububm
; CHECK-PWR8: vmaxsb
; CHECK-PWR8: blr
; CHECK-PWR7-LABEL: simple_absv_8
; CHECK-PWR7: xxlxor
; CHECK-PWR7: vsububm
; CHECK-PWR7: vmaxsb
; CHECK-PWR7: blr
}

; v2i64 vmax isn't avaiable on pwr7 
define <2 x i64> @sub_absv_64(<2 x i64> %a, <2 x i64> %b) local_unnamed_addr {
entry:
  %0 = sub nsw <2 x i64> %a, %b
  %1 = icmp sgt <2 x i64> %0, <i64 -1, i64 -1>
  %2 = sub <2 x i64> zeroinitializer, %0
  %3 = select <2 x i1> %1, <2 x i64> %0, <2 x i64> %2
  ret <2 x i64> %3
; CHECK-LABEL: sub_absv_64
; CHECK: vsubudm
; CHECK: vnegd
; CHECK: vmaxsd
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_64
; CHECK-PWR8-DAG: vsubudm
; CHECK-PWR8-DAG: xxlxor
; CHECK-PWR8: vmaxsd
; CHECK-PWR8: blr
; CHECK-PWR7-LABEL: sub_absv_64
; CHECK-PWR7-NOT: vmaxsd
; CHECK-PWR7: blr
}

; The select pattern can only be detected for v4i32.
define <4 x i32> @sub_absv_32(<4 x i32> %a, <4 x i32> %b) local_unnamed_addr {
entry:
  %0 = sub nsw <4 x i32> %a, %b
  %1 = icmp sgt <4 x i32> %0, <i32 -1, i32 -1, i32 -1, i32 -1>
  %2 = sub <4 x i32> zeroinitializer, %0
  %3 = select <4 x i1> %1, <4 x i32> %0, <4 x i32> %2
  ret <4 x i32> %3
; CHECK-LABEL: sub_absv_32
; CHECK-NOT:  vsubuwm
; CHECK-NOT:  vnegw
; CHECK-NOT:  vmaxsw
; CHECK-DAG:  xvnegsp v2, v2
; CHECK-DAG:  xvnegsp v3, v3
; CHECK-NEXT: vabsduw v2, v{{[23]}}, v{{[23]}}
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_32
; CHECK-PWR8-DAG: vsubuwm
; CHECK-PWR8-DAG: xxlxor
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
; CHECK-PWR7-LABEL: sub_absv_32
; CHECK-PWR7-DAG: vsubuwm
; CHECK-PWR7-DAG: xxlxor
; CHECK-PWR7: vmaxsw
; CHECK-PWR7: blr
}

define <8 x i16> @sub_absv_16(<8 x i16> %a, <8 x i16> %b) local_unnamed_addr {
entry:
  %0 = sub nsw <8 x i16> %a, %b
  %1 = icmp sgt <8 x i16> %0, <i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1>
  %2 = sub <8 x i16> zeroinitializer, %0
  %3 = select <8 x i1> %1, <8 x i16> %0, <8 x i16> %2
  ret <8 x i16> %3
; CHECK-LABEL: sub_absv_16
; CHECK-NOT:  vabsduh
; CHECK-DAG:  xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-DAG:  vsubuhm v[[SUB:[0-9]+]], v2, v3
; CHECK:      vsubuhm v[[SUB1:[0-9]+]], v[[ZERO]], v[[SUB]]
; CHECK-NEXT: vmaxsh v2, v[[SUB]], v[[SUB1]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_16
; CHECK-PWR8-DAG:  xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-PWR8-DAG:  vsubuhm v[[SUB:[0-9]+]], v2, v3
; CHECK-PWR8:      vsubuhm v[[SUB1:[0-9]+]], v[[ZERO]], v[[SUB]]
; CHECK-PWR8-NEXT: vmaxsh v2, v[[SUB]], v[[SUB1]]
; CHECK-PWR8-NEXT: blr
; CHECK-PWR7-LABEL: sub_absv_16
; CHECK-PWR7-DAG: vsubuhm
; CHECK-PWR7-DAG: xxlxor
; CHECK-PWR7: vmaxsh
; CHECK-PWR7-NEXT: blr
}

define <16 x i8> @sub_absv_8(<16 x i8> %a, <16 x i8> %b) local_unnamed_addr {
entry:
  %0 = sub nsw <16 x i8> %a, %b
  %1 = icmp sgt <16 x i8> %0, <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
  %2 = sub <16 x i8> zeroinitializer, %0
  %3 = select <16 x i1> %1, <16 x i8> %0, <16 x i8> %2
  ret <16 x i8> %3
; CHECK-LABEL: sub_absv_8
; CHECK-NOT:  vabsdub
; CHECK-DAG:  xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-DAG:  vsububm v[[SUB:[0-9]+]], v2, v3
; CHECK:      vsububm v[[SUB1:[0-9]+]], v[[ZERO]], v[[SUB]]
; CHECK-NEXT: vmaxsb v2, v[[SUB]], v[[SUB1]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_8
; CHECK-PWR8-DAG:  xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-PWR8-DAG:  vsububm v[[SUB:[0-9]+]], v2, v3
; CHECK-PWR8:      vsububm v[[SUB1:[0-9]+]], v[[ZERO]], v[[SUB]]
; CHECK-PWR8-NEXT: vmaxsb v2, v[[SUB]], v[[SUB1]]
; CHECK-PWR8-NEXT: blr
; CHECK-PWR7-LABEL: sub_absv_8
; CHECK-PWR7-DAG:  xxlxor
; CHECK-PWR7-DAG:  vsububm
; CHECK-PWR7: vmaxsb
; CHECK-PWR7-NEXT: blr
}

; FIXME: This does not produce the ISD::ABS that we are looking for.
; We should fix the missing canonicalization.
; We do manage to find the word version of ABS but not the halfword.
; Threfore, we end up doing more work than is required with a pair of abs for word
;  instead of just one for the halfword.
define <8 x i16> @sub_absv_16_ext(<8 x i16> %a, <8 x i16> %b) local_unnamed_addr {
entry:
  %0 = sext <8 x i16> %a to <8 x i32>
  %1 = sext <8 x i16> %b to <8 x i32>
  %2 = sub nsw <8 x i32> %0, %1
  %3 = icmp sgt <8 x i32> %2, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %4 = sub nsw <8 x i32> zeroinitializer, %2
  %5 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> %4
  %6 = trunc <8 x i32> %5 to <8 x i16>
  ret <8 x i16> %6
; CHECK-LABEL: sub_absv_16_ext
; CHECK-NOT: vabsduh
; CHECK: vabsduw
; CHECK-NOT: vnegw
; CHECK-NOT: vabsduh
; CHECK: vabsduw
; CHECK-NOT: vnegw
; CHECK-NOT: vabsduh
; CHECK: blr
; CHECK-PWR8-LABEL: sub_absv_16
; CHECK-PWR8-DAG: vsubuwm
; CHECK-PWR8-DAG: xxlxor
; CHECK-PWR8: blr
}

; FIXME: This does not produce ISD::ABS. This does not even vectorize correctly!
; This function should look like sub_absv_32 and sub_absv_16 except that the type is v16i8.
; Function Attrs: norecurse nounwind readnone
define <16 x i8> @sub_absv_8_ext(<16 x i8> %a, <16 x i8> %b) local_unnamed_addr {
entry:
  %vecext = extractelement <16 x i8> %a, i32 0
  %conv = zext i8 %vecext to i32
  %vecext1 = extractelement <16 x i8> %b, i32 0
  %conv2 = zext i8 %vecext1 to i32
  %sub = sub nsw i32 %conv, %conv2
  %ispos = icmp sgt i32 %sub, -1
  %neg = sub nsw i32 0, %sub
  %0 = select i1 %ispos, i32 %sub, i32 %neg
  %conv3 = trunc i32 %0 to i8
  %vecins = insertelement <16 x i8> <i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 undef, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, i8 %conv3, i32 0
  %vecext4 = extractelement <16 x i8> %a, i32 1
  %conv5 = zext i8 %vecext4 to i32
  %vecext6 = extractelement <16 x i8> %b, i32 1
  %conv7 = zext i8 %vecext6 to i32
  %sub8 = sub nsw i32 %conv5, %conv7
  %ispos171 = icmp sgt i32 %sub8, -1
  %neg172 = sub nsw i32 0, %sub8
  %1 = select i1 %ispos171, i32 %sub8, i32 %neg172
  %conv10 = trunc i32 %1 to i8
  %vecins11 = insertelement <16 x i8> %vecins, i8 %conv10, i32 1
  %vecext12 = extractelement <16 x i8> %a, i32 2
  %conv13 = zext i8 %vecext12 to i32
  %vecext14 = extractelement <16 x i8> %b, i32 2
  %conv15 = zext i8 %vecext14 to i32
  %sub16 = sub nsw i32 %conv13, %conv15
  %ispos173 = icmp sgt i32 %sub16, -1
  %neg174 = sub nsw i32 0, %sub16
  %2 = select i1 %ispos173, i32 %sub16, i32 %neg174
  %conv18 = trunc i32 %2 to i8
  %vecins19 = insertelement <16 x i8> %vecins11, i8 %conv18, i32 2
  %vecext20 = extractelement <16 x i8> %a, i32 3
  %conv21 = zext i8 %vecext20 to i32
  %vecext22 = extractelement <16 x i8> %b, i32 3
  %conv23 = zext i8 %vecext22 to i32
  %sub24 = sub nsw i32 %conv21, %conv23
  %ispos175 = icmp sgt i32 %sub24, -1
  %neg176 = sub nsw i32 0, %sub24
  %3 = select i1 %ispos175, i32 %sub24, i32 %neg176
  %conv26 = trunc i32 %3 to i8
  %vecins27 = insertelement <16 x i8> %vecins19, i8 %conv26, i32 3
  %vecext28 = extractelement <16 x i8> %a, i32 4
  %conv29 = zext i8 %vecext28 to i32
  %vecext30 = extractelement <16 x i8> %b, i32 4
  %conv31 = zext i8 %vecext30 to i32
  %sub32 = sub nsw i32 %conv29, %conv31
  %ispos177 = icmp sgt i32 %sub32, -1
  %neg178 = sub nsw i32 0, %sub32
  %4 = select i1 %ispos177, i32 %sub32, i32 %neg178
  %conv34 = trunc i32 %4 to i8
  %vecins35 = insertelement <16 x i8> %vecins27, i8 %conv34, i32 4
  %vecext36 = extractelement <16 x i8> %a, i32 5
  %conv37 = zext i8 %vecext36 to i32
  %vecext38 = extractelement <16 x i8> %b, i32 5
  %conv39 = zext i8 %vecext38 to i32
  %sub40 = sub nsw i32 %conv37, %conv39
  %ispos179 = icmp sgt i32 %sub40, -1
  %neg180 = sub nsw i32 0, %sub40
  %5 = select i1 %ispos179, i32 %sub40, i32 %neg180
  %conv42 = trunc i32 %5 to i8
  %vecins43 = insertelement <16 x i8> %vecins35, i8 %conv42, i32 5
  %vecext44 = extractelement <16 x i8> %a, i32 6
  %conv45 = zext i8 %vecext44 to i32
  %vecext46 = extractelement <16 x i8> %b, i32 6
  %conv47 = zext i8 %vecext46 to i32
  %sub48 = sub nsw i32 %conv45, %conv47
  %ispos181 = icmp sgt i32 %sub48, -1
  %neg182 = sub nsw i32 0, %sub48
  %6 = select i1 %ispos181, i32 %sub48, i32 %neg182
  %conv50 = trunc i32 %6 to i8
  %vecins51 = insertelement <16 x i8> %vecins43, i8 %conv50, i32 6
  %vecext52 = extractelement <16 x i8> %a, i32 7
  %conv53 = zext i8 %vecext52 to i32
  %vecext54 = extractelement <16 x i8> %b, i32 7
  %conv55 = zext i8 %vecext54 to i32
  %sub56 = sub nsw i32 %conv53, %conv55
  %ispos183 = icmp sgt i32 %sub56, -1
  %neg184 = sub nsw i32 0, %sub56
  %7 = select i1 %ispos183, i32 %sub56, i32 %neg184
  %conv58 = trunc i32 %7 to i8
  %vecins59 = insertelement <16 x i8> %vecins51, i8 %conv58, i32 7
  %vecext60 = extractelement <16 x i8> %a, i32 8
  %conv61 = zext i8 %vecext60 to i32
  %vecext62 = extractelement <16 x i8> %b, i32 8
  %conv63 = zext i8 %vecext62 to i32
  %sub64 = sub nsw i32 %conv61, %conv63
  %ispos185 = icmp sgt i32 %sub64, -1
  %neg186 = sub nsw i32 0, %sub64
  %8 = select i1 %ispos185, i32 %sub64, i32 %neg186
  %conv66 = trunc i32 %8 to i8
  %vecins67 = insertelement <16 x i8> %vecins59, i8 %conv66, i32 8
  %vecext68 = extractelement <16 x i8> %a, i32 9
  %conv69 = zext i8 %vecext68 to i32
  %vecext70 = extractelement <16 x i8> %b, i32 9
  %conv71 = zext i8 %vecext70 to i32
  %sub72 = sub nsw i32 %conv69, %conv71
  %ispos187 = icmp sgt i32 %sub72, -1
  %neg188 = sub nsw i32 0, %sub72
  %9 = select i1 %ispos187, i32 %sub72, i32 %neg188
  %conv74 = trunc i32 %9 to i8
  %vecins75 = insertelement <16 x i8> %vecins67, i8 %conv74, i32 9
  %vecext76 = extractelement <16 x i8> %a, i32 10
  %conv77 = zext i8 %vecext76 to i32
  %vecext78 = extractelement <16 x i8> %b, i32 10
  %conv79 = zext i8 %vecext78 to i32
  %sub80 = sub nsw i32 %conv77, %conv79
  %ispos189 = icmp sgt i32 %sub80, -1
  %neg190 = sub nsw i32 0, %sub80
  %10 = select i1 %ispos189, i32 %sub80, i32 %neg190
  %conv82 = trunc i32 %10 to i8
  %vecins83 = insertelement <16 x i8> %vecins75, i8 %conv82, i32 10
  %vecext84 = extractelement <16 x i8> %a, i32 11
  %conv85 = zext i8 %vecext84 to i32
  %vecext86 = extractelement <16 x i8> %b, i32 11
  %conv87 = zext i8 %vecext86 to i32
  %sub88 = sub nsw i32 %conv85, %conv87
  %ispos191 = icmp sgt i32 %sub88, -1
  %neg192 = sub nsw i32 0, %sub88
  %11 = select i1 %ispos191, i32 %sub88, i32 %neg192
  %conv90 = trunc i32 %11 to i8
  %vecins91 = insertelement <16 x i8> %vecins83, i8 %conv90, i32 11
  %vecext92 = extractelement <16 x i8> %a, i32 12
  %conv93 = zext i8 %vecext92 to i32
  %vecext94 = extractelement <16 x i8> %b, i32 12
  %conv95 = zext i8 %vecext94 to i32
  %sub96 = sub nsw i32 %conv93, %conv95
  %ispos193 = icmp sgt i32 %sub96, -1
  %neg194 = sub nsw i32 0, %sub96
  %12 = select i1 %ispos193, i32 %sub96, i32 %neg194
  %conv98 = trunc i32 %12 to i8
  %vecins99 = insertelement <16 x i8> %vecins91, i8 %conv98, i32 12
  %vecext100 = extractelement <16 x i8> %a, i32 13
  %conv101 = zext i8 %vecext100 to i32
  %vecext102 = extractelement <16 x i8> %b, i32 13
  %conv103 = zext i8 %vecext102 to i32
  %sub104 = sub nsw i32 %conv101, %conv103
  %ispos195 = icmp sgt i32 %sub104, -1
  %neg196 = sub nsw i32 0, %sub104
  %13 = select i1 %ispos195, i32 %sub104, i32 %neg196
  %conv106 = trunc i32 %13 to i8
  %vecins107 = insertelement <16 x i8> %vecins99, i8 %conv106, i32 13
  %vecext108 = extractelement <16 x i8> %a, i32 14
  %conv109 = zext i8 %vecext108 to i32
  %vecext110 = extractelement <16 x i8> %b, i32 14
  %conv111 = zext i8 %vecext110 to i32
  %sub112 = sub nsw i32 %conv109, %conv111
  %ispos197 = icmp sgt i32 %sub112, -1
  %neg198 = sub nsw i32 0, %sub112
  %14 = select i1 %ispos197, i32 %sub112, i32 %neg198
  %conv114 = trunc i32 %14 to i8
  %vecins115 = insertelement <16 x i8> %vecins107, i8 %conv114, i32 14
  %vecext116 = extractelement <16 x i8> %a, i32 15
  %conv117 = zext i8 %vecext116 to i32
  %vecext118 = extractelement <16 x i8> %b, i32 15
  %conv119 = zext i8 %vecext118 to i32
  %sub120 = sub nsw i32 %conv117, %conv119
  %ispos199 = icmp sgt i32 %sub120, -1
  %neg200 = sub nsw i32 0, %sub120
  %15 = select i1 %ispos199, i32 %sub120, i32 %neg200
  %conv122 = trunc i32 %15 to i8
  %vecins123 = insertelement <16 x i8> %vecins115, i8 %conv122, i32 15
  ret <16 x i8> %vecins123
; CHECK-LABEL: sub_absv_8_ext
; CHECK-NOT: vabsdub
; CHECK: sub
; CHECK-NOT: vabsdub
; CHECK: xor
; CHECK-NOT: vabsdub
; CHECK: blr
; CHECK-PWR8-LABEL: sub_absv_8_ext
; CHECK-PWR8: sub
; CHECK-PWR8: xor
; CHECK-PWR8: blr
}

define <4 x i32> @sub_absv_vec_32(<4 x i32> %a, <4 x i32> %b) local_unnamed_addr {
entry:
  %sub = sub <4 x i32> %a, %b
  %sub.i = sub <4 x i32> zeroinitializer, %sub
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %sub, <4 x i32> %sub.i)
  ret <4 x i32> %0
; CHECK-LABEL: sub_absv_vec_32
; CHECK-NOT:  vsubuwm
; CHECK-NOT:  vnegw
; CHECK-NOT:  vmaxsw
; CHECK-DAG:  xvnegsp v2, v2
; CHECK-DAG:  xvnegsp v3, v3
; CHECK-NEXT: vabsduw v2, v{{[23]}}, v{{[23]}}
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_vec_32
; CHECK-PWR8-DAG: xxlxor
; CHECK-PWR8-DAG: vsubuwm
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
}

define <8 x i16> @sub_absv_vec_16(<8 x i16> %a, <8 x i16> %b) local_unnamed_addr {
entry:
  %sub = sub <8 x i16> %a, %b
  %sub.i = sub <8 x i16> zeroinitializer, %sub
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %sub, <8 x i16> %sub.i)
  ret <8 x i16> %0
; CHECK-LABEL: sub_absv_vec_16
; CHECK-NOT:  mtvsrws
; CHECK-NOT:  vabsduh
; CHECK-DAG:  xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-DAG:  vsubuhm v[[SUB:[0-9]+]], v2, v3
; CHECK:      vsubuhm v[[SUB1:[0-9]+]], v[[ZERO]], v[[SUB]]
; CHECK-NEXT: vmaxsh v2, v[[SUB]], v[[SUB1]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_vec_16
; CHECK-PWR8-DAG: xxlxor
; CHECK-PWR8-DAG: vsubuhm
; CHECK-PWR8: vmaxsh
; CHECK-PWR8: blr
}

define <16 x i8> @sub_absv_vec_8(<16 x i8> %a, <16 x i8> %b) local_unnamed_addr {
entry:
  %sub = sub <16 x i8> %a, %b
  %sub.i = sub <16 x i8> zeroinitializer, %sub
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %sub, <16 x i8> %sub.i)
  ret <16 x i8> %0
; CHECK-LABEL: sub_absv_vec_8
; CHECK-NOT:  xxspltib
; CHECK-NOT:  vabsdub
; CHECK-DAG:  xxlxor v[[ZERO:[0-9]+]], v[[ZERO]], v[[ZERO]]
; CHECK-DAG:  vsububm v[[SUB:[0-9]+]], v2, v3
; CHECK:      vsububm v[[SUB1:[0-9]+]], v[[ZERO]], v[[SUB]]
; CHECK-NEXT: vmaxsb v2, v[[SUB]], v[[SUB1]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_vec_8
; CHECK-PWR8-DAG: xxlxor
; CHECK-PWR8-DAG: vsububm
; CHECK-PWR8: vmaxsb
; CHECK-PWR8: blr
}

define <4 x i32> @zext_sub_absd32(<4 x i16>, <4 x i16>) local_unnamed_addr {
    %3 = zext <4 x i16> %0 to <4 x i32>
    %4 = zext <4 x i16> %1 to <4 x i32>
    %5 = sub <4 x i32> %3, %4
    %6 = sub <4 x i32> zeroinitializer, %5
    %7 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %5, <4 x i32> %6)
    ret <4 x i32> %7
; CHECK-LABEL: zext_sub_absd32
; CHECK-NOT: xvnegsp
; CHECK:     vabsduw
; CHECK:     blr
; CHECK-PWR8-LABEL: zext_sub_absd32
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
}

define <8 x i16> @zext_sub_absd16(<8 x i8>, <8 x i8>) local_unnamed_addr {
    %3 = zext <8 x i8> %0 to <8 x i16>
    %4 = zext <8 x i8> %1 to <8 x i16>
    %5 = sub <8 x i16> %3, %4
    %6 = sub <8 x i16> zeroinitializer, %5
    %7 = tail call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %5, <8 x i16> %6)
    ret <8 x i16> %7
; CHECK-LABEL: zext_sub_absd16
; CHECK-NOT: vadduhm
; CHECK:     vabsduh
; CHECK:     blr
; CHECK-PWR8-LABEL: zext_sub_absd16
; CHECK-PWR8: vmaxsh
; CHECK-PWR8: blr
}

define <16 x i8> @zext_sub_absd8(<16 x i4>, <16 x i4>) local_unnamed_addr {
    %3 = zext <16 x i4> %0 to <16 x i8>
    %4 = zext <16 x i4> %1 to <16 x i8>
    %5 = sub <16 x i8> %3, %4
    %6 = sub <16 x i8> zeroinitializer, %5
    %7 = tail call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %5, <16 x i8> %6)
    ret <16 x i8> %7
; CHECK-LABEL: zext_sub_absd8
; CHECK-NOT: vaddubm
; CHECK:     vabsdub
; CHECK:     blr
; CHECK-PWR8-LABEL: zext_sub_absd8
; CHECK-PWR8: vmaxsb
; CHECK-PWR8: blr
}

; To verify vabsdu* exploitation for ucmp + sub + select sequence

define <4 x i32> @absd_int32_ugt(<4 x i32>, <4 x i32>) {
  %3 = icmp ugt <4 x i32> %0, %1
  %4 = sub <4 x i32> %0, %1
  %5 = sub <4 x i32> %1, %0
  %6 = select <4 x i1> %3, <4 x i32> %4, <4 x i32> %5
  ret <4 x i32> %6
; CHECK-LABEL: absd_int32_ugt
; CHECK-NOT: vcmpgtuw
; CHECK-NOT: xxsel
; CHECK: vabsduw v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int32_ugt
; CHECK-PWR8: vcmpgtuw
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <4 x i32> @absd_int32_uge(<4 x i32>, <4 x i32>) {
  %3 = icmp uge <4 x i32> %0, %1
  %4 = sub <4 x i32> %0, %1
  %5 = sub <4 x i32> %1, %0
  %6 = select <4 x i1> %3, <4 x i32> %4, <4 x i32> %5
  ret <4 x i32> %6
; CHECK-LABEL: absd_int32_uge
; CHECK-NOT: vcmpgtuw
; CHECK-NOT: xxsel
; CHECK: vabsduw v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int32_uge
; CHECK-PWR8: vcmpgtuw
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <4 x i32> @absd_int32_ult(<4 x i32>, <4 x i32>) {
  %3 = icmp ult <4 x i32> %0, %1
  %4 = sub <4 x i32> %0, %1
  %5 = sub <4 x i32> %1, %0
  %6 = select <4 x i1> %3, <4 x i32> %5, <4 x i32> %4
  ret <4 x i32> %6
; CHECK-LABEL: absd_int32_ult
; CHECK-NOT: vcmpgtuw
; CHECK-NOT: xxsel
; CHECK: vabsduw v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int32_ult
; CHECK-PWR8: vcmpgtuw
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <4 x i32> @absd_int32_ule(<4 x i32>, <4 x i32>) {
  %3 = icmp ule <4 x i32> %0, %1
  %4 = sub <4 x i32> %0, %1
  %5 = sub <4 x i32> %1, %0
  %6 = select <4 x i1> %3, <4 x i32> %5, <4 x i32> %4
  ret <4 x i32> %6
; CHECK-LABEL: absd_int32_ule
; CHECK-NOT: vcmpgtuw
; CHECK-NOT: xxsel
; CHECK: vabsduw v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int32_ule
; CHECK-PWR8: vcmpgtuw
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <8 x i16> @absd_int16_ugt(<8 x i16>, <8 x i16>) {
  %3 = icmp ugt <8 x i16> %0, %1
  %4 = sub <8 x i16> %0, %1
  %5 = sub <8 x i16> %1, %0
  %6 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> %5
  ret <8 x i16> %6
; CHECK-LABEL: absd_int16_ugt
; CHECK-NOT: vcmpgtuh
; CHECK-NOT: xxsel
; CHECK: vabsduh v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int16_ugt
; CHECK-PWR8: vcmpgtuh
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <8 x i16> @absd_int16_uge(<8 x i16>, <8 x i16>) {
  %3 = icmp uge <8 x i16> %0, %1
  %4 = sub <8 x i16> %0, %1
  %5 = sub <8 x i16> %1, %0
  %6 = select <8 x i1> %3, <8 x i16> %4, <8 x i16> %5
  ret <8 x i16> %6
; CHECK-LABEL: absd_int16_uge
; CHECK-NOT: vcmpgtuh
; CHECK-NOT: xxsel
; CHECK: vabsduh v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int16_uge
; CHECK-PWR8: vcmpgtuh
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <8 x i16> @absd_int16_ult(<8 x i16>, <8 x i16>) {
  %3 = icmp ult <8 x i16> %0, %1
  %4 = sub <8 x i16> %0, %1
  %5 = sub <8 x i16> %1, %0
  %6 = select <8 x i1> %3, <8 x i16> %5, <8 x i16> %4
  ret <8 x i16> %6
; CHECK-LABEL: absd_int16_ult
; CHECK-NOT: vcmpgtuh
; CHECK-NOT: xxsel
; CHECK: vabsduh v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int16_ult
; CHECK-PWR8: vcmpgtuh
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <8 x i16> @absd_int16_ule(<8 x i16>, <8 x i16>) {
  %3 = icmp ule <8 x i16> %0, %1
  %4 = sub <8 x i16> %0, %1
  %5 = sub <8 x i16> %1, %0
  %6 = select <8 x i1> %3, <8 x i16> %5, <8 x i16> %4
  ret <8 x i16> %6
; CHECK-LABEL: absd_int16_ule
; CHECK-NOT: vcmpgtuh
; CHECK-NOT: xxsel
; CHECK: vabsduh v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int16_ule
; CHECK-PWR8: vcmpgtuh
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <16 x i8> @absd_int8_ugt(<16 x i8>, <16 x i8>) {
  %3 = icmp ugt <16 x i8> %0, %1
  %4 = sub <16 x i8> %0, %1
  %5 = sub <16 x i8> %1, %0
  %6 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> %5
  ret <16 x i8> %6
; CHECK-LABEL: absd_int8_ugt
; CHECK-NOT: vcmpgtub
; CHECK-NOT: xxsel
; CHECK: vabsdub v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int8_ugt
; CHECK-PWR8: vcmpgtub
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <16 x i8> @absd_int8_uge(<16 x i8>, <16 x i8>) {
  %3 = icmp uge <16 x i8> %0, %1
  %4 = sub <16 x i8> %0, %1
  %5 = sub <16 x i8> %1, %0
  %6 = select <16 x i1> %3, <16 x i8> %4, <16 x i8> %5
  ret <16 x i8> %6
; CHECK-LABEL: absd_int8_uge
; CHECK-NOT: vcmpgtub
; CHECK-NOT: xxsel
; CHECK: vabsdub v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int8_uge
; CHECK-PWR8: vcmpgtub
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <16 x i8> @absd_int8_ult(<16 x i8>, <16 x i8>) {
  %3 = icmp ult <16 x i8> %0, %1
  %4 = sub <16 x i8> %0, %1
  %5 = sub <16 x i8> %1, %0
  %6 = select <16 x i1> %3, <16 x i8> %5, <16 x i8> %4
  ret <16 x i8> %6
; CHECK-LABEL: absd_int8_ult
; CHECK-NOT: vcmpgtub
; CHECK-NOT: xxsel
; CHECK: vabsdub v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int8_ult
; CHECK-PWR8: vcmpgtub
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <16 x i8> @absd_int8_ule(<16 x i8>, <16 x i8>) {
  %3 = icmp ule <16 x i8> %0, %1
  %4 = sub <16 x i8> %0, %1
  %5 = sub <16 x i8> %1, %0
  %6 = select <16 x i1> %3, <16 x i8> %5, <16 x i8> %4
  ret <16 x i8> %6
; CHECK-LABEL: absd_int8_ule
; CHECK-NOT: vcmpgtub
; CHECK-NOT: xxsel
; CHECK: vabsdub v2, v2, v3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: absd_int8_ule
; CHECK-PWR8: vcmpgtub
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

; some cases we are unable to optimize
; check whether goes beyond the scope
define <4 x i32> @absd_int32_ugt_opp(<4 x i32>, <4 x i32>) {
  %3 = icmp ugt <4 x i32> %0, %1
  %4 = sub <4 x i32> %0, %1
  %5 = sub <4 x i32> %1, %0
  %6 = select <4 x i1> %3, <4 x i32> %5, <4 x i32> %4
  ret <4 x i32> %6
; CHECK-LABEL: absd_int32_ugt_opp
; CHECK-NOT: vabsduw
; CHECK: vcmpgtuw
; CHECK: xxsel
; CHECK: blr
; CHECK-PWR8-LABEL: absd_int32_ugt_opp
; CHECK-PWR8: vcmpgtuw
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

define <2 x i64> @absd_int64_ugt(<2 x i64>, <2 x i64>) {
  %3 = icmp ugt <2 x i64> %0, %1
  %4 = sub <2 x i64> %0, %1
  %5 = sub <2 x i64> %1, %0
  %6 = select <2 x i1> %3, <2 x i64> %4, <2 x i64> %5
  ret <2 x i64> %6
; CHECK-LABEL: absd_int64_ugt
; CHECK-NOT: vabsduw
; CHECK: vcmpgtud
; CHECK: xxsel
; CHECK: blr
; CHECK-PWR8-LABEL: absd_int64_ugt
; CHECK-PWR8: vcmpgtud
; CHECK-PWR8: xxsel
; CHECK-PWR8: blr
}

declare <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32>, <4 x i32>)

declare <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16>, <8 x i16>)

declare <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8>, <16 x i8>)

