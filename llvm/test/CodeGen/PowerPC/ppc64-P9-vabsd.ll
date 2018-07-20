; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr9 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -verify-machineinstrs | FileCheck %s -check-prefix=CHECK-PWR8 -implicit-check-not vabsdu

; Function Attrs: nounwind readnone
define <4 x i32> @simple_absv_32(<4 x i32> %a) local_unnamed_addr {
entry:
  %sub.i = sub <4 x i32> zeroinitializer, %a
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %a, <4 x i32> %sub.i)
  ret <4 x i32> %0
; CHECK-LABEL: simple_absv_32
; CHECK-DAG: vxor {{[0-9]+}}, [[REG:[0-9]+]], [[REG]]
; CHECK-DAG: xvnegsp 34, 34
; CHECK-DAG: xvnegsp 35, {{[0-9]+}}
; CHECK-NEXT: vabsduw 2, 2, {{[0-9]+}}
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_32
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuwm
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
}

; Function Attrs: nounwind readnone
define <4 x i32> @simple_absv_32_swap(<4 x i32> %a) local_unnamed_addr {
entry:
  %sub.i = sub <4 x i32> zeroinitializer, %a
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %sub.i, <4 x i32> %a)
  ret <4 x i32> %0
; CHECK-LABEL: simple_absv_32_swap
; CHECK-DAG: vxor {{[0-9]+}}, [[REG:[0-9]+]], [[REG]]
; CHECK-DAG: xvnegsp 34, 34
; CHECK-DAG: xvnegsp 35, {{[0-9]+}}
; CHECK-NEXT: vabsduw 2, 2, {{[0-9]+}}
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
; CHECK: mtvsrws {{[0-9]+}}, {{[0-9]+}}
; CHECK-NEXT: vadduhm 2, 2, [[IMM:[0-9]+]]
; CHECK-NEXT: vabsduh 2, 2, [[IMM]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_16
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuhm
; CHECK-PWR8: vmaxsh
; CHECK-PWR8: blr
}

; Function Attrs: nounwind readnone
define <16 x i8> @simple_absv_8(<16 x i8> %a) local_unnamed_addr {
entry:
  %sub.i = sub <16 x i8> zeroinitializer, %a
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %a, <16 x i8> %sub.i)
  ret <16 x i8> %0
; CHECK-LABEL: simple_absv_8
; CHECK: xxspltib {{[0-9]+}}, 128
; CHECK-NEXT: vaddubm 2, 2, [[IMM:[0-9]+]]
; CHECK-NEXT: vabsdub 2, 2, [[IMM]]
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: simple_absv_8
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsububm
; CHECK-PWR8: vmaxsb
; CHECK-PWR8: blr
}

; The select pattern can only be detected for v4i32.
; Function Attrs: norecurse nounwind readnone
define <4 x i32> @sub_absv_32(<4 x i32> %a, <4 x i32> %b) local_unnamed_addr {
entry:
  %0 = sub nsw <4 x i32> %a, %b
  %1 = icmp sgt <4 x i32> %0, <i32 -1, i32 -1, i32 -1, i32 -1>
  %2 = sub <4 x i32> zeroinitializer, %0
  %3 = select <4 x i1> %1, <4 x i32> %0, <4 x i32> %2
  ret <4 x i32> %3
; CHECK-LABEL: sub_absv_32
; CHECK-DAG: xvnegsp 34, 34
; CHECK-DAG: xvnegsp 35, 35
; CHECK-NEXT: vabsduw 2, 2, 3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_32
; CHECK-PWR8: vsubuwm
; CHECK-PWR8: xxlxor
; CHECK-PWR8: blr
}

; FIXME: This does not produce the ISD::ABS that we are looking for.
; We should fix the missing canonicalization.
; We do manage to find the word version of ABS but not the halfword.
; Threfore, we end up doing more work than is required with a pair of abs for word
;  instead of just one for the halfword.
; Function Attrs: norecurse nounwind readnone
define <8 x i16> @sub_absv_16(<8 x i16> %a, <8 x i16> %b) local_unnamed_addr {
entry:
  %0 = sext <8 x i16> %a to <8 x i32>
  %1 = sext <8 x i16> %b to <8 x i32>
  %2 = sub nsw <8 x i32> %0, %1
  %3 = icmp sgt <8 x i32> %2, <i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1, i32 -1>
  %4 = sub nsw <8 x i32> zeroinitializer, %2
  %5 = select <8 x i1> %3, <8 x i32> %2, <8 x i32> %4
  %6 = trunc <8 x i32> %5 to <8 x i16>
  ret <8 x i16> %6
; CHECK-LABEL: sub_absv_16
; CHECK-NOT: vabsduh
; CHECK: vabsduw
; CHECK-NOT: vabsduh
; CHECK: vabsduw
; CHECK-NOT: vabsduh
; CHECK: blr
; CHECK-PWR8-LABEL: sub_absv_16
; CHECK-PWR8: vsubuwm
; CHECK-PWR8: xxlxor
; CHECK-PWR8: blr
}

; FIXME: This does not produce ISD::ABS. This does not even vectorize correctly!
; This function should look like sub_absv_32 and sub_absv_16 except that the type is v16i8.
; Function Attrs: norecurse nounwind readnone
define <16 x i8> @sub_absv_8(<16 x i8> %a, <16 x i8> %b) local_unnamed_addr {
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
; CHECK-LABEL: sub_absv_8
; CHECK-NOT: vabsdub
; CHECK: subf
; CHECK-NOT: vabsdub
; CHECK: xor
; CHECK-NOT: vabsdub
; CHECK: blr
; CHECK-PWR8-LABEL: sub_absv_8
; CHECK-PWR8: subf
; CHECK-PWR8: xor
; CHECK-PWR8: blr
}

; Function Attrs: nounwind readnone
define <4 x i32> @sub_absv_vec_32(<4 x i32> %a, <4 x i32> %b) local_unnamed_addr {
entry:
  %sub = sub <4 x i32> %a, %b
  %sub.i = sub <4 x i32> zeroinitializer, %sub
  %0 = tail call <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32> %sub, <4 x i32> %sub.i)
  ret <4 x i32> %0
; CHECK-LABEL: sub_absv_vec_32
; CHECK: vabsduw 2, 2, 3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_vec_32
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuwm
; CHECK-PWR8: vmaxsw
; CHECK-PWR8: blr
}

; Function Attrs: nounwind readnone
define <8 x i16> @sub_absv_vec_16(<8 x i16> %a, <8 x i16> %b) local_unnamed_addr {
entry:
  %sub = sub <8 x i16> %a, %b
  %sub.i = sub <8 x i16> zeroinitializer, %sub
  %0 = tail call <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16> %sub, <8 x i16> %sub.i)
  ret <8 x i16> %0
; CHECK-LABEL: sub_absv_vec_16
; CHECK: vabsduh 2, 2, 3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_vec_16
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsubuhm
; CHECK-PWR8: vmaxsh
; CHECK-PWR8: blr
}

; Function Attrs: nounwind readnone
define <16 x i8> @sub_absv_vec_8(<16 x i8> %a, <16 x i8> %b) local_unnamed_addr {
entry:
  %sub = sub <16 x i8> %a, %b
  %sub.i = sub <16 x i8> zeroinitializer, %sub
  %0 = tail call <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8> %sub, <16 x i8> %sub.i)
  ret <16 x i8> %0
; CHECK-LABEL: sub_absv_vec_8
; CHECK: vabsdub 2, 2, 3
; CHECK-NEXT: blr
; CHECK-PWR8-LABEL: sub_absv_vec_8
; CHECK-PWR8: xxlxor
; CHECK-PWR8: vsububm
; CHECK-PWR8: vmaxsb
; CHECK-PWR8: blr
}


; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.altivec.vmaxsw(<4 x i32>, <4 x i32>)

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.ppc.altivec.vmaxsh(<8 x i16>, <8 x i16>)

; Function Attrs: nounwind readnone
declare <16 x i8> @llvm.ppc.altivec.vmaxsb(<16 x i8>, <16 x i8>)

