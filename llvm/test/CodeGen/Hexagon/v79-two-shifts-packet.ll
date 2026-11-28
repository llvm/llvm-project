; RUN: llc -march=hexagon -mcpu=hexagonv81 -mattr=+hvxv81,+hvx-length128b < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv75 -mattr=+hvxv75,+hvx-length128b < %s | FileCheck %s --check-prefix=V75

; Check that two shifts are packetized together in the same packet in v79+.
; Before v79 only one shift resource is available.
; CHECK-LABEL: test:
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: }
; V75-LABEL: test:
; V75: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }
; V75-NEXT: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }

define <32 x i32> @test(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vdelta.128B(<32 x i32> %0, <32 x i32> %1)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %1, <32 x i32> %2)
  ret <32 x i32> %3
}

; Check that a shift-resource-only instruction (prefixsum) and a regular shift
; are packetized together in v79+: the regular shift uses the shift/permute
; resource.
; CHECK-LABEL: prefixsum_shift:
; CHECK: v{{[0-9]+}}.b = prefixsum(q{{[0-3]}})
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: }
; V75-LABEL: prefixsum_shift:
; V75: v{{[0-9]+}}.b = prefixsum(q{{[0-3]}})
; V75-NEXT: }
; V75-NEXT: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }

define <32 x i32> @prefixsum_shift(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, i32 %r) #0 {
entry:
  %q = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 %r)
  %s = tail call <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32> %b, <32 x i32> %c)
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vprefixqb.128B(<128 x i1> %q)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %s, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %0, <32 x i32> %1)
  ret <32 x i32> %2
}

; Check that two shifts and an ALU instruction are packetized together in v79+:
; the ALU instruction uses a multiply resource.
; CHECK-LABEL: two_shifts_alu:
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: v{{[0-9]+}}.w = vsub(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: }
; V75-LABEL: two_shifts_alu:
; V75: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: v{{[0-9]+}}.w = vsub(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }
; V75-NEXT: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }

define <32 x i32> @two_shifts_alu(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32> %c, <32 x i32> %d)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %0, <32 x i32> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %3, <32 x i32> %2)
  ret <32 x i32> %4
}

; Check that at most two shifts are packetized together.
; CHECK-LABEL: three_shifts:
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-LABEL: three_shifts:
; V75: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }
; V75-NEXT: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }
; V75-NEXT: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)

define <32 x i32> @three_shifts(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %d)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %0, <32 x i32> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %3, <32 x i32> %2)
  ret <32 x i32> %4
}

; Check that two shifts and a permute are not packetized together: both
; resources that can execute shifts are used.
; CHECK-LABEL: two_shifts_permute:
; CHECK: {
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: v{{[0-9]+}} = vdelta(v{{[0-9]+}},v{{[0-9]+}})
; V75-LABEL: two_shifts_permute:
; V75: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: v{{[0-9]+}} = vdelta(v{{[0-9]+}},v{{[0-9]+}})
; V75-NEXT: }
; V75-NEXT: {
; V75-NEXT: v{{[0-9]+}}.w = vlsr(v{{[0-9]+}}.w,v{{[0-9]+}}.w)
; V75-NEXT: }

define <32 x i32> @two_shifts_permute(<32 x i32> %a, <32 x i32> %b, <32 x i32> %c, <32 x i32> %d) #0 {
entry:
  %0 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %b)
  %1 = tail call <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32> %a, <32 x i32> %c)
  %2 = tail call <32 x i32> @llvm.hexagon.V6.vdelta.128B(<32 x i32> %c, <32 x i32> %d)
  %3 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %0, <32 x i32> %1)
  %4 = tail call <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32> %3, <32 x i32> %2)
  ret <32 x i32> %4
}

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32)
declare <32 x i32> @llvm.hexagon.V6.vprefixqb.128B(<128 x i1>)
declare <32 x i32> @llvm.hexagon.V6.vlsrwv.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vaddw.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vsubw.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vdelta.128B(<32 x i32>, <32 x i32>)

attributes #0 = { nounwind readnone }
