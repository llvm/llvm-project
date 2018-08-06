; RUN: llc -mattr=-neon < %s -verify-machineinstrs -o - | FileCheck %s

target triple = "thumbv7a-none--eabi"

@a = external global i32*
@b = external global i32*

; Function Attrs: nounwind
define void @foo24() #0 {
entry:
; CHECK-LABEL: foo24:
; We use '[rl0-9]*' to allow 'r0'..'r12', 'lr'
; CHECK: movt [[LB:[rl0-9]+]], :upper16:b
; CHECK: movt [[SB:[rl0-9]+]], :upper16:a
; CHECK: add{{s?}}{{(\.w)?}} [[NLB:[rl0-9]+]], [[LB]], #4
; CHECK: adds [[SB]], #4
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]], {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]], [[R4:[rl0-9]+]], [[R5:[rl0-9]+]], [[R6:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]], {[[R1]], [[R2]], [[R3]], [[R4]], [[R5]], [[R6]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 24, i1 false)
  ret void
}

define void @foo28() #0 {
entry:
; CHECK-LABEL: foo28:
; CHECK: movt [[LB:[rl0-9]+]], :upper16:b
; CHECK: movt [[SB:[rl0-9]+]], :upper16:a
; CHECK: add{{(\.w)?}} [[NLB:[rl0-9]+]], [[LB]], #4
; CHECK: adds [[SB]], #4
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]]!, {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]]!, {[[R1]], [[R2]], [[R3]]}
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]], {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]], [[R4:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]], {[[R1]], [[R2]], [[R3]], [[R4]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 28, i1 false)
  ret void
}

define void @foo32() #0 {
entry:
; CHECK-LABEL: foo32:
; CHECK: movt [[LB:[rl0-9]+]], :upper16:b
; CHECK: movt [[SB:[rl0-9]+]], :upper16:a
; CHECK: add{{(\.w)?}} [[NLB:[rl0-9]+]], [[LB]], #4
; CHECK: adds [[SB]], #4
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]]!, {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]], [[R4:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]]!, {[[R1]], [[R2]], [[R3]], [[R4]]}
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]], {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]], [[R4:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]], {[[R1]], [[R2]], [[R3]], [[R4]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 32, i1 false)
  ret void
}

define void @foo36() #0 {
entry:
; CHECK-LABEL: foo36:
; CHECK: movt [[LB:[rl0-9]+]], :upper16:b
; CHECK: movt [[SB:[rl0-9]+]], :upper16:a
; CHECK: add{{(\.w)?}} [[NLB:[rl0-9]+]], [[LB]], #4
; CHECK: adds [[SB]], #4
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]]!, {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]], [[R4:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]]!, {[[R1]], [[R2]], [[R3]], [[R4]]}
; CHECK-NEXT: ldm{{(\.w)?}} [[NLB]], {[[R1:[rl0-9]+]], [[R2:[rl0-9]+]], [[R3:[rl0-9]+]], [[R4:[rl0-9]+]], [[R5:[rl0-9]+]]}
; CHECK-NEXT: stm{{(\.w)?}} [[SB]], {[[R1]], [[R2]], [[R3]], [[R4]], [[R5]]}
  %0 = load i32*, i32** @a, align 4
  %arrayidx = getelementptr inbounds i32, i32* %0, i32 1
  %1 = bitcast i32* %arrayidx to i8*
  %2 = load i32*, i32** @b, align 4
  %arrayidx1 = getelementptr inbounds i32, i32* %2, i32 1
  %3 = bitcast i32* %arrayidx1 to i8*
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %1, i8* align 4 %3, i32 36, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1) #1
