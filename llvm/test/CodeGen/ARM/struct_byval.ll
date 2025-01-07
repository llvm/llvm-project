; RUN: llc < %s -mtriple=armv7-apple-ios6.0 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios6.0 | FileCheck %s
; RUN: llc < %s -mtriple=armv7-unknown-nacl-gnueabi | FileCheck %s -check-prefix=NACL
; RUN: llc < %s -mtriple=armv5-none-linux-gnueabi | FileCheck %s -check-prefix=NOMOVT

; NOMOVT-NOT: movt

; rdar://9877866
%struct.SmallStruct = type { i32, [8 x i32], [37 x i8] }
%struct.LargeStruct = type { i32, [1001 x i8], [300 x i32] }

define i32 @f() nounwind ssp {
entry:
; CHECK-LABEL: f:
; CHECK: ldr
; CHECK: str
; CHECK-NOT:bne
  %st = alloca %struct.SmallStruct, align 4
  %call = call i32 @e1(ptr byval(%struct.SmallStruct) %st)
  ret i32 0
}

; Generate a loop for large struct byval
define i32 @g() nounwind ssp {
entry:
; CHECK-LABEL: g:
; CHECK: ldr
; CHECK: sub
; CHECK: str
; CHECK: bne
; NACL-LABEL: g:
; Ensure that use movw instead of constpool for the loop trip count. But don't
; match the __stack_chk_guard movw
; NACL: movw {{r[0-9]+|lr}}, #
; NACL: ldr
; NACL: sub
; NACL: str
; NACL: bne
  %st = alloca %struct.LargeStruct, align 4
  %call = call i32 @e2(ptr byval(%struct.LargeStruct) %st)
  ret i32 0
}

; Generate a loop using NEON instructions
define i32 @h() nounwind ssp {
entry:
; CHECK-LABEL: h:
; CHECK: vld1
; CHECK: sub
; CHECK: vst1
; CHECK: bne
; NACL: movw {{r[0-9]+|lr}}, #
; NACL: vld1
; NACL: sub
; NACL: vst1
; NACL: bne
  %st = alloca %struct.LargeStruct, align 16
  %call = call i32 @e3(ptr byval(%struct.LargeStruct) align 16 %st)
  ret i32 0
}

declare i32 @e1(ptr nocapture byval(%struct.SmallStruct) %in) nounwind
declare i32 @e2(ptr nocapture byval(%struct.LargeStruct) %in) nounwind
declare i32 @e3(ptr nocapture byval(%struct.LargeStruct) align 16 %in) nounwind

; We can do tail call here since s is in the incoming argument area.
define void @f5(i32 %a, i32 %b, i32 %c, i32 %d, ptr nocapture byval(%struct.SmallStruct) %s) nounwind optsize {
; CHECK-LABEL: f5
; CHECK: b{{(\.w)?}} _consumestruct
entry:
  tail call void @consumestruct(ptr %s, i32 80) optsize
  ret void
}

define void @f6(i32 %a, i32 %b, i32 %c, i32 %d, ptr nocapture byval(%struct.SmallStruct) %s) nounwind optsize {
; CHECK-LABEL: f6
; CHECK: b{{(\.w)?}} _consumestruct
entry:
  tail call void @consumestruct(ptr %s, i32 80) optsize
  ret void
}

declare void @consumestruct(ptr nocapture %structp, i32 %structsize) nounwind

; PR17309
%struct.I.8 = type { [10 x i32], [3 x i8] }

declare void @use_I(ptr byval(%struct.I.8))
define void @test_I_16() {
; CHECK-LABEL: test_I_16
; CHECK: ldrb
; CHECK: strb
entry:
  call void @use_I(ptr byval(%struct.I.8) align 16 undef)
  ret void
}
