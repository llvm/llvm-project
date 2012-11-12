; RUN: llc -mcpu=i686 -mattr=+mmx < %s | FileCheck %s
; ModuleID = 'tq.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-macosx10.6.6"

%0 = type { x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx, x86_mmx }

define i32 @pixman_fill_mmx(i32* nocapture %bits, i32 %stride, i32 %bpp, i32 %x, i32 %y, i32 %width, i32 %height, i32 %xor) nounwind ssp {
entry:
  %conv = zext i32 %xor to i64
  %shl = shl nuw i64 %conv, 32
  %or = or i64 %shl, %conv
  %0 = bitcast i64 %or to x86_mmx
; CHECK:      movq [[MMXR:%mm[0-7],]] {{%mm[0-7]}}
; CHECK-NEXT: movq [[MMXR]] {{%mm[0-7]}}
; CHECK-NEXT: movq [[MMXR]] {{%mm[0-7]}}
; CHECK-NEXT: movq [[MMXR]] {{%mm[0-7]}}
; CHECK-NEXT: movq [[MMXR]] {{%mm[0-7]}}
; CHECK-NEXT: movq [[MMXR]] {{%mm[0-7]}}
; CHECK-NEXT: movq [[MMXR]] {{%mm[0-7]}}
  %1 = tail call %0 asm "movq\09\09$7,\09$0\0Amovq\09\09$7,\09$1\0Amovq\09\09$7,\09$2\0Amovq\09\09$7,\09$3\0Amovq\09\09$7,\09$4\0Amovq\09\09$7,\09$5\0Amovq\09\09$7,\09$6\0A", "=&y,=&y,=&y,=&y,=&y,=&y,=y,y,~{dirflag},~{fpsr},~{flags}"(x86_mmx %0) nounwind, !srcloc !0
  %asmresult = extractvalue %0 %1, 0
  %asmresult6 = extractvalue %0 %1, 1
  %asmresult7 = extractvalue %0 %1, 2
  %asmresult8 = extractvalue %0 %1, 3
  %asmresult9 = extractvalue %0 %1, 4
  %asmresult10 = extractvalue %0 %1, 5
  %asmresult11 = extractvalue %0 %1, 6
; CHECK:      movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
; CHECK-NEXT: movq {{%mm[0-7]}},
  tail call void asm sideeffect "movq\09$1,\09  ($0)\0Amovq\09$2,\09 8($0)\0Amovq\09$3,\0916($0)\0Amovq\09$4,\0924($0)\0Amovq\09$5,\0932($0)\0Amovq\09$6,\0940($0)\0Amovq\09$7,\0948($0)\0Amovq\09$8,\0956($0)\0A", "r,y,y,y,y,y,y,y,y,~{memory},~{dirflag},~{fpsr},~{flags}"(i8* undef, x86_mmx %0, x86_mmx %asmresult, x86_mmx %asmresult6, x86_mmx %asmresult7, x86_mmx %asmresult8, x86_mmx %asmresult9, x86_mmx %asmresult10, x86_mmx %asmresult11) nounwind, !srcloc !1
  tail call void @llvm.x86.mmx.emms() nounwind
  ret i32 1
}

declare void @llvm.x86.mmx.emms() nounwind

!0 = metadata !{i32 888, i32 917, i32 945, i32 973, i32 1001, i32 1029, i32 1057}
!1 = metadata !{i32 1390, i32 1430, i32 1469, i32 1508, i32 1547, i32 1586, i32 1625, i32 1664}
