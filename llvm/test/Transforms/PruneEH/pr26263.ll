; PruneEH is less powerful than simplify-cfg in terms of cfg simplification,
; so it leaves some of the unreachable stuff hanging around.
; Checking it with CHECK-OLD.
;
; RUN: opt -prune-eh -S < %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-OLD
; RUN: opt -passes='function-attrs,function(simplify-cfg)' -S < %s | FileCheck %s  --check-prefix=CHECK --check-prefix=CHECK-NEW

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc"

declare void @neverthrows() nounwind

define void @test1() personality i32 (...)* @__CxxFrameHandler3 {
  invoke void @neverthrows()
          to label %try.cont unwind label %cleanuppad

try.cont:
  ret void

cleanuppad:
  %cp = cleanuppad within none []
  br label %cleanupret

cleanupret:
  cleanupret from %cp unwind to caller
}

; CHECK-LABEL: define void @test1(
; CHECK:       call void @neverthrows()
; CHECK-NEW-NEXT: ret void
; CHECK-NEW-NEXT: }
; CHECK-OLD:	  ret void

; CHECK-OLD: %[[cp:.*]] = cleanuppad within none []
; CHECK-OLD-NEXT: unreachable

; CHECK-OLD: cleanupret from %[[cp]] unwind to caller

define void @test2() personality i32 (...)* @__CxxFrameHandler3 {
  invoke void @neverthrows()
          to label %try.cont unwind label %catchswitch

try.cont:
  ret void

catchswitch:
  %cs = catchswitch within none [label %catchpad] unwind to caller

catchpad:
  %cp = catchpad within %cs []
  unreachable

ret:
  ret void
}

; CHECK-LABEL: define void @test2(
; CHECK:       call void @neverthrows()
; CHECK-NEW-NEXT: ret void
; CHECK-NEW-NEXT: }
; CHECK-OLD:      ret void

; CHECK-OLD: %[[cs:.*]] = catchswitch within none [label

; CHECK-OLD: catchpad within %[[cs]] []
; CHECK-OLD-NEXT: unreachable

; CHECK-OLD:ret void

declare i32 @__CxxFrameHandler3(...)
