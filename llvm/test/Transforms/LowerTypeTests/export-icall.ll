; RUN: opt -S -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml -lowertypetests-write-summary=%t < %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @h(i8 %x) !type !2 {
  ret void
}

declare !type !8 void @f(i32 %x)
define available_externally void @f2(i32 %x) !type !8 {
  ret void
}
define void @f3(i32 %x) !type !8 {
  ret void
}

!cfi.functions = !{!0, !1, !3, !9, !10, !4, !5, !6}

; declaration of @h with a different type is ignored
!0 = !{!"h", i8 1, !7}

; extern_weak declaration of @h with a different type is ignored as well
!1 = !{!"h", i8 2, !8}
!2 = !{i64 0, !"typeid1"}

; definitions of @f and @f2 replace types on the IR declarations above
!3 = !{!"f", i8 0, !2}
!9 = !{!"f2", i8 0, !2}
!10 = !{!"f3", i8 0, !2}
!4 = !{!"external", i8 1, !2}
!5 = !{!"external_weak", i8 2, !2}
!6 = !{!"g", i8 0, !7}
!7 = !{i64 0, !"typeid2"}
!8 = !{i64 0, !"typeid3"}


; CHECK-DAG: @__typeid_typeid1_global_addr = hidden alias i8, bitcast (void ()* [[JT1:.*]] to i8*)
; CHECK-DAG: @__typeid_typeid1_align = hidden alias i8, inttoptr (i8 3 to i8*)
; CHECK-DAG: @__typeid_typeid1_size_m1 = hidden alias i8, inttoptr (i64 4 to i8*)

; CHECK-DAG: @h                    = alias void (i8), bitcast (void ()* [[JT1]] to void (i8)*)
; CHECK-DAG: @f                    = alias void (i32), {{.*}}getelementptr {{.*}}void ()* [[JT1]]
; CHECK-DAG: @f2                   = alias void (i32), {{.*}}getelementptr {{.*}}void ()* [[JT1]]
; CHECK-DAG: @external.cfi_jt      = hidden alias void (), {{.*}}getelementptr {{.*}}void ()* [[JT1]]
; CHECK-DAG: @external_weak.cfi_jt = hidden alias void (), {{.*}}getelementptr {{.*}}void ()* [[JT1]]

; CHECK-DAG: @__typeid_typeid2_global_addr = hidden alias i8, bitcast (void ()* [[JT2:.*]] to i8*)

; CHECK-DAG: @g                    = alias void (), void ()* [[JT2]]

; CHECK-DAG: define internal void @h.cfi(i8 {{.*}}) !type !{{.*}}
; CHECK-DAG: declare !type !{{.*}} void @external()
; CHECK-DAG: declare !type !{{.*}} void @external_weak()
; CHECK-DAG: declare !type !{{.*}} void @f.cfi(i32)
; CHECK-DAG: declare !type !{{.*}} void @f2.cfi(i32)
; CHECK-DAG: define void @f3(i32 {{.*}}) !type !3
; CHECK-DAG: !3 = !{i64 0, !"typeid3"}
; CHECK-DAG: declare !type !{{.*}} void @g.cfi()


; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            AllOnes
; SUMMARY-NEXT:       SizeM1BitWidth:  7
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:
; SUMMARY-NEXT:   typeid2:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Single
; SUMMARY-NEXT:       SizeM1BitWidth:  0
; SUMMARY-NEXT:       AlignLog2:       0
; SUMMARY-NEXT:       SizeM1:          0
; SUMMARY-NEXT:       BitMask:         0
; SUMMARY-NEXT:       InlineBits:      0
; SUMMARY-NEXT:     WPDRes:

; SUMMARY:      CfiFunctionDefs:
; SUMMARY-NEXT:   - f
; SUMMARY-NEXT:   - f2
; SUMMARY-NEXT:   - g
; SUMMARY-NEXT:   - h
; SUMMARY-NEXT: CfiFunctionDecls:
; SUMMARY-NEXT:   - external
; SUMMARY-NEXT:   - external_weak
; SUMMARY-NEXT: ...
