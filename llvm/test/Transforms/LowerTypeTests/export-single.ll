; RUN: opt -S -lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%S/Inputs/use-typeid1-typeid2.yaml -lowertypetests-write-summary=%t < %s | FileCheck %s
; RUN: FileCheck --check-prefix=SUMMARY %s < %t

@foo = constant i32 42, !type !0

!0 = !{i32 0, !"typeid1"}

; CHECK: [[G:@[0-9]+]] = private constant { i32 } { i32 42 }

; CHECK: @__typeid_typeid1_global_addr = hidden alias i8, bitcast ({ i32 }* [[G]] to i8*)
; CHECK: @foo = alias i32, getelementptr inbounds ({ i32 }, { i32 }* [[G]], i32 0, i32 0)

; SUMMARY:      TypeIdMap:
; SUMMARY-NEXT:   typeid1:
; SUMMARY-NEXT:     TTRes:
; SUMMARY-NEXT:       Kind:            Single
; SUMMARY-NEXT:       SizeM1BitWidth:  0
