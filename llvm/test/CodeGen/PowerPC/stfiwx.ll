; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mattr=stfiwx | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu -mattr=-stfiwx | FileCheck -check-prefix=CHECK-LS %s

define void @test1(float %a, ptr %b) nounwind {
; CHECK-LABEL: @test1
; CHECK-LS-LABEL: @test1
        %tmp.2 = fptosi float %a to i32         ; <i32> [#uses=1]
        store i32 %tmp.2, ptr %b
        ret void

; CHECK: stwu
; CHECK-NOT: lwz
; CHECK-NOT: stw
; CHECK: stfiwx
; CHECK: blr

; CHECK-LS: lwz
; CHECK-LS: stw
; CHECK-LS-NOT: stfiwx
; CHECK-LS: blr
}

define void @test2(float %a, ptr %b, i32 %i) nounwind {
; CHECK-LABEL: @test2
; CHECK-LS-LABEL: @test2
        %tmp.2 = getelementptr i32, ptr %b, i32 1           ; <ptr> [#uses=1]
        %tmp.5 = getelementptr i32, ptr %b, i32 %i          ; <ptr> [#uses=1]
        %tmp.7 = fptosi float %a to i32         ; <i32> [#uses=3]
        store i32 %tmp.7, ptr %tmp.5
        store i32 %tmp.7, ptr %tmp.2
        store i32 %tmp.7, ptr %b
        ret void

; CHECK: stwu
; CHECK-NOT: lwz
; CHECK-NOT: stw
; CHECK: stfiwx
; CHECK: blr

; CHECK-LS: lwz
; CHECK-LS: stw
; CHECK-LS-NOT: stfiwx
; CHECK-LS: blr
}
