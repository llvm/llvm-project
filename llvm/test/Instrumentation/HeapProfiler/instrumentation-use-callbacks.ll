; Test heapprof internal compiler flags:
;   -heapprof-use-callbacks
;   -heapprof-memory-access-callback-prefix

; RUN: opt < %s -heapprof -heapprof-module -heapprof-use-callbacks -S | FileCheck %s --check-prefix=CHECK-CALL --check-prefix=CHECK-CALL-DEFAULT
; RUN: opt < %s -heapprof -heapprof-module -heapprof-use-callbacks -heapprof-memory-access-callback-prefix=__foo_ -S | FileCheck %s --check-prefix=CHECK-CALL --check-prefix=CHECK-CALL-CUSTOM
; RUN: opt < %s -heapprof -heapprof-module -heapprof-use-callbacks=false -S | FileCheck %s --check-prefix=CHECK-INLINE
; RUN: opt < %s -heapprof -heapprof-module  -S | FileCheck %s --check-prefix=CHECK-INLINE
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @test_load(i32* %a, i64* %b, i512* %c, i80* %d) {
entry:
; CHECK-CALL:             %[[LOAD_ADDR1:[^ ]*]] = ptrtoint i32* %a to i64
; CHECK-CALL-DEFAULT:     call void @__heapprof_load(i64 %[[LOAD_ADDR1]])
; CHECK-CALL-CUSTOM:      call void @__foo_load(i64 %[[LOAD_ADDR1]])
; CHECK-CALL:             %[[LOAD_ADDR2:[^ ]*]] = ptrtoint i64* %b to i64
; CHECK-CALL-DEFAULT:     call void @__heapprof_load(i64 %[[LOAD_ADDR2]])
; CHECK-CALL-CUSTOM:      call void @__foo_load(i64 %[[LOAD_ADDR2]])
; CHECK-CALL:             %[[LOAD_ADDR3:[^ ]*]] = ptrtoint i512* %c to i64
; CHECK-CALL-DEFAULT:     call void @__heapprof_load(i64 %[[LOAD_ADDR3]])
; CHECK-CALL-CUSTOM:      call void @__foo_load(i64 %[[LOAD_ADDR3]])
; CHECK-CALL:             %[[LOAD_ADDR4:[^ ]*]] = ptrtoint i80* %d to i64
; CHECK-CALL-DEFAULT:     call void @__heapprof_load(i64 %[[LOAD_ADDR4]])
; CHECK-CALL-CUSTOM:      call void @__foo_load(i64 %[[LOAD_ADDR4]])
; CHECK-CALL-DEFAULT-NOT: call void @__heapprof_load
; CHECK-CALL-CUSTOM-NOT:  call void @__foo_load
; CHECK-INLINE-NOT:       call void @__heapprof_load
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = load i64, i64* %b, align 8
  %tmp3 = load i512, i512* %c, align 32
  %tmp4 = load i80, i80* %d, align 8
  ret void
}


