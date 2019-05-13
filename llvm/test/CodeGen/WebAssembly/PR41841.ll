; RUN: llc < %s -O0 -wasm-disable-explicit-locals -wasm-keep-registers -asm-verbose=false | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare void @foo(i128)

; CHECK-LABEL: test_zext:
; CHECK-NEXT: .functype test_zext (i32) -> (){{$}}
; CHECK-NEXT: i64.extend_i32_u $[[TMP3:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: i64.const $[[TMP4:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.and $[[TMP1:[0-9]+]]=, $[[TMP3]], $[[TMP4]]{{$}}
; CHECK-NEXT: i64.const $[[TMP2:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: call foo, $[[TMP1]], $[[TMP2]]{{$}}
; CHECK-NEXT: return{{$}}
define void @test_zext(i1 %b) nounwind {
  %res = zext i1 %b to i128
  br label %next

next:                                             ; preds = %start
  call void @foo(i128 %res)
  ret void
}

; CHECK-LABEL: test_sext:
; CHECK-NEXT:.functype test_sext (i32) -> (){{$}}
; CHECK-NEXT: i64.extend_i32_u $[[TMP3:[0-9]+]]=, $0{{$}}
; CHECK-NEXT: i64.const $[[TMP4:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: i64.and $[[TMP5:[0-9]+]]=, $[[TMP3]], $[[TMP4]]{{$}}
; CHECK-NEXT: i64.const $[[TMP6:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: i64.sub $[[TMP1:[0-9]+]]=, $[[TMP6]], $[[TMP5]]{{$}}
; CHECK-NEXT: local.copy $[[TMP2:[0-9]+]]=, $[[TMP1]]{{$}}
; CHECK-NEXT: call foo, $[[TMP1]], $[[TMP2]]{{$}}
; CHECK-NEXT: return{{$}}
define void @test_sext(i1 %b) nounwind {
  %res = sext i1 %b to i128
  br label %next

next:                                             ; preds = %start
  call void @foo(i128 %res)
  ret void
}
