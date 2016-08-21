; RUN: opt < %s -asan -asan-module -asan-experimental-poisoning -S | FileCheck --check-prefixes=CHECK-ON,CHECK %s
; RUN: opt < %s -asan -asan-module -S | FileCheck --check-prefixes=CHECK-OFF,CHECK %s

target datalayout = "e-i64:64-f80:128-s:64-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @Foo(i8*)

define void @Bar() uwtable sanitize_address {
entry:
  ; CHECK: store i32 -235802127
  ; CHECK: store i64 -868082074056920318
  ; CHECK: store i64 -868082074056920077
  ; CHECK: store i16 -3085
  ; CHECK: store i8 -13
  ; CHECK-LABEL: call void @Foo
  
  ; CHECK-LABEL: <label>
  ; CHECK-ON-NOT: store i64
  ; CHECK-ON: call void @__asan_set_shadow_f5(i64 %{{[0-9]+}}, i64 128)
 
  ; CHECK-OFF-NOT: call void @__asan_set_shadow_f5
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; CHECK-OFF: store i64 -723401728380766731
  ; And more...

  ; CHECK-LABEL: <label>
  ; CHECK-NOT: call void @__asan_set_shadow_00
  ; CHECK: store i32 0
  ; CHECK: store i64 0
  ; CHECK: store i64 0
  ; CHECK: store i16 0
  ; CHECK: store i8 0

  ; CHECK-LABEL: <label>
  ; CHECK: ret void

  %x = alloca [650 x i8], align 16
  %arraydecay = getelementptr inbounds [650 x i8], [650 x i8]* %x, i64 0, i64 0
  call void @Foo(i8* %arraydecay)
  ret void
}

; CHECK-ON: declare void @__asan_set_shadow_00(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f1(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f2(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f3(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f5(i64, i64)
; CHECK-ON: declare void @__asan_set_shadow_f8(i64, i64)

; CHECK-OFF-NOT: declare void @__asan_set_shadow_
