; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-load=false -dfsan-combine-offset-labels-on-gep=false -dfsan-combine-taint-lookup-table=lookup_table_a -S | FileCheck %s --check-prefixes=CHECK,LOOKUP_A
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-load=false -dfsan-combine-offset-labels-on-gep=false -S | FileCheck %s --check-prefixes=CHECK,NO_LOOKUP_A
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
@lookup_table_a = external local_unnamed_addr constant [256 x i8], align 16
@lookup_table_b = external local_unnamed_addr constant [256 x i8], align 16

define i8 @load_lookup_table_a(i8 %p) {
  ; CHECK-LABEL:           @load_lookup_table_a.dfsan
  ; CHECK-NEXT:            %[[#PS:]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK-NEXT:            %c = zext i8 %p to i64
  ; CHECK-NEXT:            %b = getelementptr inbounds [256 x i8], ptr @lookup_table_a, i64 0, i64 %c
  ; CHECK-NEXT:            %a = load i8, ptr %b, align 1
  ; Propagates p shadow when lookup_table_a flag is provided, otherwise propagates 0 shadow
  ; LOOKUP_A-NEXT:         store i8 %[[#PS]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; NO_LOOKUP_A-NEXT:      store i8 0, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i8 %a

  %c = zext i8 %p to i64
  %b = getelementptr inbounds [256 x i8], ptr @lookup_table_a, i64 0, i64 %c
  %a = load i8, ptr %b
  ret i8 %a
}

define i8 @load_lookup_table_b(i8 %p) {
  ; CHECK-LABEL:           @load_lookup_table_b.dfsan
  ; CHECK-NEXT:            %[[#PS:]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK-NEXT:            %c = zext i8 %p to i64
  ; CHECK-NEXT:            %b = getelementptr inbounds [256 x i8], ptr @lookup_table_b, i64 0, i64 %c
  ; CHECK-NEXT:            %a = load i8, ptr %b, align 1
  ; Propagates 0 shadow
  ; CHECK-NEXT:            store i8 0, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i8 %a

  %c = zext i8 %p to i64
  %b = getelementptr inbounds [256 x i8], ptr @lookup_table_b, i64 0, i64 %c
  %a = load i8, ptr %b, align 1
  ret i8 %a
}
