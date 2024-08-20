; RUN: opt < %s -S -passes=instrprof -conditional-counter-update | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

@__profn_foo = private constant [3 x i8] c"foo"
@__profn_bar = private constant [3 x i8] c"bar"

; CHECK-LABEL: define void @foo
; CHECK-NEXT: %pgocount = load i8, ptr @__profc_foo, align 1
; CHECK-NEXT: %pgocount.ifnonzero = icmp ne i8 %pgocount, 0
; CHECK-NEXT: br i1 %pgocount.ifnonzero, label %1, label %2

; CHECK-LABEL: 1:
; CHECK-NEXT: store i8 0, ptr @__profc_foo, align 1
; CHECK-NEXT: br label %2

; CHECK-LABEL: 2:
; CHECK-NEXT: ret void
define void @foo() {
  call void @llvm.instrprof.cover(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret void
}

; CHECK-LABEL: define i32 @bar
; CHECK-LABEL: entry:
; CHECK-NEXT: %retval = alloca i32, align 4
; CHECK-NEXT: %cond.addr = alloca i32, align 4
; CHECK-NEXT: store i32 %cond, ptr %cond.addr, align 4
; CHECK-NEXT: %pgocount = load i8, ptr @__profc_bar, align 1
; CHECK-NEXT: %pgocount.ifnonzero = icmp ne i8 %pgocount, 0
; CHECK-NEXT: br i1 %pgocount.ifnonzero, label %0, label %1

; CHECK-LABEL: 0:                                  ; preds = %entry
; CHECK-NEXT: store i8 0, ptr @__profc_bar, align 1
; CHECK-NEXT: br label %1

; CHECK-LABEL: 1:                                  ; preds = %entry, %0
; CHECK-NEXT: %2 = load i32, ptr %cond.addr, align 4
; CHECK-NEXT: %cmp = icmp slt i32 %2, 0
; CHECK-NEXT: br i1 %cmp, label %if.then, label %if.end

; CHECK-LABEL: if.then:                            ; preds = %1
; CHECK-NEXT: %pgocount1 = load i8, ptr getelementptr inbounds ([3 x i8], ptr @__profc_bar, i32 0, i32 1), align 1
; CHECK-NEXT: %pgocount.ifnonzero2 = icmp ne i8 %pgocount1, 0
; CHECK-NEXT: br i1 %pgocount.ifnonzero2, label %3, label %4
define i32 @bar(i32 %cond) #0 {
entry:
  %retval = alloca i32, align 4
  %cond.addr = alloca i32, align 4
  store i32 %cond, ptr %cond.addr, align 4
  call void @llvm.instrprof.cover(ptr @__profn_bar, i64 0, i32 3, i32 0)
  %0 = load i32, ptr %cond.addr, align 4
  %cmp = icmp slt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @llvm.instrprof.cover(ptr @__profn_bar, i64 0, i32 3, i32 1)
  store i32 -1, ptr %retval, align 4
  br label %return

if.end:                                           ; preds = %entry
  call void @llvm.instrprof.cover(ptr @__profn_bar, i64 0, i32 3, i32 2)
  store i32 0, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.end, %if.then
  %1 = load i32, ptr %retval, align 4
  ret i32 %1
}
