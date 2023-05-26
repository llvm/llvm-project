; RUN: opt < %s -passes=argpromotion -S | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

%struct.ss = type { i32, i64 }

define internal void @f(ptr byval(ptr) align 4 %p) nounwind  {
; CHECK-LABEL: define {{[^@]+}}@f
; CHECK-SAME: (ptr byval(ptr) align 4 [[P:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    store ptr [[P]], ptr [[P]]
; CHECK-NEXT:    ret void
;
entry:
  store ptr %p, ptr %p
  ret void
}

define internal void @g(ptr byval(ptr) align 4 %p) nounwind  {
; CHECK-LABEL: define {{[^@]+}}@g
; CHECK-SAME: (ptr byval(ptr) align 4 [[P:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr i8, ptr [[P]], i64 4
; CHECK-NEXT:    store ptr [[P]], ptr [[P1]]
; CHECK-NEXT:    ret void
;
entry:
  %p1 = getelementptr i8, ptr %p, i64 4
  store ptr %p, ptr %p1
  ret void
}

define internal void @h(ptr byval(ptr) align 4 %p) nounwind  {
; CHECK-LABEL: define {{[^@]+}}@h
; CHECK-SAME: (ptr byval(ptr) align 4 [[P:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[P1:%.*]] = getelementptr i8, ptr [[P]], i64 4
; CHECK-NEXT:    store ptr [[P1]], ptr [[P]]
; CHECK-NEXT:    ret void
;
entry:
  %p1 = getelementptr i8, ptr %p, i64 4
  store ptr %p1, ptr %p
  ret void
}

define internal void @k(ptr byval(ptr) align 4 %p) nounwind  {
; CHECK-LABEL: define {{[^@]+}}@k
; CHECK-SAME: (ptr byval(ptr) align 4 [[P:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[X:%.*]] = load ptr, ptr [[P]]
; CHECK-NEXT:    store ptr [[P]], ptr [[X]]
; CHECK-NEXT:    ret void
;
entry:
  %x = load ptr, ptr %p
  store ptr %p, ptr %x
  ret void
}

define internal void @l(ptr byval(ptr) align 4 %p) nounwind  {
; CHECK-LABEL: define {{[^@]+}}@l
; CHECK-SAME: () #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    ret void
;
entry:
  %x = load ptr, ptr %p
  store ptr %x, ptr %p
  ret void
}

define i32 @main() nounwind  {
; CHECK-LABEL: define {{[^@]+}}@main
; CHECK-SAME: () #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[S:%.*]] = alloca [[STRUCT_SS:%.*]], align 32
; CHECK-NEXT:    [[TEMP1:%.*]] = getelementptr [[STRUCT_SS]], ptr [[S]], i32 0, i32 0
; CHECK-NEXT:    store i32 1, ptr [[TEMP1]], align 4
; CHECK-NEXT:    [[TEMP4:%.*]] = getelementptr [[STRUCT_SS]], ptr [[S]], i32 0, i32 1
; CHECK-NEXT:    store i64 2, ptr [[TEMP4]], align 8
; CHECK-NEXT:    call void @f(ptr byval(ptr) align 4 [[S]]) #[[ATTR0]]
; CHECK-NEXT:    call void @g(ptr byval(ptr) align 4 [[S]]) #[[ATTR0]]
; CHECK-NEXT:    call void @h(ptr byval(ptr) align 4 [[S]]) #[[ATTR0]]
; CHECK-NEXT:    call void @k(ptr byval(ptr) align 4 [[S]]) #[[ATTR0]]
; CHECK-NEXT:    call void @l() #[[ATTR0]]
; CHECK-NEXT:    ret i32 0
;
entry:
  %S = alloca %struct.ss, align 32
  %temp1 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 0
  store i32 1, i32* %temp1, align 4
  %temp4 = getelementptr %struct.ss, %struct.ss* %S, i32 0, i32 1
  store i64 2, i64* %temp4, align 8
  call void @f(ptr byval(ptr) align 4 %S) nounwind
  call void @g(ptr byval(ptr) align 4 %S) nounwind
  call void @h(ptr byval(ptr) align 4 %S) nounwind
  call void @k(ptr byval(ptr) align 4 %S) nounwind
  call void @l(ptr byval(ptr) align 4 %S) nounwind
  ret i32 0
}
