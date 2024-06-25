; RUN: opt -S -passes=lowertypetests -mtriple=i686-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: opt -S -passes=lowertypetests -mtriple=x86_64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,X86 %s
; RUN: opt -S -passes=lowertypetests -mtriple=arm-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,ARM %s
; RUN: opt -S -passes=lowertypetests -mtriple=aarch64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,ARM %s
; RUN: opt -S -passes=lowertypetests -mtriple=riscv32-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,RISCV %s
; RUN: opt -S -passes=lowertypetests -mtriple=riscv64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,RISCV %s
; RUN: opt -S -passes=lowertypetests -mtriple=loongarch64-unknown-linux-gnu %s | FileCheck --check-prefixes=CHECK,LOONGARCH64 %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @x = global ptr null, align 8
@x = global ptr @f, align 8

; CHECK: @x2 = global ptr null, align 8
@x2 = global ptr @f, align 8

; CHECK: @x3 = internal global ptr null, align 8
@x3 = internal constant ptr @f, align 8

; f + addend
; CHECK: @x4 = global ptr null, align 8
@x4 = global ptr getelementptr (i8, ptr @f, i64 42), align 8

; aggregate initializer
; CHECK: @s = global { ptr, ptr, i32 } zeroinitializer, align 8
@s = global { ptr, ptr, i32 } { ptr @f, ptr @f, i32 42 }, align 8

; CHECK:  @llvm.global_ctors = appending global {{.*}}{ i32 0, ptr @__cfi_global_var_init

; CHECK: declare !type !0 extern_weak void @f()
declare !type !0 extern_weak void @f()

; CHECK: define zeroext i1 @check_f()
define zeroext i1 @check_f() {
entry:
; CHECK: [[CMP:%.*]] = icmp ne ptr @f, null
; CHECK: [[SEL:%.*]] = select i1 [[CMP]], ptr @[[JT:.*]], ptr null
; CHECK: [[PTI:%.*]] = ptrtoint ptr [[SEL]] to i1
; CHECK: ret i1 [[PTI]]
  ret i1 ptrtoint (ptr @f to i1)
}

; CHECK: define void @call_f() {
define void @call_f() {
entry:
; CHECK: call void @f()
  call void @f()
  ret void
}

define void @struct() {
; CHECK-LABEL: define void @struct() {
; CHECK: [[CMP:%.*]] = icmp ne ptr @f, null
; CHECK: [[SEL:%.*]] = select i1 [[CMP]], ptr @.cfi.jumptable, ptr null
; CHECK-NEXT: [[PTI:%.*]] = ptrtoint ptr [[SEL]] to i1
; CHECK-NEXT: [[IV:%.*]] = insertvalue { i1, i8 } poison, i1 [[PTI]], 0
; CHECK-NEXT: [[IV2:%.*]] = insertvalue { i1, i8 } [[IV]], i8 0, 1
; CHECK-NEXT: %x = extractvalue { i1, i8 } [[IV2]], 0

entry:
  %x = extractvalue { i1, i8 } { i1 ptrtoint (ptr @f to i1), i8 0 }, 0
  ret void
}

define void @phi(i1 %c) {
; CHECK-LABEL: define void @phi(i1 %c) {
; CHECK: entry:
; CHECK:   [[CMP:%.*]] = icmp ne ptr @f, null
; CHECK:   [[SEL:%.*]] = select i1 [[CMP]], ptr @.cfi.jumptable, ptr null
; CHECK:   br i1 %c, label %if, label %join
; CHECK: if:
; CHECK:   [[CMP2:%.*]] = icmp ne ptr @f, null
; CHECK:   [[SEL2:%.*]] = select i1 [[CMP2]], ptr @.cfi.jumptable, ptr null
; CHECK:   br label %join
; CHECK: join:
; CHECK:   %phi = phi ptr [ [[SEL2]], %if ], [ null, %entry ]
; CHECK:   %phi2 = phi ptr [ null, %if ], [ [[SEL]], %entry ]

entry:
  br i1 %c, label %if, label %join

if:
  br label %join

join:
  %phi = phi ptr [ @f, %if ], [ null, %entry ]
  %phi2 = phi ptr [ null, %if ], [ @f, %entry ]
  ret void
}

define void @phi2(i1 %c, i32 %x) {
; CHECK-LABEL: define void @phi2(i1 %c, i32 %x) {
; CHECK: entry:
; CHECK:   br i1 %c, label %if, label %else
; CHECK: if:                                               ; preds = %entry
; CHECK:   [[CMP:%.*]] = icmp ne ptr @f, null
; CHECK:   [[SEL:%.*]] = select i1 [[CMP]], ptr @.cfi.jumptable, ptr null
; CHECK:   switch i32 %x, label %join [
; CHECK:     i32 0, label %join
; CHECK:   ]
; CHECK: else:                                             ; preds = %entry
; CHECK:   [[CMP2:%.*]] = icmp ne ptr @f, null
; CHECK:   [[SEL2:%.*]] = select i1 [[CMP2]], ptr @.cfi.jumptable, ptr null
; CHECK:   switch i32 %x, label %join [
; CHECK:     i32 0, label %join
; CHECK:   ]
; CHECK: join:                                             ; preds = %else, %else, %if, %if
; CHECK:   %phi2 = phi ptr [ [[SEL]], %if ], [ [[SEL]], %if ], [ [[SEL2]], %else ], [ [[SEL2]], %else ]

entry:
  br i1 %c, label %if, label %else

if:
  switch i32 %x, label %join [
    i32 0, label %join
  ]

else:
  switch i32 %x, label %join [
    i32 0, label %join
  ]

join:
  %phi2 = phi ptr [ @f, %if ], [ @f, %if ], [ @f, %else ], [ @f, %else ]
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @foo(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

; X86: define private void @[[JT]]() #{{.*}} align 8 {
; ARM: define private void @[[JT]]() #{{.*}} align 4 {
; RISCV: define private void @[[JT]]() #{{.*}} align 8 {
; LOONGARCH64: define private void @[[JT]]() #{{.*}} align 8 {

; CHECK-LABEL: define internal void @__cfi_global_var_init() section ".text.startup" {
; CHECK-NEXT: entry:
; CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr @f, null
; CHECK-NEXT: [[SEL:%.*]] = select i1 [[CMP]], ptr @[[JT]], ptr null
; CHECK-NEXT: store ptr [[SEL]], ptr @x, align 8
; CHECK-NEXT: [[CMP2:%.*]] = icmp ne ptr @f, null
; CHECK-NEXT: [[SEL2:%.*]] = select i1 [[CMP2]], ptr @[[JT]], ptr null
; CHECK-NEXT: store ptr [[SEL2]], ptr @x2, align 8
; CHECK-NEXT: [[CMP3:%.*]] = icmp ne ptr @f, null
; CHECK-NEXT: [[SEL3:%.*]] = select i1 [[CMP3]], ptr @[[JT]], ptr null
; CHECK-NEXT: store ptr [[SEL3]], ptr @x3, align 8
; CHECK-NEXT: [[CMP4:%.*]] = icmp ne ptr @f, null
; CHECK-NEXT: [[SEL4:%.*]] = select i1 [[CMP4]], ptr @[[JT]], ptr null
; CHECK-NEXT: [[GEP:%.*]] = getelementptr i8, ptr [[SEL4]], i64 42
; CHECK-NEXT: store ptr [[GEP]], ptr @x4, align 8
; CHECK-NEXT: [[CMP5:%.*]] = icmp ne ptr @f, null
; CHECK-NEXT: [[SEL5:%.*]] = select i1 [[CMP5]], ptr @[[JT]], ptr null
; CHECK-NEXT: [[IV:%.*]] = insertvalue { ptr, ptr, i32 } poison, ptr [[SEL5]], 0
; CHECK-NEXT: [[CMP6:%.*]] = icmp ne ptr @f, null
; CHECK-NEXT: [[SEL6:%.*]] = select i1 [[CMP6]], ptr @[[JT]], ptr null
; CHECK-NEXT: [[IV2:%.*]] = insertvalue { ptr, ptr, i32 } [[IV]], ptr [[SEL6]], 1
; CHECK-NEXT: [[IV3:%.*]] = insertvalue { ptr, ptr, i32 } [[IV2]], i32 42, 2
; CHECK-NEXT: store { ptr, ptr, i32 } [[IV3]], ptr @s, align 8
; CHECK-NEXT: ret void
; CHECK-NEXT: }

!0 = !{i32 0, !"typeid1"}
