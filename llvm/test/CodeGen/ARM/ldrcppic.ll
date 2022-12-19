; The failure is caused by ARM LDRcp/PICADD pairs. In PIC mode, the constant pool
; need a label to do address computation. This label is emitted when backend emits
; PICADD. When the target becomes dead, PICADD will be deleted. without this patch
; LDRcp is dead but not being deleted. This will cause a dead contant pool entry
; using a non existing label. This will cause an error in MC object emitting pass.

; RUN: llc -relocation-model=pic -mcpu=cortex-a53 %s -filetype=obj -o - | llvm-nm - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8-unknown-linux-android"

@_ZN15UsecaseSelector25AllowedImplDefinedFormatsE = external dso_local unnamed_addr constant <{ i32, i32, i32, i32, [12 x i32] }>, align 4

; Function Attrs: noinline nounwind optnone sspstrong uwtable
define dso_local fastcc void @_ZN15UsecaseSelector26IsAllowedImplDefinedFormatE15ChiBufferFormatj() unnamed_addr #1 align 2 {
  br label %1

; <label>:1:                                      ; preds = %13, %0
  %2 = icmp ult i32 undef, 4
  br i1 %2, label %3, label %14

; <label>:3:                                      ; preds = %1
  br i1 undef, label %4, label %13

; <label>:4:                                      ; preds = %3
  %5 = getelementptr inbounds [16 x i32], ptr bitcast (<{ i32, i32, i32, i32, [12 x i32] }>* @_ZN15UsecaseSelector25AllowedImplDefinedFormatsE to ptr), i32 0, i32 undef
  %6 = load i32, ptr %5, align 4
  %7 = icmp eq i32 10, %6
  br i1 %7, label %9, label %8

; <label>:8:                                      ; preds = %4
  br i1 undef, label %9, label %12

; <label>:9:                                      ; preds = %8, %4
  br i1 undef, label %10, label %13

; <label>:10:                                     ; preds = %9
  br i1 undef, label %11, label %13

; <label>:11:                                     ; preds = %10
  br label %14

; <label>:12:                                     ; preds = %8
  br label %14

; <label>:13:                                     ; preds = %10, %9, %3
  br label %1

; <label>:14:                                     ; preds = %12, %11, %1
  ret void
}

attributes #1 = { noinline optnone }

; CHECK: _ZN15UsecaseSelector26IsAllowedImplDefinedFormatE15ChiBufferFormatj

