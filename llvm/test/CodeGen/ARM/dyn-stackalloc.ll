; RUN: llc -mcpu=generic -mtriple=arm-eabi -verify-machineinstrs < %s | FileCheck %s

%struct.comment = type { ptr, ptr, i32, ptr }
%struct.info = type { i32, i32, i32, i32, i32, i32, i32, ptr }
%struct.state = type { i32, ptr, ptr, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, ptr }

@str215 = external global [2 x i8]

define void @t1(ptr %v) {

; Make sure we generate:
;   sub	sp, sp, r1
; instead of:
;   sub	r1, sp, r1
;   mov	sp, r1

; CHECK-LABEL: @t1
; CHECK: bic [[REG1:r[0-9]+]],
; CHECK-NOT: sub r{{[0-9]+}}, sp, [[REG1]]
; CHECK: sub sp, sp, [[REG1]]

  %tmp6 = load i32, ptr null
  %tmp8 = alloca float, i32 %tmp6
  store i32 1, ptr null
  br i1 false, label %bb123.preheader, label %return

bb123.preheader:                                  ; preds = %0
  br i1 false, label %bb43, label %return

bb43:                                             ; preds = %bb123.preheader
  call fastcc void @f1(ptr %tmp8, ptr null, i32 0)
  %tmp70 = load i32, ptr null
  call fastcc void @f2(ptr null, ptr null, ptr %tmp8, i32 %tmp70)
  ret void

return:                                           ; preds = %bb123.preheader, %0
  ret void
}

declare fastcc void @f1(ptr, ptr, i32)

declare fastcc void @f2(ptr, ptr, ptr, i32)

define void @t2(ptr %vc, ptr %tag, ptr %contents) {
  %tmp1 = call i32 @strlen(ptr %tag)
  %tmp3 = call i32 @strlen(ptr %contents)
  %tmp4 = add i32 %tmp1, 2
  %tmp5 = add i32 %tmp4, %tmp3
  %tmp6 = alloca i8, i32 %tmp5
  %tmp9 = call ptr @strcpy(ptr %tmp6, ptr %tag)
  %tmp6.len = call i32 @strlen(ptr %tmp6)
  %tmp6.indexed = getelementptr i8, ptr %tmp6, i32 %tmp6.len
  call void @llvm.memcpy.p0.p0.i32(ptr align 1 %tmp6.indexed, ptr align 1 @str215, i32 2, i1 false)
  %tmp15 = call ptr @strcat(ptr %tmp6, ptr %contents)
  call fastcc void @comment_add(ptr %vc, ptr %tmp6)
  ret void
}

declare i32 @strlen(ptr)

declare ptr @strcat(ptr, ptr)

declare fastcc void @comment_add(ptr, ptr)

declare ptr @strcpy(ptr, ptr)

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture, ptr nocapture, i32, i1) nounwind
