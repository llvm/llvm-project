; RUN: llc -mcpu=generic -mtriple=arm-eabi -verify-machineinstrs < %s | FileCheck %s

%struct.comment = type { i8**, i32*, i32, i8* }
%struct.info = type { i32, i32, i32, i32, i32, i32, i32, i8* }
%struct.state = type { i32, %struct.info*, float**, i32, i32, i32, i32, i32, i32, i32, i32, i32, i64, i64, i64, i64, i64, i64, i8* }

@str215 = external global [2 x i8]

define void @t1(%struct.state* %v) {

; Make sure we generate:
;   sub	sp, sp, r1
; instead of:
;   sub	r1, sp, r1
;   mov	sp, r1

; CHECK-LABEL: @t1
; CHECK: bic [[REG1:r[0-9]+]],
; CHECK-NOT: sub r{{[0-9]+}}, sp, [[REG1]]
; CHECK: sub sp, sp, [[REG1]]

  %tmp6 = load i32, i32* null
  %tmp8 = alloca float, i32 %tmp6
  store i32 1, i32* null
  br i1 false, label %bb123.preheader, label %return

bb123.preheader:                                  ; preds = %0
  br i1 false, label %bb43, label %return

bb43:                                             ; preds = %bb123.preheader
  call fastcc void @f1(float* %tmp8, float* null, i32 0)
  %tmp70 = load i32, i32* null
  %tmp85 = getelementptr float, float* %tmp8, i32 0
  call fastcc void @f2(float* null, float* null, float* %tmp85, i32 %tmp70)
  ret void

return:                                           ; preds = %bb123.preheader, %0
  ret void
}

declare fastcc void @f1(float*, float*, i32)

declare fastcc void @f2(float*, float*, float*, i32)

define void @t2(%struct.comment* %vc, i8* %tag, i8* %contents) {
  %tmp1 = call i32 @strlen(i8* %tag)
  %tmp3 = call i32 @strlen(i8* %contents)
  %tmp4 = add i32 %tmp1, 2
  %tmp5 = add i32 %tmp4, %tmp3
  %tmp6 = alloca i8, i32 %tmp5
  %tmp9 = call i8* @strcpy(i8* %tmp6, i8* %tag)
  %tmp6.len = call i32 @strlen(i8* %tmp6)
  %tmp6.indexed = getelementptr i8, i8* %tmp6, i32 %tmp6.len
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 1 %tmp6.indexed, i8* align 1 getelementptr inbounds ([2 x i8], [2 x i8]* @str215, i32 0, i32 0), i32 2, i1 false)
  %tmp15 = call i8* @strcat(i8* %tmp6, i8* %contents)
  call fastcc void @comment_add(%struct.comment* %vc, i8* %tmp6)
  ret void
}

declare i32 @strlen(i8*)

declare i8* @strcat(i8*, i8*)

declare fastcc void @comment_add(%struct.comment*, i8*)

declare i8* @strcpy(i8*, i8*)

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
