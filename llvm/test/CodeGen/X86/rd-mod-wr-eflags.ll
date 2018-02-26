; RUN: llc < %s -mtriple=x86_64-- | FileCheck %s

%struct.obj = type { i64 }

; CHECK: _Z7releaseP3obj
define void @_Z7releaseP3obj(%struct.obj* nocapture %o) nounwind uwtable ssp {
entry:
; CHECK: decq	(%{{rdi|rcx}})
; CHECK-NEXT: je
  %refcnt = getelementptr inbounds %struct.obj, %struct.obj* %o, i64 0, i32 0
  %0 = load i64, i64* %refcnt, align 8
  %dec = add i64 %0, -1
  store i64 %dec, i64* %refcnt, align 8
  %tobool = icmp eq i64 %dec, 0
  br i1 %tobool, label %if.end, label %return

if.end:                                           ; preds = %entry
  %1 = bitcast %struct.obj* %o to i8*
  tail call void @free(i8* %1)
  br label %return

return:                                           ; preds = %entry, %if.end
  ret void
}

@c = common global i64 0, align 8
@a = common global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"%ld\0A\00", align 1
@b = common global i32 0, align 4

; CHECK: test
define i32 @test() nounwind uwtable ssp {
entry:
; CHECK: decq
; CHECK-NOT: decq
%0 = load i64, i64* @c, align 8
%dec.i = add nsw i64 %0, -1
store i64 %dec.i, i64* @c, align 8
%tobool.i = icmp ne i64 %dec.i, 0
%lor.ext.i = zext i1 %tobool.i to i32
store i32 %lor.ext.i, i32* @a, align 4
%call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i64 %dec.i) nounwind
ret i32 0
}

; CHECK: test2
define i32 @test2() nounwind uwtable ssp {
entry:
; CHECK-NOT: decq ({{.*}})
%0 = load i64, i64* @c, align 8
%dec.i = add nsw i64 %0, -1
store i64 %dec.i, i64* @c, align 8
%tobool.i = icmp ne i64 %0, 0
%lor.ext.i = zext i1 %tobool.i to i32
store i32 %lor.ext.i, i32* @a, align 4
%call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str, i64 0, i64 0), i64 %dec.i) nounwind
ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @free(i8* nocapture) nounwind

%struct.obj2 = type { i64, i32, i16, i8 }

declare void @other(%struct.obj2* ) nounwind;

; CHECK: example_dec
define void @example_dec(%struct.obj2* %o) nounwind uwtable ssp {
; 64 bit dec
entry:
  %s64 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 0
; CHECK-NOT: load 
  %0 = load i64, i64* %s64, align 8
; CHECK: decq ({{.*}})
  %dec = add i64 %0, -1
  store i64 %dec, i64* %s64, align 8
  %tobool = icmp eq i64 %dec, 0
  br i1 %tobool, label %if.end, label %return

; 32 bit dec
if.end:
  %s32 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 1
; CHECK-NOT: load 
  %1 = load i32, i32* %s32, align 4
; CHECK: decl {{[0-9][0-9]*}}({{.*}})
  %dec1 = add i32 %1, -1
  store i32 %dec1, i32* %s32, align 4
  %tobool2 = icmp eq i32 %dec1, 0
  br i1 %tobool2, label %if.end1, label %return

; 16 bit dec
if.end1:
  %s16 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 2
; CHECK-NOT: load 
  %2 = load i16, i16* %s16, align 2
; CHECK: decw {{[0-9][0-9]*}}({{.*}})
  %dec2 = add i16 %2, -1
  store i16 %dec2, i16* %s16, align 2
  %tobool3 = icmp eq i16 %dec2, 0
  br i1 %tobool3, label %if.end2, label %return

; 8 bit dec
if.end2:
  %s8 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 3
; CHECK-NOT: load 
  %3 = load i8, i8* %s8
; CHECK: decb {{[0-9][0-9]*}}({{.*}})
  %dec3 = add i8 %3, -1
  store i8 %dec3, i8* %s8
  %tobool4 = icmp eq i8 %dec3, 0
  br i1 %tobool4, label %if.end4, label %return

if.end4:
  tail call void @other(%struct.obj2* %o) nounwind
  br label %return

return:                                           ; preds = %if.end4, %if.end, %entry                                                                                                                                                                               
  ret void
}

; CHECK: example_inc
define void @example_inc(%struct.obj2* %o) nounwind uwtable ssp {
; 64 bit inc
entry:
  %s64 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 0
; CHECK-NOT: load 
  %0 = load i64, i64* %s64, align 8
; CHECK: incq ({{.*}})
  %inc = add i64 %0, 1
  store i64 %inc, i64* %s64, align 8
  %tobool = icmp eq i64 %inc, 0
  br i1 %tobool, label %if.end, label %return

; 32 bit inc
if.end:
  %s32 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 1
; CHECK-NOT: load 
  %1 = load i32, i32* %s32, align 4
; CHECK: incl {{[0-9][0-9]*}}({{.*}})
  %inc1 = add i32 %1, 1
  store i32 %inc1, i32* %s32, align 4
  %tobool2 = icmp eq i32 %inc1, 0
  br i1 %tobool2, label %if.end1, label %return

; 16 bit inc
if.end1:
  %s16 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 2
; CHECK-NOT: load 
  %2 = load i16, i16* %s16, align 2
; CHECK: incw {{[0-9][0-9]*}}({{.*}})
  %inc2 = add i16 %2, 1
  store i16 %inc2, i16* %s16, align 2
  %tobool3 = icmp eq i16 %inc2, 0
  br i1 %tobool3, label %if.end2, label %return

; 8 bit inc
if.end2:
  %s8 = getelementptr inbounds %struct.obj2, %struct.obj2* %o, i64 0, i32 3
; CHECK-NOT: load 
  %3 = load i8, i8* %s8
; CHECK: incb {{[0-9][0-9]*}}({{.*}})
  %inc3 = add i8 %3, 1
  store i8 %inc3, i8* %s8
  %tobool4 = icmp eq i8 %inc3, 0
  br i1 %tobool4, label %if.end4, label %return

if.end4:
  tail call void @other(%struct.obj2* %o) nounwind
  br label %return

return:
  ret void
}

; Deal with TokenFactor chain
; rdar://11236106
@foo = external global i64*, align 8

define void @test3() nounwind ssp {
entry:
; CHECK-LABEL: test3:
; CHECK: decq 16(%rax)
  %0 = load i64*, i64** @foo, align 8
  %arrayidx = getelementptr inbounds i64, i64* %0, i64 2
  %1 = load i64, i64* %arrayidx, align 8
  %dec = add i64 %1, -1
  store i64 %dec, i64* %arrayidx, align 8
  %cmp = icmp eq i64 %dec, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @baz() nounwind
  br label %if.end

if.end:
  ret void
}

declare void @baz()

; Avoid creating a cycle in the DAG which would trigger an assert in the
; scheduler.
; PR12565
; rdar://11451474
@x = external global i32, align 4
@y = external global i32, align 4
@z = external global i32, align 4

define void @test4() nounwind uwtable ssp {
entry:
  %0 = load i32, i32* @x, align 4
  %1 = load i32, i32* @y, align 4
  %dec = add nsw i32 %1, -1
  store i32 %dec, i32* @y, align 4
  %tobool.i = icmp ne i32 %dec, 0
  %cond.i = select i1 %tobool.i, i32 %0, i32 0
  store i32 %cond.i, i32* @z, align 4
  ret void
}
