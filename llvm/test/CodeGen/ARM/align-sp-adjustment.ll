; RUN: llc -mtriple=thumbv7 -o - %s | FileCheck %s

; CHECK: [sp, #2120]

%struct.struct_2 = type { [172 x %struct.struct_1] }
%struct.struct_1 = type { i32, i32, i32 }

@.str = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"c\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"d\00", align 1

declare i32* @_Z4bar3iiPKcS0_i(i32, i32, i8*, i8*, i32)
declare void @_Z4bar1i8struct_2(i32, %struct.struct_2* byval align 4)
declare i32 @_Z4bar2PiPKc(i32*, i8*)

define void @_Z3fooiiiii(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5) #0 {
entry:
  %params = alloca %struct.struct_2, align 4
  %0 = bitcast %struct.struct_2* %params to i8*
  br label %for.body

for.body:
  %i.015 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %call = tail call i32* @_Z4bar3iiPKcS0_i(i32 %p1, i32 %p5, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i32 0, i32 0), i32 %i.015) #4
  %cmp1 = icmp eq i32* %call, null
  br i1 %cmp1, label %cleanup.8, label %for.inc

for.inc:
  %call2 = tail call i32 @_Z4bar2PiPKc(i32* %call, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i32 0, i32 0)) #4
  %f1 = getelementptr inbounds %struct.struct_2, %struct.struct_2* %params, i32 0, i32 0, i32 %i.015, i32 0
  store i32 %call2, i32* %f1, align 4
  %call3 = tail call i32 @_Z4bar2PiPKc(i32* %call, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.3, i32 0, i32 0)) #4
  %f2 = getelementptr inbounds %struct.struct_2, %struct.struct_2* %params, i32 0, i32 0, i32 %i.015, i32 1
  store i32 %call3, i32* %f2, align 4
  %inc = add nuw nsw i32 %i.015, 1
  %cmp = icmp slt i32 %inc, 4
  br i1 %cmp, label %for.body, label %for.end

for.end:
  call void @_Z4bar1i8struct_2(i32 %p4, %struct.struct_2* byval nonnull align 4 %params) #4
  br label %cleanup.8

cleanup.8:
  ret void
}

