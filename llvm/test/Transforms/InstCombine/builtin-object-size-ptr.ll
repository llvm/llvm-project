; RUN: opt -instcombine -S < %s | FileCheck %s

; int foo() {
; struct V { char buf1[10];
;            int b;
;            char buf2[10];
;           } var;
;
;           char *p = &var.buf1[1];
;           return __builtin_object_size (p, 0);
; }

%struct.V = type { [10 x i8], i32, [10 x i8] }

define i32 @foo() #0 {
entry:
  %var = alloca %struct.V, align 4
  %0 = bitcast %struct.V* %var to i8*
  call void @llvm.lifetime.start.p0i8(i64 28, i8* %0) #3
  %buf1 = getelementptr inbounds %struct.V, %struct.V* %var, i32 0, i32 0
  %arrayidx = getelementptr inbounds [10 x i8], [10 x i8]* %buf1, i64 0, i64 1
  %1 = call i64 @llvm.objectsize.i64.p0i8(i8* %arrayidx, i1 false)
  %conv = trunc i64 %1 to i32
  call void @llvm.lifetime.end.p0i8(i64 28, i8* %0) #3
  ret i32 %conv
; CHECK: ret i32 27
; CHECK-NOT: ret i32 -1
}

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #1

declare i64 @llvm.objectsize.i64.p0i8(i8*, i1) #2

declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #1
