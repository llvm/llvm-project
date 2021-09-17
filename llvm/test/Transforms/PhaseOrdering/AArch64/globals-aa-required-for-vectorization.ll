; RUN: opt -passes='lto<O3>' -S %s | FileCheck %s

target triple = "arm64e-apple-darwin"

@A = external unnamed_addr global i8**, align 8
@B = external unnamed_addr global i8**, align 8
@C = internal unnamed_addr global i32 0, align 4
@D = external unnamed_addr global i32, align 4

; CHECK-LABEL: @fn
; CHECK: vector.body:
;
define void @fn() {
entry:
  %v.D = load i32, i32* @D, align 4
  store i32 %v.D, i32* @C, align 4
  call void @clobber()

  %v.B = load i8**, i8*** @B, align 8
  %v.A = load i8**, i8*** @A, align 8
  %gep.1 = getelementptr inbounds i8*, i8** %v.A, i64 0
  %v.gep.1 = load i8*, i8** %gep.1, align 8
  %gep.2 = getelementptr inbounds i8*, i8** %v.B, i64 0
  %v.gep.2 = load i8*, i8** %gep.2, align 8
  %cmp = icmp eq i8* %v.gep.2, null
  br i1 %cmp, label %exit, label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.3 = getelementptr inbounds i8, i8* %v.gep.2, i32 %iv
  %v.gep.3 = load i8, i8* %gep.3, align 1
  %gep.4 = getelementptr inbounds i8, i8* %v.gep.1, i32 %iv
  store i8 %v.gep.3, i8* %gep.4, align 1
  %iv.next = add nuw nsw i32 %iv, 1
  %v.C = load i32, i32* @C, align 4
  %exit.cond = icmp sgt i32 %iv, %v.C
  br i1 %exit.cond, label %exit, label %loop

exit:
  ret void
}

declare void @clobber()
