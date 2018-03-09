; RUN: opt -S -mtriple=x86_64-pc-linux-gnu -mcpu=generic -slp-vectorizer -pass-remarks-output=%t < %s | FileCheck %s
; RUN: FileCheck --input-file=%t --check-prefix=YAML %s

define i32 @foo(i32* nocapture readonly %diff) #0 {
entry:
  %m2 = alloca [8 x [8 x i32]], align 16
  %0 = bitcast [8 x [8 x i32]]* %m2 to i8*
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %a.088 = phi i32 [ 0, %entry ], [ %add24, %for.body ]
  %1 = shl i64 %indvars.iv, 3
  %arrayidx = getelementptr inbounds i32, i32* %diff, i64 %1
  %2 = load i32, i32* %arrayidx, align 4
  %3 = or i64 %1, 4
  %arrayidx2 = getelementptr inbounds i32, i32* %diff, i64 %3
  %4 = load i32, i32* %arrayidx2, align 4
  %add3 = add nsw i32 %4, %2
  %arrayidx6 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 %indvars.iv, i64 0
  store i32 %add3, i32* %arrayidx6, align 16
  %add10 = add nsw i32 %add3, %a.088
  %5 = or i64 %1, 1
  %arrayidx13 = getelementptr inbounds i32, i32* %diff, i64 %5
  %6 = load i32, i32* %arrayidx13, align 4
  %7 = or i64 %1, 5
  %arrayidx16 = getelementptr inbounds i32, i32* %diff, i64 %7
  %8 = load i32, i32* %arrayidx16, align 4
  %add17 = add nsw i32 %8, %6
  %arrayidx20 = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 %indvars.iv, i64 1
  store i32 %add17, i32* %arrayidx20, align 4
  %add24 = add nsw i32 %add10, %add17

 ; CHECK-NOT: add nsw <{{[0-9]+}} x i32> 
 ; YAML:      Pass:            slp-vectorizer
 ; YAML-NEXT: Name:            InequableTypes
 ; YAML-NEXT: Function:        foo
 ; YAML-NEXT: Args:
 ; YAML-NEXT:   - String:          'Cannot SLP vectorize list: not all of the '
 ; YAML-NEXT:   - String:          'parts of scalar instructions are of the same type: '
 ; YAML-NEXT:   - Instruction1Opcode: add
 ; YAML-NEXT:   - String:          ' and '
 ; YAML-NEXT:   - Instruction2Opcode: phi

 ; YAML:      Pass:            slp-vectorizer
 ; YAML-NEXT: Name:            NotPossible
 ; YAML-NEXT: Function:        foo
 ; YAML-NEXT: Args:
 ; YAML-NEXT:   - String:          'Cannot SLP vectorize list: vectorization was impossible'
 ; YAML-NEXT:   - String:          ' with available vectorization factors'

  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  %arraydecay = getelementptr inbounds [8 x [8 x i32]], [8 x [8 x i32]]* %m2, i64 0, i64 0
  ret i32 %add24
}

