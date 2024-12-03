; RUN: llc -O3 -march=hexagon < %s | FileCheck %s

; We do not want the opt-addr-mode pass to modify the addi instructions whose
; base register has a predicated register definition
; CHECK: if ({{.*}}) [[REG1:r([0-9]+)]] = {{.*}}
; CHECK: r{{[0-9]+}} = add([[REG1]],#{{[0-9]+}})
; CHECK: r{{[0-9]+}} = add([[REG1]],#{{[0-9]+}})

@seqToUnseq = external dso_local local_unnamed_addr global [256 x i8], align 8
@unseqToSeq = external dso_local local_unnamed_addr global [256 x i8], align 8

define dso_local void @makeMaps() local_unnamed_addr {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc.3, %entry
  %0 = phi i32 [ 0, %entry ], [ %inc.7, %for.inc.3 ]
  %inc.1 = add nsw i32 %0, 1
  br i1 undef, label %for.inc.3, label %if.then.3

if.then.3:                                        ; preds = %for.body
  %arrayidx1.3 = getelementptr inbounds [256 x i8], [256 x i8]* @seqToUnseq, i32 0, i32 %inc.1
  store i8 undef, i8* %arrayidx1.3, align 1
  br label %for.inc.3

for.inc.3:                                        ; preds = %if.then.3, %for.body
  %1 = phi i32 [ %inc.1, %for.body ], [ 0, %if.then.3 ]
  %arrayidx3.4 = getelementptr inbounds [256 x i8], [256 x i8]* @unseqToSeq, i32 0, i32 undef
  store i8 0, i8* %arrayidx3.4, align 4
  %inc.4 = add nsw i32 %1, 1
  %conv2.7 = trunc i32 %inc.4 to i8
  %arrayidx3.7 = getelementptr inbounds [256 x i8], [256 x i8]* @unseqToSeq, i32 0, i32 undef
  store i8 %conv2.7, i8* %arrayidx3.7, align 1
  %inc.7 = add nsw i32 %inc.4, 1
  br label %for.body
}
