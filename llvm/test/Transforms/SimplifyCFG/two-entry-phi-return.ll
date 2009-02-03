; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | not grep br

define i1 @qux(i8* %m, i8* %n, i8* %o, i8* %p) nounwind  {
entry:
        %tmp7 = icmp eq i8* %m, %n
        br i1 %tmp7, label %bb, label %UnifiedReturnBlock

bb:
        %tmp15 = icmp eq i8* %o, %p
        br label %UnifiedReturnBlock

UnifiedReturnBlock:
        %result = phi i1 [ 0, %entry ], [ %tmp15, %bb ]
        ret i1 %result
}
