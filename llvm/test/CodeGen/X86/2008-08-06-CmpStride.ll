; RUN: llc -march=x86-64 < %s -o - | grep "cmpl	\$[1], %"

@.str = internal constant [4 x i8] c"%d\0A\00"

declare i32 @printf(i8* noalias , ...) nounwind

define i32 @main() nounwind {
entry:
        br label %forbody

forbody:
        %i.0 = phi i32 [ 0, %entry ], [ %inc, %forbody ]                ; <i32>[#uses=3]
        %sub14 = sub i32 1027, %i.0             ; <i32> [#uses=1]
        %mul15 = mul i32 %sub14, 10             ; <i32> [#uses=1]
        %add166 = or i32 %mul15, 1              ; <i32> [#uses=1] *
        call i32 (i8*, ...) @printf( i8* noalias  getelementptr ([4 x i8], [4 x i8]* @.str, i32 0, i32 0), i32 %add166 ) nounwind
        %inc = add i32 %i.0, 1          ; <i32> [#uses=3]
        %cmp = icmp ne i32 %inc, 1027          ; <i1> [#uses=1]
        br i1 %cmp, label %forbody, label %afterfor

afterfor:               ; preds = %forcond
        ret i32 0
}
