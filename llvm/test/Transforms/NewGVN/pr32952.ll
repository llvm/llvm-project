; PR32952: Don't erroneously consider congruent two phi nodes which
; have the same arguments but different incoming edges.
; RUN: opt -passes=newgvn -S %s | FileCheck %s

@a = common global i16 0, align 2
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

define i32 @tinkywinky() {
entry:
  %0 = load i16, i16* @a, align 2
  %conv = sext i16 %0 to i32
  %neg = xor i32 %conv, -1
  %conv1 = trunc i32 %neg to i16
  %conv3 = zext i16 %conv1 to i32
  %cmp = icmp slt i32 %conv, %conv3
  br i1 %cmp, label %tinky, label %winky

tinky:
  store i16 2, i16* @a, align 2
  br label %patatino

winky:
  br label %patatino

patatino:
; CHECK: %meh = phi i16 [ %0, %winky ], [ %conv1, %tinky ]
; CHECK: %banana = phi i16 [ %0, %tinky ], [ %conv1, %winky ]
  %meh = phi i16 [ %0, %winky ], [ %conv1, %tinky ]
  %banana = phi i16 [ %0, %tinky ], [ %conv1, %winky ]
  br label %end

end:
; CHECK: %promoted = zext i16 %banana to i32
; CHECK: %other = zext i16 %meh to i32
  %promoted = zext i16 %banana to i32
  %other = zext i16 %meh to i32
  %first = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %promoted)
  %second = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %other)
  ret i32 0
}

declare i32 @printf(i8*, ...)
