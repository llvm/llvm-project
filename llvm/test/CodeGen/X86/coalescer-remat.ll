; RUN: llc < %s -mtriple=x86_64-apple-darwin | grep xor | count 3

@val = internal global i64 0
@"\01LC" = internal constant [7 x i8] c"0x%lx\0A\00"

define i32 @main() nounwind {
entry:
  %0 = cmpxchg i64* @val, i64 0, i64 1 monotonic
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr ([7 x i8]* @"\01LC", i32 0, i64 0), i64 %0) nounwind
  ret i32 0
}

declare i32 @printf(i8*, ...) nounwind
