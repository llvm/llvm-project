; RUN: opt -passes=verify < %s

%struct.__sFILE = type { %struct.__sFILE }

@.str = private unnamed_addr constant [13 x i8] c"Hello world\0A\00", align 1

; Function Attrs: nounwind ssp
define void @test(ptr %stream, ptr %str) {
  %fputs = call i32 @fputs(ptr %str, ptr %stream)
  ret void
}

; Function Attrs: nounwind
declare i32 @fputs(ptr nocapture, ptr nocapture)

