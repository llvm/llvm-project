; RUN: opt -passes=verify < %s

%struct.__sFILE = type { %struct.__sFILE }

@.str = private unnamed_addr constant [13 x i8] c"Hello world\0A\00", align 1

; Function Attrs: nounwind ssp
define void @test(%struct.__sFILE* %stream, i8* %str) {
  %fputs = call i32 @fputs(i8* %str, %struct.__sFILE* %stream)
  ret void
}

; Function Attrs: nounwind
declare i32 @fputs(i8* nocapture, %struct.__sFILE* nocapture)

