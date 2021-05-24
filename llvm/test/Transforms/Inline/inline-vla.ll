; RUN: opt -S -inline %s -o - | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' %s -o - | FileCheck %s

; Check that memcpy2 is completely inlined away.
; CHECK-NOT: memcpy2

@.str = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str1 = private unnamed_addr constant [3 x i8] c"ab\00", align 1

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, i8** nocapture readnone %argv) #0 {
entry:
  %data = alloca [2 x i8], align 1
  %arraydecay = getelementptr inbounds [2 x i8], [2 x i8]* %data, i64 0, i64 0
  call fastcc void @memcpy2(i8* %arraydecay, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i64 0, i64 0), i64 1)
  call fastcc void @memcpy2(i8* %arraydecay, i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str1, i64 0, i64 0), i64 2)
  ret i32 0
}

; Function Attrs: inlinehint nounwind ssp uwtable
define internal fastcc void @memcpy2(i8* nocapture %dst, i8* nocapture readonly %src, i64 %size) #1 {
entry:
  %vla = alloca i64, i64 %size, align 16
  %0 = bitcast i64* %vla to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %src, i64 %size, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %0, i64 %size, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1) #2

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 (trunk 205695) (llvm/trunk 205706)"}
