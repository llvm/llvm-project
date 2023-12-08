; RUN: opt -S -passes=inline %s -o - | FileCheck %s
; RUN: opt -S -passes='cgscc(inline)' %s -o - | FileCheck %s
; RUN: opt -S -passes='module-inline' %s -o - | FileCheck %s

; Check that memcpy2 is completely inlined away.
; CHECK-NOT: memcpy2

@.str = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str1 = private unnamed_addr constant [3 x i8] c"ab\00", align 1

; Function Attrs: nounwind ssp uwtable
define i32 @main(i32 %argc, ptr nocapture readnone %argv) #0 {
entry:
  %data = alloca [2 x i8], align 1
  call fastcc void @memcpy2(ptr %data, ptr @.str, i64 1)
  call fastcc void @memcpy2(ptr %data, ptr @.str1, i64 2)
  ret i32 0
}

; Function Attrs: inlinehint nounwind ssp uwtable
define internal fastcc void @memcpy2(ptr nocapture %dst, ptr nocapture readonly %src, i64 %size) #1 {
entry:
  %vla = alloca i64, i64 %size, align 16
  call void @llvm.memcpy.p0.p0.i64(ptr %vla, ptr %src, i64 %size, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %vla, i64 %size, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #2

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { inlinehint nounwind ssp uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.5.0 (trunk 205695) (llvm/trunk 205706)"}
