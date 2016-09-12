; RUN: llc -relocation-model=static < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s
; RUN: llc -relocation-model=ropi < %s | FileCheck %s
; RUN: llc -relocation-model=rwpi < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "armv7--linux-gnueabihf"

@.str = private unnamed_addr constant [2 x i8] c"s\00", align 1
@.str1 = private unnamed_addr constant [69 x i8] c"this string is far too long to fit in a literal pool by far and away\00", align 1
@.str2 = private unnamed_addr constant [27 x i8] c"this string is just right!\00", align 1
@.str3 = private unnamed_addr constant [26 x i8] c"this string is used twice\00", align 1
@.str4 = private unnamed_addr constant [29 x i8] c"same string in two functions\00", align 1
@.arr1 = private unnamed_addr constant [2 x i16] [i16 3, i16 4], align 2

; CHECK-LABEL: @test1
; CHECK: adr r0, [[x:.*]]
; CHECK: [[x]]:
; CHECK: .asciz "s\000\000"
define void @test1() #0 {
  tail call void @a(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i32 0, i32 0)) #2
  ret void
}

declare void @a(i8*) #1

; CHECK-LABEL: @test2
; CHECK-NOT: .asci
; CHECK: .fnend
define void @test2() #0 {
  tail call void @a(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.str1, i32 0, i32 0)) #2
  ret void
}

; CHECK-LABEL: @test3
; CHECK: adr r0, [[x:.*]]
; CHECK: [[x]]:
; CHECK: .asciz "this string is just right!\000"
define void @test3() #0 {
  tail call void @a(i8* getelementptr inbounds ([27 x i8], [27 x i8]* @.str2, i32 0, i32 0)) #2
  ret void
}


; CHECK-LABEL: @test4
; CHECK: adr r{{.*}}, [[x:.*]]
; CHECK: [[x]]:
; CHECK: .asciz "this string is used twice\000\000"
define void @test4() #0 {
  tail call void @a(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str3, i32 0, i32 0)) #2
  tail call void @a(i8* getelementptr inbounds ([26 x i8], [26 x i8]* @.str3, i32 0, i32 0)) #2
  ret void
}

; CHECK-LABEL: @test5a
; CHECK-NOT: adr
define void @test5a() #0 {
  tail call void @a(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str4, i32 0, i32 0)) #2
  ret void
}

define void @test5b() #0 {
  tail call void @b(i8* getelementptr inbounds ([29 x i8], [29 x i8]* @.str4, i32 0, i32 0)) #2
  ret void
}

; CHECK-LABEL: @test6a
; CHECK: adr r0, [[x:.*]]
; CHECK: [[x]]:
; CHECK: .short 3
; CHECK: .short 4
define void @test6a() #0 {
  tail call void @c(i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr1, i32 0, i32 0)) #2
  ret void
}

; CHECK-LABEL: @test6b
; CHECK: adr r0, [[x:.*]]
; CHECK: [[x]]:
; CHECK: .short 3
; CHECK: .short 4
define void @test6b() #0 {
  tail call void @c(i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr1, i32 0, i32 0)) #2
  ret void
}

declare void @b(i8*) #1
declare void @c(i16*) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
!2 = !{!"Apple LLVM version 6.1.0 (clang-602.0.53) (based on LLVM 3.6.0svn)"}
