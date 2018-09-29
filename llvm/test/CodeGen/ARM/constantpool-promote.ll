; RUN: llc -mtriple armv7--linux-gnueabihf -relocation-model=static -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7ARM,CHECK-STATIC
; RUN: llc -mtriple armv7--linux-gnueabihf -relocation-model=pic -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7ARM,CHECK-PIC
; RUN: llc -mtriple armv7--linux-gnueabihf -relocation-model=ropi -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7ARM
; RUN: llc -mtriple armv7--linux-gnueabihf -relocation-model=rwpi -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7ARM
; RUN: llc -mtriple thumbv7--linux-gnueabihf -relocation-model=static -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7THUMB
; RUN: llc -mtriple thumbv7--linux-gnueabihf -relocation-model=pic -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7THUMB
; RUN: llc -mtriple thumbv7--linux-gnueabihf -relocation-model=ropi -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7THUMB
; RUN: llc -mtriple thumbv7--linux-gnueabihf -relocation-model=rwpi -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V7,CHECK-V7THUMB
; RUN: llc -mtriple thumbv6m--linux-gnueabihf -relocation-model=static -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V6M
; RUN: llc -mtriple thumbv6m--linux-gnueabihf -relocation-model=pic -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V6M
; RUN: llc -mtriple thumbv6m--linux-gnueabihf -relocation-model=ropi -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V6M
; RUN: llc -mtriple thumbv6m--linux-gnueabihf -relocation-model=rwpi -arm-promote-constant < %s | FileCheck %s --check-prefixes=CHECK,CHECK-V6M

@.str = private unnamed_addr constant [2 x i8] c"s\00", align 1
@.str1 = private unnamed_addr constant [69 x i8] c"this string is far too long to fit in a literal pool by far and away\00", align 1
@.str2 = private unnamed_addr constant [27 x i8] c"this string is just right!\00", align 1
@.str3 = private unnamed_addr constant [26 x i8] c"this string is used twice\00", align 1
@.str4 = private unnamed_addr constant [29 x i8] c"same string in two functions\00", align 1
@.str5 = private unnamed_addr constant [2 x i8] c"s\00", align 1
@.arr1 = private unnamed_addr constant [2 x i16] [i16 3, i16 4], align 2
@.arr2 = private unnamed_addr constant [2 x i16] [i16 7, i16 8], align 2
@.arr3 = private unnamed_addr constant [2 x i16*] [i16* null, i16* null], align 4
@.ptr = private unnamed_addr constant [2 x i16*] [i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr2, i32 0, i32 0), i16* null], align 2
@.arr4 = private unnamed_addr constant [2 x i16] [i16 3, i16 4], align 16
@.arr5 = private unnamed_addr constant [2 x i16] [i16 3, i16 4], align 2
@.zerosize = private unnamed_addr constant [0 x i16] zeroinitializer, align 4
@implicit_alignment_vector = private unnamed_addr constant <4 x i32> <i32 1, i32 2, i32 3, i32 4>

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
; CHECK: L.arr1
define void @test6a() #0 {
  tail call void @c(i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr1, i32 0, i32 0)) #2
  ret void
}

; CHECK-LABEL: @test6b
; CHECK: L.arr1
define void @test6b() #0 {
  tail call void @c(i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr1, i32 0, i32 0)) #2
  ret void
}

; This shouldn't be promoted, as the string is used by another global.
; CHECK-LABEL: @test7
; CHECK-NOT: adr
define void @test7() #0 {
  tail call void @c(i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr2, i32 0, i32 0)) #2
  ret void  
}

; This can be promoted; it contains pointers, but they don't need relocations.
; CHECK-LABEL: @test8
; CHECK: .zero
; CHECK: .fnend
define void @test8() #0 {
  %a = load i16*, i16** getelementptr inbounds ([2 x i16*], [2 x i16*]* @.arr3, i32 0, i32 0)
  tail call void @c(i16* %a) #2
  ret void
}

; This can't be promoted in PIC mode because it contains pointers to other globals.
; CHECK-LABEL: @test8a
; CHECK-STATIC: .long .L.arr2
; CHECK-PIC: .long .L.ptr
; CHECK: .fnend
define void @test8a() #0 {
  %a = load i16*, i16** getelementptr inbounds ([2 x i16*], [2 x i16*]* @.ptr, i32 0, i32 0)
  tail call void @c(i16* %a) #2
  ret void
}

@fn1.a = private unnamed_addr constant [4 x i16] [i16 4, i16 0, i16 0, i16 0], align 2
@fn2.a = private unnamed_addr constant [8 x i8] [i8 4, i8 0, i8 0, i8 0, i8 23, i8 0, i8 6, i8 0], align 1

; Just check these don't crash.
define void @fn1() "target-features"="+strict-align"  {
entry:
  %a = alloca [4 x i16], align 2
  %0 = bitcast [4 x i16]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %0, i8* align 2 bitcast ([4 x i16]* @fn1.a to i8*), i32 8, i1 false)
  ret void
}

define void @fn2() "target-features"="+strict-align"  {
entry:
  %a = alloca [8 x i8], align 2
  %0 = bitcast [8 x i8]* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* bitcast ([8 x i8]* @fn2.a to i8*), i32 16, i1 false)
  ret void
}

; This shouldn't be promoted, as the global requires >4 byte alignment.
; CHECK-LABEL: @test9
; CHECK-NOT: adr
define void @test9() #0 {
  tail call void @c(i16* getelementptr inbounds ([2 x i16], [2 x i16]* @.arr4, i32 0, i32 0)) #2
  ret void
}

; Ensure that zero sized values are supported / not promoted.
; CHECK-LABEL: @pr32130
; CHECK-NOT: adr
define void @pr32130() #0 {
  tail call void @c(i16* getelementptr inbounds ([0 x i16], [0 x i16]* @.zerosize, i32 0, i32 0)) #2
  ret void
}

; CHECK-LABEL: @test10
; CHECK-V6M: adr r{{[0-9]*}}, [[x:.*]]
; CHECK-V6M: [[x]]:
; CHECK-V6M: .asciz "s\000\000"
; CHECK-V7: ldrb{{(.w)?}} r{{[0-9]*}}, [[x:.*]]
; CHECK-V7: [[x]]:
; CHECK-V7: .asciz "s\000\000"
define void @test10(i8* %a) local_unnamed_addr #0 {
  call void @llvm.memmove.p0i8.p0i8.i32(i8* align 1 %a, i8* align 1 getelementptr inbounds ([2 x i8], [2 x i8]* @.str5, i32 0, i32 0), i32 1, i1 false)
  ret void
}

; CHECK-LABEL: @test11
; CHECK-V6M: adr r{{[0-9]*}}, [[x:.*]]
; CHECK-V6M: [[x]]:
; CHECK-V6M: .short 3
; CHECK-V6M: .short 4
; CHECK-V7THUMB: ldrh{{(.w)?}} r{{[0-9]*}}, [[x:.*]]
; CHECK-V7THUMB: [[x]]:
; CHECK-V7THUMB: .short 3
; CHECK-V7THUMB: .short 4
; CHECK-V7ARM: adr r{{[0-9]*}}, [[x:.*]]
; CHECK-V7ARM: [[x]]:
; CHECK-V7ARM: .short 3
; CHECK-V7ARM: .short 4
define void @test11(i16* %a) local_unnamed_addr #0 {
  call void @llvm.memmove.p0i16.p0i16.i32(i16* align 2 %a, i16* align 2 getelementptr inbounds ([2 x i16], [2 x i16]* @.arr5, i32 0, i32 0), i32 2, i1 false)
  ret void
}

; Promotion only works with globals with alignment 4 or less; a vector has
; implicit alignment 16.
; CHECK-LABEL: @test12
; CHECK-NOT: adr
define void @test12() local_unnamed_addr #0 {
  call void @d(<4 x i32>* @implicit_alignment_vector)
  ret void
}


declare void @b(i8*) #1
declare void @c(i16*) #1
declare void @d(<4 x i32>*) #1
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i1) local_unnamed_addr
declare void @llvm.memmove.p0i16.p0i16.i32(i16*, i16*, i32, i1) local_unnamed_addr

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"min_enum_size", i32 4}
