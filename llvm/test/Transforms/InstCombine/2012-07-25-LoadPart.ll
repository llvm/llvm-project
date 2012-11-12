; RUN: opt < %s -default-data-layout="e-p:32:32:32" -instcombine -S | FileCheck %s --check-prefix=LE
; RUN: opt < %s -default-data-layout="E-p:32:32:32" -instcombine -S | FileCheck %s --check-prefix=BE
; PR13442

@test = constant [4 x i32] [i32 1, i32 2, i32 3, i32 4]

define i64 @foo() {
  %ret = load i64* bitcast (i8* getelementptr (i8* bitcast ([4 x i32]* @test to i8*), i64 2) to i64*), align 1
  ret i64 %ret
  ; 0x00030000_00020000 in [01 00/00 00 02 00 00 00 03 00/00 00 04 00 00 00]
  ; LE: ret i64 844424930263040
  ; 0x00000200_00000300 in [00 00/00 01 00 00 00 02 00 00/00 03 00 00 00 04]
  ; BE: ret i64 281474976841728
}
