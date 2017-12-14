; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/comdat.ll -o %t2.o
; RUN: %gold -shared -o %t3.o -plugin %llvmshlibdir/LLVMgold%shlibext %t1.o %t2.o \
; RUN:  -m elf_x86_64 \
; RUN:  -plugin-opt=save-temps
; RUN: FileCheck --check-prefix=RES %s < %t3.o.resolution.txt
; RUN: llvm-readobj -t %t3.o | FileCheck --check-prefix=OBJ %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$c1 = comdat any

@v1 = weak_odr global i32 42, comdat($c1)
define weak_odr i32 @f1(i8*) comdat($c1) {
bb10:
  br label %bb11
bb11:
  ret i32 42
}

@r11 = global i32* @v1
@r12 = global i32 (i8*)* @f1

@a11 = alias i32, i32* @v1
@a12 = alias i16, bitcast (i32* @v1 to i16*)

@a13 = alias i32 (i8*), i32 (i8*)* @f1
@a14 = alias i16, bitcast (i32 (i8*)* @f1 to i16*)
@a15 = alias i16, i16* @a14

; gold's resolutions should tell us that our $c1 wins, and the other input's $c2
; wins. f1 is also local due to having protected visibility in the other object.

; RES: 1.o,f1,plx{{$}}
; RES: 1.o,v1,px{{$}}
; RES: 1.o,r11,px{{$}}
; RES: 1.o,r12,px{{$}}
; RES: 1.o,a11,px{{$}}
; RES: 1.o,a12,px{{$}}
; RES: 1.o,a13,px{{$}}
; RES: 1.o,a14,px{{$}}
; RES: 1.o,a15,px{{$}}

; RES: 2.o,f1,l{{$}}
; RES: 2.o,will_be_undefined,{{$}}
; RES: 2.o,v1,{{$}}
; RES: 2.o,r21,px{{$}}
; RES: 2.o,r22,px{{$}}
; RES: 2.o,a21,px{{$}}
; RES: 2.o,a22,px{{$}}
; RES: 2.o,a23,px{{$}}
; RES: 2.o,a24,px{{$}}
; RES: 2.o,a25,px{{$}}

; f1's protected visibility should be reflected in the DSO.

; OBJ:      Name: f1 (
; OBJ-NEXT: Value:
; OBJ-NEXT: Size:
; OBJ-NEXT: Binding:
; OBJ-NEXT: Type:
; OBJ-NEXT: Other [
; OBJ-NEXT:   STV_PROTECTED
; OBJ-NEXT: ]
