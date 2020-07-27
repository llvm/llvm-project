; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     -m elf_x86_64 --plugin-opt=save-temps \
; RUN:    -shared %t.o -o %t3.o
; RUN: FileCheck --check-prefix=RES %s < %t3.o.resolution.txt
; RUN: llvm-dis %t3.o.0.2.internalize.bc -o - | FileCheck %s
; RUN: llvm-dis %t3.o.0.4.opt.bc -o - | FileCheck --check-prefix=OPT %s
; RUN: llvm-dis %t3.o.0.4.opt.bc -o - | FileCheck --check-prefix=OPT2 %s
; RUN: llvm-nm %t3.o.lto.o | FileCheck --check-prefix=NM %s

; RUN: rm -f %t4.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:     -m elf_x86_64 --plugin-opt=disable-output \
; RUN:    -shared %t.o -o %t4.o
; RUN: not test -a %t4.o

; NM: T f3

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-DAG: @g1 = weak_odr constant i32 32
@g1 = linkonce_odr constant i32 32

; CHECK-DAG: @g2 = internal constant i32 32
@g2 = linkonce_odr local_unnamed_addr constant i32 32

; CHECK-DAG: @g3 = internal unnamed_addr constant i32 32
@g3 = linkonce_odr unnamed_addr constant i32 32

; CHECK-DAG: @g4 = weak_odr global i32 32
@g4 = linkonce_odr global i32 32

; CHECK-DAG: @g5 = weak_odr global i32 32
@g5 = linkonce_odr local_unnamed_addr global i32 32

; CHECK-DAG: @g6 = internal unnamed_addr global i32 32
@g6 = linkonce_odr unnamed_addr global i32 32

@g7 = extern_weak global i32
; CHECK-DAG: @g7 = extern_weak global i32

@g8 = external global i32

; CHECK-DAG: define internal void @f1()
; OPT2-NOT: @f1
define hidden void @f1() {
  ret void
}

; CHECK-DAG: define hidden void @f2()
; OPT-DAG: define hidden void @f2()
define hidden void @f2() {
  ret void
}

@llvm.used = appending global [1 x i8*] [ i8* bitcast (void ()* @f2 to i8*)]

; CHECK-DAG: define void @f3()
; OPT-DAG: define void @f3()
define void @f3() {
  call void @f4()
  ret void
}

; CHECK-DAG: define internal void @f4()
; OPT2-NOT: @f4
define linkonce_odr void @f4() local_unnamed_addr {
  ret void
}

; CHECK-DAG: define weak_odr void @f5()
; OPT-DAG: define weak_odr void @f5()
define linkonce_odr void @f5() {
  ret void
}
@g9 = global void()* @f5

; CHECK-DAG: define internal void @f6() unnamed_addr
; OPT-DAG: define internal void @f6() unnamed_addr
define linkonce_odr void @f6() unnamed_addr {
  ret void
}
@g10 = global void()* @f6

define i32* @f7() {
  ret i32* @g7
}

define i32* @f8() {
  ret i32* @g8
}

; RES: .o,f1,pl{{$}}
; RES: .o,f2,pl{{$}}
; RES: .o,f3,px{{$}}
; RES: .o,f4,p{{$}}
; RES: .o,f5,px{{$}}
; RES: .o,f6,p{{$}}
; RES: .o,f7,px{{$}}
; RES: .o,f8,px{{$}}
; RES: .o,g1,px{{$}}
; RES: .o,g2,p{{$}}
; RES: .o,g3,p{{$}}
; RES: .o,g4,px{{$}}
; RES: .o,g5,px{{$}}
; RES: .o,g6,p{{$}}
; RES: .o,g7,{{$}}
; RES: .o,g8,{{$}}
; RES: .o,g9,px{{$}}
; RES: .o,g10,px{{$}}
