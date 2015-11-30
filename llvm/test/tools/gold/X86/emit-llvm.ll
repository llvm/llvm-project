; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    --plugin-opt=generate-api-file \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck %s
; RUN: FileCheck --check-prefix=API %s < %T/../apifile.txt

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     -m elf_x86_64 --plugin-opt=save-temps \
; RUN:    -shared %t.o -o %t3.o
; RUN: llvm-dis %t3.o.bc -o - | FileCheck %s
; RUN: llvm-dis %t3.o.opt.bc -o - | FileCheck --check-prefix=OPT %s
; RUN: llvm-dis %t3.o.opt.bc -o - | FileCheck --check-prefix=OPT2 %s
; RUN: llvm-nm %t3.o.o | FileCheck --check-prefix=NM %s

; RUN: rm -f %t4.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:     -m elf_x86_64 --plugin-opt=disable-output \
; RUN:    -shared %t.o -o %t4.o
; RUN: not test -a %t4.o

; NM: T f3

target triple = "x86_64-unknown-linux-gnu"

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
define linkonce_odr void @f4() {
  ret void
}

; CHECK-DAG: define linkonce_odr void @f5()
; OPT-DAG: define linkonce_odr void @f5()
define linkonce_odr void @f5() {
  ret void
}
@g5 = global void()* @f5

; CHECK-DAG: define internal void @f6() unnamed_addr
; OPT-DAG: define internal void @f6() unnamed_addr
define linkonce_odr void @f6() unnamed_addr {
  ret void
}
@g6 = global void()* @f6

define i32* @f7() {
  ret i32* @g7
}

define i32* @f8() {
  ret i32* @g8
}

; API: f1 PREVAILING_DEF_IRONLY
; API: f2 PREVAILING_DEF_IRONLY
; API: f3 PREVAILING_DEF_IRONLY_EXP
; API: f4 PREVAILING_DEF_IRONLY_EXP
; API: f5 PREVAILING_DEF_IRONLY_EXP
; API: f6 PREVAILING_DEF_IRONLY_EXP
; API: f7 PREVAILING_DEF_IRONLY_EXP
; API: f8 PREVAILING_DEF_IRONLY_EXP
; API: g7 UNDEF
; API: g8 UNDEF
; API: g5 PREVAILING_DEF_IRONLY_EXP
; API: g6 PREVAILING_DEF_IRONLY_EXP
