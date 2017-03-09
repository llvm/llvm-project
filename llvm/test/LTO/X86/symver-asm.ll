; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -exported-symbol=io_cancel_0_4 -exported-symbol=io_cancel_weak_0_4 -exported-symbol=foo -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s
; RUN: llvm-lto2 -r %t1,io_cancel_0_4,plx -r %t1,io_cancel_0_4,plx -r %t1,io_cancel_local_0_4,plx -r %t1,io_cancel_weak_0_4,plx -r %t1,io_cancel_weak_0_4,plx -r %t1,io_cancel@@LIBAIO_0.4,plx -r %t1,io_cancel_weak@@LIBAIO_0.4,plx -r %t1,io_cancel_weak@@LIBAIO_0.4.1,plx -r %t1,foo,plx -r %t1,foo,plx -r %t1,foo@@VER1,plx -o %t3 %t1 -save-temps
; RUN: llvm-nm %t3.0 | FileCheck %s
; RUN: llvm-dis %t3.0.2.internalize.bc -o - | FileCheck %s --check-prefix=INTERN

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".symver io_cancel_0_4,io_cancel@@LIBAIO_0.4"
module asm ".symver io_cancel_local_0_4,io_cancel_local@@LIBAIO_0.4"
module asm ".symver io_cancel_weak_0_4,io_cancel_weak@@LIBAIO_0.4"
; Ensure we handle case of same aliasee with two version aliases.
module asm ".symver io_cancel_weak_0_4,io_cancel_weak@@LIBAIO_0.4.1"
module asm ".symver foo,foo@@VER1"

; Local values used in inline assembly must be specified on the
; llvm.compiler.used so they aren't incorrectly DCE'd during module linking.
@llvm.compiler.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @io_cancel_local_0_4 to i8*)], section "llvm.metadata"

define i32 @io_cancel_0_4() {
; CHECK-DAG: T io_cancel@@LIBAIO_0.4
; CHECK-DAG: T io_cancel_0_4
  ret i32 0
}

define internal i32 @io_cancel_local_0_4() {
; INTERN: llvm.compiler.used {{.*}} @io_cancel_local_0_4
; INTERN: define internal i32 @io_cancel_local_0_4()
; CHECK-DAG: t io_cancel_local@@LIBAIO_0.4
; CHECK-DAG: t io_cancel_local_0_4
  ret i32 0
}

define weak i32 @io_cancel_weak_0_4() {
; CHECK-DAG: W io_cancel_weak@@LIBAIO_0.4
; CHECK-DAG: W io_cancel_weak@@LIBAIO_0.4.1
; CHECK-DAG: W io_cancel_weak_0_4
ret i32 0
}

define i32 @"\01foo"() {
; CHECK-DAG: T foo@@VER1
; CHECK-DAG: T foo
  ret i32 0
}
