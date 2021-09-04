; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link %t/1.ll -S -o - | FileCheck %s --check-prefix=CHECK1
;; The source doesn't have comdat.
; RUN: llvm-link %t/2.ll -S -o - | FileCheck %s --check-prefix=CHECK2

;; The leader is an alias.
; RUN: llvm-link %t/3.ll %t/3-aux.ll -S -o - | FileCheck %s
; RUN: llvm-link %t/3-aux.ll %t/3.ll -S -o - | FileCheck %s

;--- 1.ll
$c = comdat any
@a = alias void (), void ()* @f
define internal void @f() comdat($c) {
  ret void
}

; CHECK1-DAG: $c = comdat any
; CHECK1-DAG: @a = alias void (), void ()* @f
; CHECK1-DAG: define internal void @f() comdat($c)

$f2 = comdat largest
define linkonce_odr void @f2() comdat($f2) {
  ret void
}
define void @f3() comdat($f2) {
  ret void
}

; CHECK1-DAG: $f2 = comdat largest
; CHECK1-DAG: define linkonce_odr void @f2()

;--- 2.ll
$c = comdat largest

; CHECK2: @c = global i32 0, comdat
@c = global i32 0, comdat

;--- 3.ll
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

$foo = comdat largest
@foo = linkonce_odr unnamed_addr constant [1 x i8*] [i8* bitcast (void ()* @bar to i8*)], comdat($foo)

; CHECK: @foo = alias i8*, getelementptr inbounds ([2 x i8*], [2 x i8*]* @some_name, i32 0, i32 1)

declare void @bar() unnamed_addr

;--- 3-aux.ll
target datalayout = "e-m:w-p:32:32-i64:64-f80:32-n8:16:32-S32"

$foo = comdat largest

@zed = external constant i8
@some_name = private unnamed_addr constant [2 x i8*] [i8* @zed, i8* bitcast (void ()* @bar to i8*)], comdat($foo)
@foo = alias i8*, getelementptr([2 x i8*], [2 x i8*]* @some_name, i32 0, i32 1)

declare void @bar() unnamed_addr
