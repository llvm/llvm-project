; Test for a bug specific to the new pass manager where we may build a domtree
; to make more precise AA queries for functions.
;
; RUN: opt -passes='no-op-module' -debug-pass-manager -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { %struct.widget }
%struct.widget = type { ptr }

; M0: @global = local_unnamed_addr global
; M1-NOT: @global
@global = local_unnamed_addr global %struct.hoge { %struct.widget { ptr getelementptr inbounds inrange(-16, 8) ({ [3 x ptr] }, ptr @global.1, i32 0, i32 0, i32 2) } }, align 8

; M0: @global.1 = external unnamed_addr constant
; M1: @global.1 = linkonce_odr unnamed_addr constant
@global.1 = linkonce_odr unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr @global.4, ptr @quux] }, align 8, !type !0

; M0: @global.2 = external global
; M1-NOT: @global.2
@global.2 = external global ptr

; M0: @global.3 = linkonce_odr constant
; M1-NOT: @global.3
@global.3 = linkonce_odr constant [22 x i8] c"zzzzzzzzzzzzzzzzzzzzz\00"

; M0: @global.4 = linkonce_odr constant
; M1: @global.4 = external constant
@global.4 = linkonce_odr constant { ptr, ptr }{ ptr getelementptr inbounds (ptr, ptr @global.2, i64 2), ptr @global.3 }

@llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

declare i32 @quux(ptr) unnamed_addr

!0 = !{i64 16, !"yyyyyyyyyyyyyyyyyyyyyyyyy"}
