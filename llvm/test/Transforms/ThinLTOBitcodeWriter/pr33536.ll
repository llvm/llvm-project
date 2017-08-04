; Test for a bug specific to the new pass manager where we may build a domtree
; to make more precise AA queries for functions.
;
; RUN: opt -aa-pipeline=default -passes='no-op-module' -debug-pass-manager -thinlto-bc -o %t %s
; RUN: llvm-modextract -b -n 0 -o - %t | llvm-dis | FileCheck --check-prefix=M0 %s
; RUN: llvm-modextract -b -n 1 -o - %t | llvm-dis | FileCheck --check-prefix=M1 %s

target triple = "x86_64-unknown-linux-gnu"

%struct.hoge = type { %struct.widget }
%struct.widget = type { i32 (...)** }

; M0: @global = local_unnamed_addr global
; M1-NOT: @global
@global = local_unnamed_addr global %struct.hoge { %struct.widget { i32 (...)** bitcast (i8** getelementptr inbounds ({ [3 x i8*] }, { [3 x i8*] }* @global.1, i32 0, inrange i32 0, i32 2) to i32 (...)**) } }, align 8

; M0: @global.1 = external unnamed_addr constant
; M1: @global.1 = linkonce_odr unnamed_addr constant
@global.1 = linkonce_odr unnamed_addr constant { [3 x i8*] } { [3 x i8*] [i8* null, i8* bitcast ({ i8*, i8* }* @global.4 to i8*), i8* bitcast (i32 (%struct.widget*)* @quux to i8*)] }, align 8, !type !0

; M0: @global.2 = external global
; M1-NOT: @global.2
@global.2 = external global i8*

; M0: @global.3 = linkonce_odr constant
; M1-NOT: @global.3
@global.3 = linkonce_odr constant [22 x i8] c"zzzzzzzzzzzzzzzzzzzzz\00"

; M0: @global.4 = linkonce_odr constant
; M1: @global.4 = external constant
@global.4 = linkonce_odr constant { i8*, i8* }{ i8* bitcast (i8** getelementptr inbounds (i8*, i8** @global.2, i64 2) to i8*), i8* getelementptr inbounds ([22 x i8], [22 x i8]* @global.3, i32 0, i32 0) }

@llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

declare i32 @quux(%struct.widget*) unnamed_addr

!0 = !{i64 16, !"yyyyyyyyyyyyyyyyyyyyyyyyy"}
