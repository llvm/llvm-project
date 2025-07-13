; Check that we promote constant object in the source module and import it
; even when it is referenced in some other GV initializer and/or is used
; by store instructions.
; RUN: opt -thinlto-bc %s -o %t1.bc
; RUN: opt -thinlto-bc %p/Inputs/import-constant.ll -o %t2.bc
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t-out \
; RUN:    -import-constants-with-refs \
; RUN:    -r=%t1.bc,main,plx \
; RUN:    -r=%t1.bc,_Z6getObjv,l \
; RUN:    -r=%t2.bc,_Z6getObjv,pl \
; RUN:    -r=%t2.bc,val,pl \
; RUN:    -r=%t2.bc,outer,pl
; RUN: llvm-dis %t-out.2.1.promote.bc -o - | FileCheck %s --check-prefix=PROMOTE
; RUN: llvm-dis %t-out.1.3.import.bc -o - | FileCheck %s --check-prefix=IMPORT
; RUN: llvm-dis %t-out.1.4.opt.bc -o - | FileCheck %s --check-prefix=OPT

; Check when importing references is prohibited
; RUN: llvm-lto2 run -save-temps %t1.bc %t2.bc -o %t-out-norefs \
; RUN:    -import-constants-with-refs=false \
; RUN:    -r=%t1.bc,main,plx \
; RUN:    -r=%t1.bc,_Z6getObjv,l \
; RUN:    -r=%t2.bc,_Z6getObjv,pl \
; RUN:    -r=%t2.bc,val,pl \
; RUN:    -r=%t2.bc,outer,pl
; RUN: llvm-dis %t-out-norefs.1.3.import.bc -o - | FileCheck %s --check-prefix=NOREFS

; Check that variable has been promoted in the source module
; PROMOTE: @_ZL3Obj.llvm.{{.*}} = hidden constant %struct.S { i32 4, i32 8, ptr @val }

; @outer is a write-only variable, so it's been converted to zeroinitializer.
; IMPORT:      @val = available_externally global i32 42
; IMPORT-NEXT: @_ZL3Obj.llvm.{{.*}} = available_externally hidden constant %struct.S { i32 4, i32 8, ptr @val }
; IMPORT-NEXT: @outer = internal local_unnamed_addr global %struct.Q zeroinitializer

; @outer is a write-only variable that's stored to once, so the store and the global can be removed.
; OPT-NOT: @outer

; OPT:      define dso_local noundef i32 @main()
; OPT-NEXT: entry:
; OPT-NEXT:   ret i32 12

; NOREFS:      @_ZL3Obj.llvm.{{.*}} = external hidden constant %struct.S
; NOREFS-NEXT: @outer = internal local_unnamed_addr global %struct.Q zeroinitializer

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i32, ptr }

define dso_local i32 @main() local_unnamed_addr {
entry:
  %call = tail call ptr @_Z6getObjv()
  %0 = load i32, ptr %call, align 8
  %v = getelementptr inbounds %struct.S, ptr %call, i64 0, i32 1
  %1 = load i32, ptr %v, align 4
  %add = add nsw i32 %1, %0
  ret i32 %add
}

declare dso_local ptr @_Z6getObjv() local_unnamed_addr
