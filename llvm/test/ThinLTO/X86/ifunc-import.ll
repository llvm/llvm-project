; RUN: rm -rf %t && split-file %s %t && cd %t
; RUN: opt -module-summary -o a.bc a.ll
; RUN: opt -module-summary -o b.bc b.ll
; RUN: llvm-lto2 run a.bc b.bc -o t --save-temps \
; RUN:   -r a.bc,external.ifunc,pl -r a.bc,use,pl -r a.bc,use2,pl -r a.bc,__cpu_model,lx \
; RUN:   -r b.bc,main,plx -r b.bc,use,l -r b.bc,use2,l
; RUN: llvm-dis < t.1.3.import.bc | FileCheck %s --check-prefix=A
; RUN: llvm-dis < t.2.3.import.bc | FileCheck %s --check-prefix=B --implicit-check-not='@internal.resolver'

; A: define internal nonnull ptr @internal.resolver()
; A: define internal i32 @internal.default.1(i32 %n)

;; The ifunc implementations of internal.ifunc are internal in A, so they cannot
;; be referenced by B. Our implementation actually ensures that the ifunc resolver
;; along with its implementations are not imported.
; B: declare i32 @use(i32) local_unnamed_addr
; B: define available_externally i32 @use2(i32 %n) local_unnamed_addr
; B: declare i32 @external.ifunc(i32)

;--- a.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$internal.resolver = comdat any

@__cpu_model = external dso_local local_unnamed_addr global { i32, i32, i32, [1 x i32] }

@internal.ifunc = internal ifunc i32 (i32), ptr @internal.resolver
@external.ifunc = ifunc i32 (i32), ptr @internal.resolver

define dso_local i32 @use(i32 %n) local_unnamed_addr {
entry:
  %call = call i32 @internal.ifunc(i32 %n)
  ret i32 %call
}

define dso_local i32 @use2(i32 %n) local_unnamed_addr {
entry:
  %call = call i32 @external.ifunc(i32 %n)
  ret i32 %call
}

define internal nonnull ptr @internal.resolver() comdat {
entry:
  %0 = load i32, ptr getelementptr inbounds ({ i32, i32, i32, [1 x i32] }, ptr @__cpu_model, i64 0, i32 3, i64 0), align 4
  %1 = and i32 %0, 4
  %.not = icmp eq i32 %1, 0
  %internal.popcnt.0.internal.default.1 = select i1 %.not, ptr @internal.default.1, ptr @internal.popcnt.0
  ret ptr %internal.popcnt.0.internal.default.1
}

define internal i32 @internal.popcnt.0(i32 %n) {
entry:
  %0 = call i32 @llvm.ctpop.i32(i32 %n)
  ret i32 %0
}

declare i32 @llvm.ctpop.i32(i32)

define internal i32 @internal.default.1(i32 %n) {
entry:
  %0 = call i32 @llvm.ctpop.i32(i32 %n)
  ret i32 %0
}

;--- b.ll
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local i32 @main() local_unnamed_addr {
entry:
  %0 = call i32 @use(i32 0)
  %1 = call i32 @use2(i32 0)
  %2 = add i32 %0, %1
  ret i32 %2
}

declare i32 @use(i32) local_unnamed_addr
declare i32 @use2(i32) local_unnamed_addr
