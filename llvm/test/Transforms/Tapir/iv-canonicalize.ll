; Thanks to Valentin Churavy for providing this test case.
;
; RUN: opt %s -indvars -S | FileCheck %s -check-prefix=IV
; RUN: opt %s -passes='indvars' -S | FileCheck %s -check-prefix=IV
; RUN: opt %s -indvars -instcombine -S | FileCheck %s -check-prefix=IC
; RUN: opt %s -passes='loop(indvars),instcombine' -S | FileCheck %s -check-prefix=IC
; RUN: opt %s -indvars -instcombine -loop-spawning-ti -S | FileCheck %s -check-prefix=LS
; RUN: opt %s -passes='function(loop(indvars),instcombine),loop-spawning' -S | FileCheck %s -check-prefix=LS

; ModuleID = 'simple.ll'
source_filename = "mynorm"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128-ni:10:11:12:13"
target triple = "x86_64-unknown-linux-gnu"

%0 = type { i8 addrspace(13)*, i64, i16, i16, i32 }
%jl_value_t = type opaque

; @llvm.compiler.used = appending global [2 x i8*] [i8* bitcast (%jl_value_t addrspace(10)* (i8*, i32, i32)* @jl_gc_pool_alloc to i8*), i8* bitcast (%jl_value_t addrspace(10)* (i8*, i64)* @jl_gc_big_alloc to i8*)], section "llvm.metadata"

; Function Attrs: argmemonly nounwind
declare token @llvm.syncregion.start() #0

define void @julia_mynorm(i64) {
top:
  %syncreg = call token @llvm.syncregion.start()
  %1 = call %jl_value_t*** @julia.ptls_states()
  %2 = icmp sgt i64 %0, 0
  br i1 %2, label %L7.L12_crit_edge, label %L27

L7.L12_crit_edge:                                 ; preds = %top
  br label %L12

; LS: {{^L7.L12_crit_edge}}
; LS: call fastcc void @julia_mynorm.outline_L12.ls1(i64 0, i64 %0
; LS-NEXT: br label %L27.loopexit

L12:                                              ; preds = %loop.cond, %L7.L12_crit_edge
  %value_phi3 = phi i64 [ 1, %L7.L12_crit_edge ], [ %6, %loop.cond ]
  detach within %syncreg, label %loop, label %loop.cond
; IV: {{^L12}}
; IV: %indvar = phi i64 [ %indvar.next, %loop.cond ], [ 0, %L7.L12_crit_edge ]
; IV: detach within %syncreg, label %loop, label %loop.cond

loop:                                             ; preds = %L12
  %3 = sitofp i64 %value_phi3 to double
  %4 = call double @julia_log_12250(double %3)
  reattach within %syncreg, label %loop.cond

loop.cond:                                        ; preds = %loop, %L12
  %5 = icmp eq i64 %value_phi3, %0
  %6 = add nuw nsw i64 %value_phi3, 1
  br i1 %5, label %L27.loopexit, label %L12, !llvm.loop !84

; IV: {{^loop.cond}}
; IV: %indvar.next = add nuw i64 %indvar, 1
; IV: br i1 %exitcond, label %L27.loopexit, label %L12

; IC: {{^loop.cond}}
; IC: %indvar.next = add nuw i64 %indvar, 1
; IC: %exitcond = icmp eq i64 %indvar.next, %0
; IC: br i1 %exitcond, label %L27.loopexit, label %L12

L27.loopexit:                                     ; preds = %loop.cond
  br label %L27

L27:                                              ; preds = %L27.loopexit, %top
  sync within %syncreg, label %exit

exit:                                             ; preds = %L27
  ret void
}

; LS: define private fastcc void @julia_mynorm.outline_L12.ls1(i64 %indvar.start.ls1, i64 %end.ls1, i64 %grainsize.ls1)

declare %jl_value_t*** @julia.ptls_states()

declare double @julia_log_12250(double)

attributes #0 = { argmemonly nounwind }

!84 = distinct !{!84, !85}
!85 = !{!"tapir.loop.spawn.strategy", i32 1}
