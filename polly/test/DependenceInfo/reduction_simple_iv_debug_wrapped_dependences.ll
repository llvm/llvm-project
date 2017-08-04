; RUN: opt %loadPolly -polly-dependences -analyze -debug-only=polly-dependence 2>&1 < %s | FileCheck %s
;
; REQUIRES: asserts
;
; CHECK:      Read: { [Stmt_for_cond[i0] -> MemRef_sum[0{{\]\]}} -> MemRef_sum[0] : 0 <= i0 <= 100 }
; CHECK-NEXT: Write: { [Stmt_for_cond[i0] -> MemRef_sum[0{{\]\]}} -> MemRef_sum[0] : 0 <= i0 <= 100 }
; CHECK-NEXT: MayWrite: {  }
;
; CHECK:      Wrapped Dependences:
; CHECK-NEXT:     RAW dependences:
; CHECK-NEXT:         { [Stmt_for_cond[i0] -> MemRef_sum[0{{\]\]}} -> [Stmt_for_cond[1 + i0] -> MemRef_sum[0{{\]\]}} : 0 <= i0 <= 99 }
; CHECK-NEXT:     WAR dependences:
; CHECK-NEXT:         { [Stmt_for_cond[i0] -> MemRef_sum[0{{\]\]}} -> [Stmt_for_cond[1 + i0] -> MemRef_sum[0{{\]\]}} : 0 <= i0 <= 99 }
; CHECK-NEXT:     WAW dependences:
; CHECK-NEXT:         { [Stmt_for_cond[i0] -> MemRef_sum[0{{\]\]}} -> [Stmt_for_cond[1 + i0] -> MemRef_sum[0{{\]\]}} : 0 <= i0 <= 99 }
; CHECK-NEXT:     Reduction dependences:
; CHECK-NEXT:         n/a
;
; CHECK:      Final Wrapped Dependences:
; CHECK-NEXT:     RAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAR dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     Reduction dependences:
; CHECK-NEXT:         { [Stmt_for_cond[i0] -> MemRef_sum[0{{\]\]}} -> [Stmt_for_cond[1 + i0] -> MemRef_sum[0{{\]\]}} : 0 <= i0 <= 99 }
;
; CHECK:      Zipped Dependences:
; CHECK-NEXT:     RAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAR dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     Reduction dependences:
; CHECK-NEXT:         { [Stmt_for_cond[i0] -> Stmt_for_cond[1 + i0{{\]\]}} -> [MemRef_sum[0] -> MemRef_sum[0{{\]\]}} : 0 <= i0 <= 99 }
;
; CHECK:      Unwrapped Dependences:
; CHECK-NEXT:     RAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAR dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     Reduction dependences:
; CHECK-NEXT:         { Stmt_for_cond[i0] -> Stmt_for_cond[1 + i0] : 0 <= i0 <= 99 }
;
; CHECK:          RAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAR dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     Reduction dependences:
; CHECK-NEXT:         { Stmt_for_cond[i0] -> Stmt_for_cond[1 + i0] : 0 <= i0 <= 99 }
;
; CHECK:          RAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAR dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     WAW dependences:
; CHECK-NEXT:         {  }
; CHECK-NEXT:     Reduction dependences:
; CHECK-NEXT:         { Stmt_for_cond[i0] -> Stmt_for_cond[1 + i0] : 0 <= i0 <= 99 }
;
; void f(int* sum) {
;   for (int i = 0; i <= 100; i++)
;     sum += i * 3;
; }
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(i32* %sum) {
entry:
  br label %entry.split1

entry.split1:                                     ; preds = %entry
  br label %entry.split

entry.split:                                      ; preds = %entry.split1
  br label %for.cond

for.cond:                                         ; preds = %for.cond, %entry.split
  %i1.0 = phi i32 [ 0, %entry.split ], [ %inc, %for.cond ]
  %sum.reload = load i32, i32* %sum
  %mul = mul nsw i32 %i1.0, 3
  %add = add nsw i32 %sum.reload, %mul
  %inc = add nsw i32 %i1.0, 1
  store i32 %add, i32* %sum
  %cmp = icmp slt i32 %i1.0, 100
  br i1 %cmp, label %for.cond, label %for.end

for.end:                                          ; preds = %for.cond
  ret void
}
