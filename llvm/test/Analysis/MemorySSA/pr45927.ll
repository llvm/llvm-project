; RUN: opt -disable-output -loop-simplify -lcssa -licm -print-memoryssa < %s -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt -disable-output -aa-pipeline=basic-aa -passes='loop-mssa(licm),print<memoryssa>' < %s 2>&1 | FileCheck %s


@a = external dso_local global i16, align 1
@c = external dso_local global i16, align 1

; CHECK-LABEL: @main()

; CHECK: entry:
; CHECK-NEXT: %res.addr.i = alloca i16
; CHECK-NEXT: ; MemoryUse(liveOnEntry)
; CHECK-NEXT: %c.promoted = load i16, i16* @c
; CHECK-NEXT: br label %for.cond.i

; CHECK: for.cond.i:
; CHECK-NEXT: ; [[NO5:.*]] = MemoryPhi({entry,liveOnEntry},{f.exit.i,[[NO5]]})
; CHECK-NEXT: %inc.i1 = phi i16 [ %inc.i, %f.exit.i ], [ %c.promoted, %entry ]
; CHECK-NEXT: %inc.i = add nsw i16 %inc.i1, 1
; CHECK-NEXT: br i1 false, label %f.exit.thread.i, label %f.exit.i

; CHECK: f.exit.thread.i:
; CHECK-NEXT: %inc.i.lcssa = phi i16 [ %inc.i, %for.cond.i ]
; CHECK-NEXT: ; [[NO6:.*]] = MemoryDef([[NO5]])
; CHECK-NEXT: store i16 %inc.i.lcssa, i16* @c, align 1
; CHECK-NEXT: ; [[NO2:.*]] = MemoryDef([[NO6]])
; CHECK-NEXT: store i16 1, i16* @a, align 1
; CHECK-NEXT: ; MemoryUse([[NO2]])
; CHECK-NEXT: %tmp2 = load i16, i16* @c, align 1
; CHECK-NEXT: br label %g.exit

; CHECK: f.exit.i
; CHECK-NEXT: br i1 false, label %g.exit.loopexit, label %for.cond.i

; CHECK: g.exit.loopexit:
; CHECK-NEXT: %inc.i.lcssa2 = phi i16 [ %inc.i, %f.exit.i ]
; CHECK-NEXT: ; [[NO7:.*]] = MemoryDef([[NO5]])
; CHECK-NEXT: store i16 %inc.i.lcssa2, i16* @c, align 1
; CHECK-NEXT: br label %g.exit

; CHECK: g.exit
; CHECK-NEXT: ; [[NO4:.*]] = MemoryPhi({f.exit.thread.i,[[NO2]]},{g.exit.loopexit,[[NO7]]})
; CHECK-NEXT: ; MemoryUse([[NO4]])
; CHECK-NEXT:  %tmp1 = load i16, i16* @c, align 1
; CHECK-NEXT: ; [[NO3:.*]] = MemoryDef([[NO4]])
; CHECK-NEXT:  store i16 %tmp1, i16* %res.addr.i, align 1
; CHECK-NEXT:  ret void

define dso_local void @main() {
entry:
  %res.addr.i = alloca i16, align 1
  br label %for.cond.i

for.cond.i:                                       ; preds = %f.exit.i, %entry
  %tmp0 = load i16, i16* @c, align 1
  %inc.i = add nsw i16 %tmp0, 1
  store i16 %inc.i, i16* @c, align 1
  br i1 false, label %f.exit.thread.i, label %f.exit.i

f.exit.thread.i:                                  ; preds = %for.cond.i
  store i16 1, i16* @a, align 1
  %tmp2 = load i16, i16* @c, align 1
  br label %g.exit

f.exit.i:                                         ; preds = %for.cond.i
  br i1 false, label %g.exit, label %for.cond.i

g.exit:                                           ; preds = %f.exit.i, %f.exit.thread.i
  %tmp1 = load i16, i16* @c, align 1
  store i16 %tmp1, i16* %res.addr.i, align 1
  ret void
}

