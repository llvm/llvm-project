; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux -sink-insts-to-avoid-spills | FileCheck %s -check-prefix=SINK

; Ensure that we sink copy-like instructions into loops to avoid register
; spills.

; CHECK: Spill
; SINK-NOT: Spill

%struct.A = type { i32, i32, i32, i32, i32, i32 }

define void @_Z1fPhP1A(i8* nocapture readonly %input, %struct.A* %a) {
  %1 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 0
  %2 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 1
  %3 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 2
  %4 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 3
  %5 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 4
  %6 = getelementptr inbounds %struct.A, %struct.A* %a, i64 0, i32 5
  br label %.backedge

.backedge:
  %.0 = phi i8* [ %input, %0 ], [ %7, %.backedge.backedge ]
  %7 = getelementptr inbounds i8, i8* %.0, i64 1
  %8 = load i8, i8* %7, align 1
  switch i8 %8, label %.backedge.backedge [
    i8 0, label %9
    i8 10, label %10
    i8 20, label %11
    i8 30, label %12
    i8 40, label %13
    i8 50, label %14
  ]

; <label>:9
  tail call void @_Z6assignPj(i32* %1)
  br label %.backedge.backedge

; <label>:10
  tail call void @_Z6assignPj(i32* %2)
  br label %.backedge.backedge

.backedge.backedge:
  br label %.backedge

; <label>:11
  tail call void @_Z6assignPj(i32* %3)
  br label %.backedge.backedge

; <label>:12
  tail call void @_Z6assignPj(i32* %4)
  br label %.backedge.backedge

; <label>:13
  tail call void @_Z6assignPj(i32* %5)
  br label %.backedge.backedge

; <label>:14
  tail call void @_Z6assignPj(i32* %6)
  br label %.backedge.backedge
}

declare void @_Z6assignPj(i32*)
