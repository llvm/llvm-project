; RUN: llc < %s -mtriple=i386-pc-windows \
; RUN:   -debug-only=machine-scheduler -o - 2>&1 | FileCheck %s
; REQUIRES: asserts


%TimeTicks = type { i64 }
%OnceCallback = type { ptr }
%WeakPtr = type { ptr, ptr }
%Tuple = type { ptr, ptr, i64, ptr, ptr, ptr }
declare x86_thiscallcc void @use(ptr)


;; MOVUPSrm is a merged load from stack objects %b, %c, and %d.
;; Check that the wide load is not marked as "from %ir.b".
;; Check that the zero-store to %c's SU is preceeded by the SU for that load.

; CHECK:       Current Schedule Region
; CHECK-LABEL: f:%bb.0 bb
; CHECK:       SU([[LOAD:[0-9]+]]): {{.*}}MOVUPSrm %fixed-stack.3, 1, $noreg, 0, $noreg :: (load (s128) from %fixed-stack.3, align 8)
; CHECK:       SU([[ZERO:[0-9]+]]): %[[ZREG:[0-9]+]]:gr32 = MOV32r0
; CHECK:       SU([[STORE:[0-9]+]]): MOV32mr {{.*}}, %[[ZREG]]:gr32
; CHECK:         Predecessors:
; CHECK:           SU([[ZERO]])
; CHECK:           SU([[LOAD]]){{.*}}Memory


define x86_thiscallcc void @f(ptr %a,
                              ptr byval(%TimeTicks) %b,
                              ptr byval(%OnceCallback) %c,
                              ptr byval(%OnceCallback) %d,
                              ptr byval(%WeakPtr) %e) #0 {
bb:
  %tuple = alloca %Tuple

  ; Copy a to the tuple.
  store ptr %a, ptr %tuple

  ; Copy *b to the tuple.
  %tt = load i64, ptr %b
  %tuple.16 = getelementptr inbounds nuw i8, ptr %tuple, i32 16
  store i64 %tt, ptr %tuple.16

  ; Copy *c to the tuple, then zero it.
  %tuple.24 = getelementptr inbounds nuw i8, ptr %tuple, i32 24
  %cc = load ptr, ptr %c
  store ptr %cc, ptr %tuple.24
  store ptr null, ptr %c

  ; Copy *d to the tuple, then zero it.
  %tuple.28 = getelementptr inbounds nuw i8, ptr %tuple, i32 28
  %dd = load ptr, ptr %d
  store ptr %dd, ptr %tuple.28
  store ptr null, ptr %d

  ; Use the tuple.
  call x86_thiscallcc void @use(ptr %tuple)
  br label %bb2

bb2:
  call x86_thiscallcc void @use(ptr %a)
  call x86_thiscallcc void @use(ptr %c)
  call x86_thiscallcc void @use(ptr %d)
  call x86_thiscallcc void @use(ptr %e)
  ret void
}

attributes #0 = { optsize "target-cpu"="pentium4" "tune-cpu"="generic" }
