; RUN: opt -passes=debugify,loop-vectorize \
; RUN: -force-tail-folding-style=data-with-evl \
; RUN: -prefer-predicate-over-epilogue=predicate-dont-vectorize \
; RUN: -mtriple=riscv64 -mattr=+v -riscv-v-vector-bits-max=128 -S < %s 2>&1 | FileCheck --check-prefix=DEBUGLOC %s

; Testing the debug locations of the generated vector intrinsic is same as
; its scalar counterpart.

define void @vp_select(ptr %a, ptr %b, ptr %c, i64 %N) {
; DEBUGLOC-LABEL: define void @vp_select(
; DEBUGLOC: vector.body:
; DEBUGLOC:   = select <vscale x 4 x i1> %{{.+}}, <vscale x 4 x i32> %{{.+}}, <vscale x 4 x i32> %{{.+}}, !dbg ![[SELLOC:[0-9]+]]
; DEBUGLOC: loop:
; DEBUGLOC:   = select i1 %{{.+}}, i32 %{{.+}}, i32 %{{.+}}, !dbg ![[SELLOC]]
;
 entry:
   br label %loop

loop:
   %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
   %gep.b = getelementptr inbounds i32, ptr %b, i64 %iv
   %load.b = load i32, ptr %gep.b, align 4
   %gep.c = getelementptr inbounds i32, ptr %c, i64 %iv
   %load.c = load i32, ptr %gep.c, align 4
   %cmp = icmp sgt i32 %load.b, %load.c
   %neg.c = sub i32 0, %load.c
   %sel = select i1 %cmp, i32 %load.c, i32 %neg.c
   %add = add i32 %sel, %load.b
   %gep.a = getelementptr inbounds i32, ptr %a, i64 %iv
   store i32 %add, ptr %gep.a, align 4
   %iv.next = add nuw nsw i64 %iv, 1
   %exitcond = icmp eq i64 %iv.next, %N
   br i1 %exitcond, label %exit, label %loop

 exit:
   ret void
 }

 ; DEBUGLOC: [[SELLOC]] = !DILocation(line: 9
