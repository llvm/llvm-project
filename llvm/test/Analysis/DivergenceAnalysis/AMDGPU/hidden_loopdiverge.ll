; RUN: opt -mtriple amdgcn-unknown-amdhsa -analyze -divergence -use-gpu-divergence-analysis %s | FileCheck %s

; divergent loop (H<header><exiting to X>, B<exiting to Y>)
; the divergent join point in %exit is obscured by uniform control joining in %X
define amdgpu_kernel void @hidden_loop_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'hidden_loop_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %X, label %H  ; uniform

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %B ]
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %div.exitx, label %X, label %B ; divergent branch
; CHECK: DIVERGENT: %div.exitx =  
; CHECK: DIVERGENT: br i1 %div.exitx, 

B:
  %uni.inc = add i32 %uni.merge.h, 1
  %div.exity = icmp sgt i32 %tid, 0
  br i1 %div.exity, label %Y, label %H ; divergent branch
; CHECK: DIVERGENT: %div.exity =  
; CHECK: DIVERGENT: br i1 %div.exity, 

X:
  %div.merge.x = phi i32 [ %a, %entry ], [ %uni.merge.h, %H ] ; temporal divergent phi
  br i1 %uni.cond, label %Y, label %exit
; CHECK: DIVERGENT: %div.merge.x =

Y:
  %div.merge.y = phi i32 [ 42, %X ], [ %b, %B ]
  br label %exit
; CHECK: DIVERGENT: %div.merge.y =

exit:
  %div.merge.exit = phi i32 [ %a, %X ], [ %b, %Y ]
  ret void
; CHECK: DIVERGENT: %div.merge.exit =
}

; divergent loop (H<header><exiting to X>, B<exiting to Y>)
; the phi nodes in X and Y don't actually receive divergent values
define amdgpu_kernel void @unobserved_loop_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'unobserved_loop_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  br i1 %uni.cond, label %X, label %H  ; uniform

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %B ]
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %div.exitx, label %X, label %B ; divergent branch
; CHECK: DIVERGENT: %div.exitx =  
; CHECK: DIVERGENT: br i1 %div.exitx, 

B:
  %uni.inc = add i32 %uni.merge.h, 1
  %div.exity = icmp sgt i32 %tid, 0
  br i1 %div.exity, label %Y, label %H ; divergent branch
; CHECK: DIVERGENT: %div.exity =  
; CHECK: DIVERGENT: br i1 %div.exity, 

X:
  %uni.merge.x = phi i32 [ %a, %entry ], [ %b, %H ] 
  br label %exit

Y:
  %uni.merge.y = phi i32 [ %b, %B ]
  br label %exit

exit:
  %div.merge.exit = phi i32 [ %a, %X ], [ %b, %Y ]
  ret void
; CHECK: DIVERGENT: %div.merge.exit =
}

; divergent loop (G<header>, L<exiting to D>) inside divergent loop (H<header>, B<exiting to X>, C<exiting to Y>, D, G, L)
; the inner loop has no exit to top level.
; the outer loop becomes divergent as its exiting branch in C is control-dependent on the inner loop's divergent loop exit in D.
define amdgpu_kernel void @hidden_nestedloop_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'hidden_nestedloop_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %uni.cond, label %X, label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %D ]
  br i1 %uni.cond, label %G, label %B
; CHECK: DIVERGENT: %div.exitx =  

B:
  br i1 %uni.cond, label %X, label %C 

C:
  br i1 %uni.cond, label %Y, label %D

D:
  %uni.inc = add i32 %uni.merge.h, 1
  br label %H

G:
  br i1 %div.exitx, label %C, label %L
; CHECK: DIVERGENT: br i1 %div.exitx, 

L:
  br i1 %uni.cond, label %D, label %G

X:
  %uni.merge.x = phi i32 [ %a, %entry ], [ %uni.merge.h, %B ]
  br i1 %uni.cond, label %Y, label %exit

Y:
  %div.merge.y = phi i32 [ 42, %X ], [ %b, %C ]
  br label %exit
; CHECK: DIVERGENT: %div.merge.y =

exit:
  %div.merge.exit = phi i32 [ %a, %X ], [ %b, %Y ]
  ret void
; CHECK: DIVERGENT: %div.merge.exit =
}

; divergent loop (G<header>, L<exiting to X>) in divergent loop (H<header>, B<exiting to C>, C, G, L)
; the outer loop has no immediately divergent exiting edge.
; the inner exiting edge is exiting to top-level through the outer loop causing both to become divergent.
define amdgpu_kernel void @hidden_doublebreak_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'hidden_doublebreak_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %uni.cond, label %X, label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc, %C ]
  br i1 %uni.cond, label %G, label %B
; CHECK: DIVERGENT: %div.exitx =  

B:
  br i1 %uni.cond, label %Y, label %C 

C:
  %uni.inc = add i32 %uni.merge.h, 1
  br label %H

G:
  br i1 %div.exitx, label %X, label %L ; two-level break
; CHECK: DIVERGENT: br i1 %div.exitx, 

L:
  br i1 %uni.cond, label %C, label %G

X:
  %div.merge.x = phi i32 [ %a, %entry ], [ %uni.merge.h, %G ] ; temporal divergence
  br label %Y
; CHECK: DIVERGENT: %div.merge.x =

Y:
  %div.merge.y = phi i32 [ 42, %X ], [ %b, %B ]
  ret void
; CHECK: DIVERGENT: %div.merge.y =
}

; divergent loop (G<header>, L<exiting to D>) contained inside a uniform loop (H<header>, B, G, L , D<exiting to x>)
define amdgpu_kernel void @hidden_containedloop_diverge(i32 %n, i32 %a, i32 %b) #0 {
; CHECK-LABEL: Printing analysis 'Legacy Divergence Analysis' for function 'hidden_containedloop_diverge':
; CHECK-NOT: DIVERGENT: %uni.
; CHECK-NOT: DIVERGENT: br i1 %uni.

entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %uni.cond = icmp slt i32 %a, 0
  %div.exitx = icmp slt i32 %tid, 0
  br i1 %uni.cond, label %X, label %H

H:
  %uni.merge.h = phi i32 [ 0, %entry ], [ %uni.inc.d, %D ]
  br i1 %uni.cond, label %G, label %B
; CHECK: DIVERGENT: %div.exitx =  

B:
  %div.merge.b = phi i32 [ 42, %H ], [ %uni.merge.g, %G ]
  br label %D
; CHECK: DIVERGENT: %div.merge.b =

G:
  %uni.merge.g = phi i32 [ 123, %H ], [ %uni.inc.l, %L ]
  br i1 %div.exitx, label %B, label %L
; CHECK: DIVERGENT: br i1 %div.exitx, 

L:
  %uni.inc.l = add i32 %uni.merge.g, 1
  br i1 %uni.cond, label %G, label %D

D:
  %uni.inc.d = add i32 %uni.merge.h, 1
  br i1 %uni.cond, label %X, label %H

X:
  %uni.merge.x = phi i32 [ %a, %entry ], [ %uni.inc.d, %D ]
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #0

attributes #0 = { nounwind readnone }
