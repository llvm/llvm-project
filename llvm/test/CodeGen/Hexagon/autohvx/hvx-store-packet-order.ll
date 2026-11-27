; RUN: llc -mtriple=hexagon -mcpu=hexagonv79 -mattr=+hvxv79,+hvx-length128b \
; RUN:   < %s | FileCheck %s

; Dynamic insertelement spills the vector, stores the element, reloads.
; The HVX spill must not share a packet with the element store.

; dst[3*i+1] = src[3*i+1] for i in [0, 20), through a <20 x float>.

; Loop body: the spill is alone in its packet, the element store follows.
; CHECK-LABEL: strided_copy:
; CHECK:      // %gather
; CHECK:      {
; CHECK-NEXT:   vmem(r29+#0) = v{{[0-9]+}}
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NOT:  }
; CHECK:        memw(r{{[0-9]+}}+#0) = r{{[0-9]+}}
; CHECK:      } :endloop0

; Last iteration, peeled after the loop: same ordering.
; CHECK-NEXT: // %bb.
; CHECK-NEXT: {
; CHECK-NOT:  memw
; CHECK:        vmem(r29+#0) = v{{[0-9]+}}
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NOT:  }
; CHECK:        memw(r{{[0-9]+}}+#0) = r{{[0-9]+}}

define void @strided_copy(ptr %src, ptr %dst) {
entry:
  br label %gather

gather:
  %v = phi <20 x float> [ poison, %entry ], [ %v.next, %gather ]
  %i = phi i32 [ 0, %entry ], [ %i.next, %gather ]
  %off = mul i32 %i, 12
  %p = getelementptr i8, ptr %src, i32 %off
  %q = getelementptr i8, ptr %p, i32 4
  %x = load float, ptr %q, align 4
  %v.next = insertelement <20 x float> %v, float %x, i32 %i
  %i.next = add i32 %i, 1
  %gdone = icmp eq i32 %i.next, 20
  br i1 %gdone, label %scatter, label %gather

scatter:
  %j = phi i32 [ 0, %gather ], [ %j.next, %scatter ]
  %y = extractelement <20 x float> %v.next, i32 %j
  %off2 = mul i32 %j, 12
  %r = getelementptr i8, ptr %dst, i32 %off2
  %s = getelementptr i8, ptr %r, i32 4
  store float %y, ptr %s, align 4
  %j.next = add i32 %j, 1
  %sdone = icmp eq i32 %j.next, 20
  br i1 %sdone, label %exit, label %scatter

exit:
  ret void
}
