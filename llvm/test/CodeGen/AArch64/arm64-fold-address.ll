; RUN: llc < %s -O2 -mtriple=arm64-apple-darwin | FileCheck %s

%0 = type opaque
%struct.CGRect = type { %struct.CGPoint, %struct.CGSize }
%struct.CGPoint = type { double, double }
%struct.CGSize = type { double, double }

@"OBJC_IVAR_$_UIScreen._bounds" = external hidden global i64, section "__DATA, __objc_ivar", align 8

define hidden %struct.CGRect @nofold(ptr nocapture %self, ptr nocapture %_cmd) nounwind readonly optsize ssp {
entry:
; CHECK-LABEL: nofold:
; CHECK: add x[[REG:[0-9]+]], x0, x{{[0-9]+}}
; CHECK: ldp d0, d1, [x[[REG]]]
; CHECK: ldp d2, d3, [x[[REG]], #16]
; CHECK: ret
  %ivar = load i64, ptr @"OBJC_IVAR_$_UIScreen._bounds", align 8, !invariant.load !4
  %add.ptr = getelementptr inbounds i8, ptr %self, i64 %ivar
  %tmp11 = load double, ptr %add.ptr, align 8
  %add.ptr.sum = add i64 %ivar, 8
  %add.ptr10.1 = getelementptr inbounds i8, ptr %self, i64 %add.ptr.sum
  %tmp12 = load double, ptr %add.ptr10.1, align 8
  %add.ptr.sum17 = add i64 %ivar, 16
  %add.ptr4.1 = getelementptr inbounds i8, ptr %self, i64 %add.ptr.sum17
  %tmp = load double, ptr %add.ptr4.1, align 8
  %add.ptr4.1.sum = add i64 %ivar, 24
  %add.ptr4.1.1 = getelementptr inbounds i8, ptr %self, i64 %add.ptr4.1.sum
  %tmp5 = load double, ptr %add.ptr4.1.1, align 8
  %insert14 = insertvalue %struct.CGPoint undef, double %tmp11, 0
  %insert16 = insertvalue %struct.CGPoint %insert14, double %tmp12, 1
  %insert = insertvalue %struct.CGRect undef, %struct.CGPoint %insert16, 0
  %insert7 = insertvalue %struct.CGSize undef, double %tmp, 0
  %insert9 = insertvalue %struct.CGSize %insert7, double %tmp5, 1
  %insert3 = insertvalue %struct.CGRect %insert, %struct.CGSize %insert9, 1
  ret %struct.CGRect %insert3
}

define hidden %struct.CGRect @fold(ptr nocapture %self, ptr nocapture %_cmd) nounwind readonly optsize ssp {
entry:
; CHECK-LABEL: fold:
; CHECK: ldr d0, [x0, x{{[0-9]+}}]
; CHECK-NOT: add x0, x0, x1
; CHECK: ret
  %ivar = load i64, ptr @"OBJC_IVAR_$_UIScreen._bounds", align 8, !invariant.load !4
  %add.ptr = getelementptr inbounds i8, ptr %self, i64 %ivar
  %tmp11 = load double, ptr %add.ptr, align 8
  %add.ptr10.1 = getelementptr inbounds i8, ptr %self, i64 %ivar
  %tmp12 = load double, ptr %add.ptr10.1, align 8
  %add.ptr4.1 = getelementptr inbounds i8, ptr %self, i64 %ivar
  %tmp = load double, ptr %add.ptr4.1, align 8
  %add.ptr4.1.1 = getelementptr inbounds i8, ptr %self, i64 %ivar
  %tmp5 = load double, ptr %add.ptr4.1.1, align 8
  %insert14 = insertvalue %struct.CGPoint undef, double %tmp11, 0
  %insert16 = insertvalue %struct.CGPoint %insert14, double %tmp12, 1
  %insert = insertvalue %struct.CGRect undef, %struct.CGPoint %insert16, 0
  %insert7 = insertvalue %struct.CGSize undef, double %tmp, 0
  %insert9 = insertvalue %struct.CGSize %insert7, double %tmp5, 1
  %insert3 = insertvalue %struct.CGRect %insert, %struct.CGSize %insert9, 1
  ret %struct.CGRect %insert3
}


!llvm.module.flags = !{!0, !1, !2, !3}

!0 = !{i32 1, !"Objective-C Version", i32 2}
!1 = !{i32 1, !"Objective-C Image Info Version", i32 0}
!2 = !{i32 1, !"Objective-C Image Info Section", !"__DATA, __objc_imageinfo, regular, no_dead_strip"}
!3 = !{i32 4, !"Objective-C Garbage Collection", i32 0}
!4 = !{}
