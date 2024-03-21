; RUN: llc -mtriple=r600 -mcpu=redwood -mtriple=r600-- < %s | FileCheck %s

; We want all MULLO_INT inst to be last in their instruction group
;CHECK: {{^}}fill3d:
;CHECK-NOT: MULLO_INT T[0-9]+

define amdgpu_kernel void @fill3d(ptr addrspace(1) nocapture %out) nounwind {
entry:
  %x.i = tail call i32 @llvm.r600.read.global.size.x() nounwind readnone
  %y.i18 = tail call i32 @llvm.r600.read.global.size.y() nounwind readnone
  %mul = mul i32 %y.i18, %x.i
  %z.i17 = tail call i32 @llvm.r600.read.global.size.z() nounwind readnone
  %mul3 = mul i32 %mul, %z.i17
  %x.i.i = tail call i32 @llvm.r600.read.tgid.x() nounwind readnone
  %x.i12.i = tail call i32 @llvm.r600.read.local.size.x() nounwind readnone
  %mul26.i = mul i32 %x.i12.i, %x.i.i
  %x.i4.i = tail call i32 @llvm.r600.read.tidig.x() nounwind readnone
  %add.i16 = add i32 %x.i4.i, %mul26.i
  %mul7 = mul i32 %add.i16, %y.i18
  %y.i.i = tail call i32 @llvm.r600.read.tgid.y() nounwind readnone
  %y.i14.i = tail call i32 @llvm.r600.read.local.size.y() nounwind readnone
  %mul30.i = mul i32 %y.i14.i, %y.i.i
  %y.i6.i = tail call i32 @llvm.r600.read.tidig.y() nounwind readnone
  %add.i14 = add i32 %mul30.i, %mul7
  %mul819 = add i32 %add.i14, %y.i6.i
  %add = mul i32 %mul819, %z.i17
  %z.i.i = tail call i32 @llvm.r600.read.tgid.z() nounwind readnone
  %z.i16.i = tail call i32 @llvm.r600.read.local.size.z() nounwind readnone
  %mul33.i = mul i32 %z.i16.i, %z.i.i
  %z.i8.i = tail call i32 @llvm.r600.read.tidig.z() nounwind readnone
  %add.i = add i32 %z.i8.i, %mul33.i
  %add13 = add i32 %add.i, %add
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %add13
  store i32 %mul3, ptr addrspace(1) %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.x() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.y() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.z() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.x() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.y() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.z() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.y() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.z() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.global.size.x() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.global.size.y() nounwind readnone

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.global.size.z() nounwind readnone

!opencl.kernels = !{!0, !1, !2}

!0 = !{null}
!1 = !{null}
!2 = !{ptr @fill3d}
