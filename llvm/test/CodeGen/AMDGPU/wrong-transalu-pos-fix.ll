; RUN: llc -march=r600 -mcpu=redwood -mtriple=r600-- < %s | FileCheck %s

; We want all MULLO_INT inst to be last in their instruction group
;CHECK: {{^}}fill3d:
;CHECK-NOT: MULLO_INT T[0-9]+

define void @fill3d(i32 addrspace(1)* nocapture %out) #0 {
entry:
  %x.i = tail call i32 @llvm.r600.read.global.size.x() #1
  %y.i18 = tail call i32 @llvm.r600.read.global.size.y() #1
  %mul = mul i32 %y.i18, %x.i
  %z.i17 = tail call i32 @llvm.r600.read.global.size.z() #1
  %mul3 = mul i32 %mul, %z.i17
  %x.i.i = tail call i32 @llvm.r600.read.tgid.x() #1
  %x.i12.i = tail call i32 @llvm.r600.read.local.size.x() #1
  %mul26.i = mul i32 %x.i12.i, %x.i.i
  %x.i4.i = tail call i32 @llvm.r600.read.tidig.x() #1
  %add.i16 = add i32 %x.i4.i, %mul26.i
  %mul7 = mul i32 %add.i16, %y.i18
  %y.i.i = tail call i32 @llvm.r600.read.tgid.y() #1
  %y.i14.i = tail call i32 @llvm.r600.read.local.size.y() #1
  %mul30.i = mul i32 %y.i14.i, %y.i.i
  %y.i6.i = tail call i32 @llvm.r600.read.tidig.y() #1
  %add.i14 = add i32 %mul30.i, %mul7
  %mul819 = add i32 %add.i14, %y.i6.i
  %add = mul i32 %mul819, %z.i17
  %z.i.i = tail call i32 @llvm.r600.read.tgid.z() #1
  %z.i16.i = tail call i32 @llvm.r600.read.local.size.z() #1
  %mul33.i = mul i32 %z.i16.i, %z.i.i
  %z.i8.i = tail call i32 @llvm.r600.read.tidig.z() #1
  %add.i = add i32 %z.i8.i, %mul33.i
  %add13 = add i32 %add.i, %add
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %add13
  store i32 %mul3, i32 addrspace(1)* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tgid.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.local.size.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.tidig.z() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.global.size.x() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.global.size.y() #1

; Function Attrs: nounwind readnone
declare i32 @llvm.r600.read.global.size.z() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.kernels = !{!0, !1, !2}

!0 = !{null}
!1 = !{null}
!2 = !{void (i32 addrspace(1)*)* @fill3d}
