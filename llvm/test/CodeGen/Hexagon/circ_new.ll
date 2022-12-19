; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test1
; CHECK: m[[REG1:([0-1])]] = r0
; CHECK: cs[[REG1]] = r1
; CHECK: = memub(r1++#4:circ(m[[REG1]])
define zeroext i8 @test1(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrub.pci(ptr %start, i32 4, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i8
  ret i8 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadrub.pci(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test2
; CHECK: m[[REG2:([0-1])]] = r0
; CHECK: cs[[REG2]] = r1
; CHECK: = memb(r1++#4:circ(m[[REG2]])
define zeroext i8 @test2(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pci(ptr %start, i32 4, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i8
  ret i8 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadrb.pci(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test3
; CHECK: m[[REG3:([0-1])]] = r0
; CHECK: cs[[REG3]] = r1
; CHECK: = memuh(r1++#4:circ(m[[REG3]])
define zeroext i16 @test3(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadruh.pci(ptr %start, i32 4, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i16
  ret i16 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadruh.pci(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test4
; CHECK: m[[REG4:([0-1])]] = r0
; CHECK: cs[[REG4]] = r1
; CHECK: = memh(r1++#4:circ(m[[REG4]])
define signext i16 @test4(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrh.pci(ptr %start, i32 4, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i16
  ret i16 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadrh.pci(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test5
; CHECK: m[[REG5:([0-1])]] = r0
; CHECK: cs[[REG5]] = r1
; CHECK: = memw(r1++#4:circ(m[[REG5]])
define i32 @test5(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadri.pci(ptr %start, i32 4, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  ret i32 %1
}

declare { i32, ptr } @llvm.hexagon.L2.loadri.pci(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test6
; CHECK: m[[REG6:([0-1])]] = r0
; CHECK: cs[[REG6]] = r1
; CHECK: = memd(r1++#8:circ(m[[REG6]])
define i64 @test6(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i64, ptr } @llvm.hexagon.L2.loadrd.pci(ptr %start, i32 8, i32 %mod, ptr %start)
  %1 = extractvalue { i64, ptr } %0, 0
  ret i64 %1
}

declare { i64, ptr } @llvm.hexagon.L2.loadrd.pci(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test7
; CHECK: m[[REG7:([0-1])]] = r0
; CHECK: cs[[REG7]] = r1
; CHECK: = memub(r1++I:circ(m[[REG7]])
define zeroext i8 @test7(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrub.pcr(ptr %start, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i8
  ret i8 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadrub.pcr(ptr, i32, ptr nocapture) #1

; CHECK-LABEL: test8
; CHECK: m[[REG8:([0-1])]] = r0
; CHECK: cs[[REG8]] = r1
; CHECK: = memb(r1++I:circ(m[[REG8]])
define zeroext i8 @test8(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrb.pcr(ptr %start, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i8
  ret i8 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadrb.pcr(ptr, i32, ptr nocapture) #1

; CHECK-LABEL: test9
; CHECK: m[[REG9:([0-1])]] = r0
; CHECK: cs[[REG9]] = r1
; CHECK: = memuh(r1++I:circ(m[[REG9]])
define zeroext i16 @test9(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadruh.pcr(ptr %start, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i16
  ret i16 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadruh.pcr(ptr, i32, ptr nocapture) #1

; CHECK-LABEL: test10
; CHECK: m[[REG10:([0-1])]] = r0
; CHECK: cs[[REG10]] = r1
; CHECK: = memh(r1++I:circ(m[[REG10]])
define signext i16 @test10(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadrh.pcr(ptr %start, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  %conv = trunc i32 %1 to i16
  ret i16 %conv
}

declare { i32, ptr } @llvm.hexagon.L2.loadrh.pcr(ptr, i32, ptr nocapture) #1

; CHECK-LABEL: test11
; CHECK: m[[REG11:([0-1])]] = r0
; CHECK: cs[[REG11]] = r1
; CHECK: = memw(r1++I:circ(m[[REG11]])
define i32 @test11(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i32, ptr } @llvm.hexagon.L2.loadri.pcr(ptr %start, i32 %mod, ptr %start)
  %1 = extractvalue { i32, ptr } %0, 0
  ret i32 %1
}

declare { i32, ptr } @llvm.hexagon.L2.loadri.pcr(ptr, i32, ptr nocapture) #1

; CHECK-LABEL: test12
; CHECK: m[[REG12:([0-1])]] = r0
; CHECK: cs[[REG12]] = r1
; CHECK: = memd(r1++I:circ(m[[REG12]])
define i64 @test12(i32 %mod, ptr %start) local_unnamed_addr #0 {
entry:
  %0 = tail call { i64, ptr } @llvm.hexagon.L2.loadrd.pcr(ptr %start, i32 %mod, ptr %start)
  %1 = extractvalue { i64, ptr } %0, 0
  ret i64 %1
}

declare { i64, ptr } @llvm.hexagon.L2.loadrd.pcr(ptr, i32, ptr nocapture) #1

; CHECK-LABEL: test13
; CHECK: m[[REG13:([0-1])]] = r0
; CHECK: cs[[REG13]] = r1
; CHECK: memb(r1++#4:circ(m[[REG13]])) =
define void @test13(i32 %mod, ptr %start, i8 zeroext %v) local_unnamed_addr #0 {
entry:
  %conv = zext i8 %v to i32
  %0 = tail call ptr @llvm.hexagon.S2.storerb.pci(ptr %start, i32 4, i32 %mod, i32 %conv, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerb.pci(ptr, i32, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test14
; CHECK: m[[REG14:([0-1])]] = r0
; CHECK: cs[[REG14]] = r1
; CHECK: memh(r1++#4:circ(m[[REG14]])) =
define void @test14(i32 %mod, ptr %start, i16 signext %v) local_unnamed_addr #0 {
entry:
  %conv = sext i16 %v to i32
  %0 = tail call ptr @llvm.hexagon.S2.storerh.pci(ptr %start, i32 4, i32 %mod, i32 %conv, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerh.pci(ptr, i32, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test15
; CHECK: m[[REG15:([0-1])]] = r0
; CHECK: cs[[REG15]] = r1
; CHECK: memh(r1++#4:circ(m[[REG15]])) = r{{[0-9]+}}.h
define void @test15(i32 %mod, ptr %start, i16 signext %v) local_unnamed_addr #0 {
entry:
  %conv = sext i16 %v to i32
  %0 = tail call ptr @llvm.hexagon.S2.storerf.pci(ptr %start, i32 4, i32 %mod, i32 %conv, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerf.pci(ptr, i32, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test16
; CHECK: m[[REG16:([0-1])]] = r0
; CHECK: cs[[REG16]] = r1
; CHECK: memw(r1++#4:circ(m[[REG16]])) =
define void @test16(i32 %mod, ptr %start, i32 %v) local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.hexagon.S2.storeri.pci(ptr %start, i32 4, i32 %mod, i32 %v, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storeri.pci(ptr, i32, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test17
; CHECK: m[[REG17:([0-1])]] = r0
; CHECK: cs[[REG17]] = r1
; CHECK: memd(r1++#8:circ(m[[REG17]])) =
define void @test17(i32 %mod, ptr %start, i64 %v) local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.hexagon.S2.storerd.pci(ptr %start, i32 8, i32 %mod, i64 %v, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerd.pci(ptr, i32, i32, i64, ptr nocapture) #1

; CHECK-LABEL: test18
; CHECK: m[[REG18:([0-1])]] = r0
; CHECK: cs[[REG18]] = r1
; CHECK: memb(r1++I:circ(m[[REG18]])) =
define void @test18(i32 %mod, ptr %start, i8 zeroext %v) local_unnamed_addr #0 {
entry:
  %conv = zext i8 %v to i32
  %0 = tail call ptr @llvm.hexagon.S2.storerb.pcr(ptr %start, i32 %mod, i32 %conv, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerb.pcr(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test19
; CHECK: m[[REG19:([0-1])]] = r0
; CHECK: cs[[REG19]] = r1
; CHECK: memh(r1++I:circ(m[[REG19]])) =
define void @test19(i32 %mod, ptr %start, i16 signext %v) local_unnamed_addr #0 {
entry:
  %conv = sext i16 %v to i32
  %0 = tail call ptr @llvm.hexagon.S2.storerh.pcr(ptr %start, i32 %mod, i32 %conv, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerh.pcr(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test20
; CHECK: m[[REG20:([0-1])]] = r0
; CHECK: cs[[REG20]] = r1
; CHECK: memh(r1++I:circ(m[[REG20]])) = r{{[0-9]+}}.h
define void @test20(i32 %mod, ptr %start, i16 signext %v) local_unnamed_addr #0 {
entry:
  %conv = sext i16 %v to i32
  %0 = tail call ptr @llvm.hexagon.S2.storerf.pcr(ptr %start, i32 %mod, i32 %conv, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerf.pcr(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test21
; CHECK: m[[REG21:([0-1])]] = r0
; CHECK: cs[[REG21]] = r1
; CHECK: memw(r1++I:circ(m[[REG21]])) =
define void @test21(i32 %mod, ptr %start, i32 %v) local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.hexagon.S2.storeri.pcr(ptr %start, i32 %mod, i32 %v, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storeri.pcr(ptr, i32, i32, ptr nocapture) #1

; CHECK-LABEL: test22
; CHECK: m[[REG22:([0-1])]] = r0
; CHECK: cs[[REG22]] = r1
; CHECK: memd(r1++I:circ(m[[REG1]])) =
define void @test22(i32 %mod, ptr %start, i64 %v) local_unnamed_addr #0 {
entry:
  %0 = tail call ptr @llvm.hexagon.S2.storerd.pcr(ptr %start, i32 %mod, i64 %v, ptr %start)
  ret void
}

declare ptr @llvm.hexagon.S2.storerd.pcr(ptr, i32, i64, ptr nocapture) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
attributes #1 = { argmemonly nounwind }
