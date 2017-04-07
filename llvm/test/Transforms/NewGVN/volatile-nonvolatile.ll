; RUN: opt -tbaa -newgvn -S < %s | FileCheck %s

%struct.t = type { i32* }

; The loaded address and the location of the address itself are not aliased,
; so the second reload is not necessary. Check that it can be eliminated.
; CHECK-LABEL: test1
; CHECK: load
; CHECK-NOT: load
define void @test1(%struct.t* nocapture readonly %p, i32 %v) #0 {
entry:
  %m = getelementptr inbounds %struct.t, %struct.t* %p, i32 0, i32 0
  %0 = load i32*, i32** %m, align 4, !tbaa !1
  store volatile i32 %v, i32* %0, align 4, !tbaa !6
  %1 = load i32*, i32** %m, align 4, !tbaa !1
  store volatile i32 %v, i32* %1, align 4, !tbaa !6
  ret void
}

; The store via the loaded address may overwrite the address itself.
; Make sure that both loads remain.
; CHECK-LABEL: test2
; CHECK: load
; CHECK: store
; CHECK: load
define void @test2(%struct.t* nocapture readonly %p, i32 %v) #0 {
entry:
  %m = getelementptr inbounds %struct.t, %struct.t* %p, i32 0, i32 0
  %0 = load i32*, i32** %m, align 4, !tbaa !1
  store volatile i32 %v, i32* %0, align 4, !tbaa !1
  %1 = load i32*, i32** %m, align 4, !tbaa !1
  store volatile i32 %v, i32* %1, align 4, !tbaa !1
  ret void
}

; The loads are ordered and non-monotonic. Although they are not aliased to
; the stores, make sure both are preserved.
; CHECK-LABEL: test3
; CHECK: load
; CHECK: store
; CHECK: load
define void @test3(%struct.t* nocapture readonly %p, i32 %v) #0 {
entry:
  %m = getelementptr inbounds %struct.t, %struct.t* %p, i32 0, i32 0
  %0 = load atomic i32*, i32** %m acquire, align 4, !tbaa !1
  store volatile i32 %v, i32* %0, align 4, !tbaa !6
  %1 = load atomic i32*, i32** %m acquire, align 4, !tbaa !1
  store volatile i32 %v, i32* %1, align 4, !tbaa !6
  ret void
}

attributes #0 = { norecurse nounwind }

!1 = !{!2, !3, i64 0}
!2 = !{!"", !3, i64 0}
!3 = !{!"any pointer", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !4, i64 0}

