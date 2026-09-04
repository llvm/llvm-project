; RUN: opt < %s -passes=pgo-icall-prom -enable-vtable-profile-use -icp-samplepgo -S | FileCheck %s

; Test that when vtable-based indirect call promotion introduces an address
; comparison against a vtable address point, unnamed_addr is stripped from the
; vtable so that it is address-significant.

; CHECK: @Derived1 = constant [3 x ptr]
@Derived1 = unnamed_addr constant [3 x ptr] [ptr null, ptr null, ptr @Derived1_bar], !type !1

; CHECK: @Unrelated = unnamed_addr constant ptr null
@Unrelated = unnamed_addr constant ptr null

; CHECK-LABEL: define void @test(
; CHECK:         %[[CMP:.*]] = icmp eq ptr %vtable, getelementptr inbounds (i8, ptr @Derived1, i32 8)
; CHECK-NEXT:    br i1 %[[CMP]], label %[[THEN:.*]], label %[[ELSE:.*]]
; CHECK:       [[THEN]]:
; CHECK-NEXT:    call void @Derived1_bar()
define void @test(ptr %d) {
entry:
  %vtable = load ptr, ptr %d, !prof !2
  %0 = call i1 @llvm.type.test(ptr %vtable, metadata !"Base1")
  call void @llvm.assume(i1 %0)
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 1
  %1 = load ptr, ptr %vfn
  call void %1(), !prof !3
  ret void
}

define void @Derived1_bar() {
  ret void
}

declare i1 @llvm.type.test(ptr, metadata)
declare void @llvm.assume(i1)

!1 = !{i64 8, !"Base1"}
!2 = !{!"VP", i32 2, i64 1600, i64 -4123858694673519054, i64 1600}
!3 = !{!"VP", i32 0, i64 1600, i64 3827408714133779784, i64 1600}
