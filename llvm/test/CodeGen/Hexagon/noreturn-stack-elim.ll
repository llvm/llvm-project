; RUN: llc -mtriple=hexagon-unknown--elf -hexagon-initial-cfg-cleanup=false < %s | FileCheck %s
; RUN: llc -mtriple=hexagon-unknown--elf -hexagon-initial-cfg-cleanup=false -mattr=+noreturn-stack-elim < %s | FileCheck %s --check-prefix=CHECK-FLAG

; Test the noreturn stack elimination feature. We've added a new flag/feature
; that attempts to eliminate the local stack for noreturn nounwind functions.
; The optimization eliminates the need to save callee saved registers, and
; eliminates the allocframe, when no local stack space is needed.

%struct.A = type { i32, i32 }

; Test the case when noreturn-stack-elim determins that both callee saved
; register do not need to be saved, and the allocframe can be eliminated.

; CHECK-LABEL: test1
; CHECK: memd(r29+#-16) = r17:16
; CHECK: allocframe

; CHECK-FLAG-LABEL: test1
; CHECK-FLAG-NOT: memd(r29+#-16) = r17:16
; CHECK-FLAG-NOT: allocframe

define dso_local void @test1(i32 %a, %struct.A* %b) local_unnamed_addr #0 {
entry:
  %n = getelementptr inbounds %struct.A, %struct.A* %b, i32 0, i32 0
  store i32 %a, i32* %n, align 4
  tail call void @f1() #3
  tail call void @nrf1(%struct.A* %b) #4
  unreachable
}

; Test that noreturn-stack-elim doesn't eliminate the local stack, when
; a function needs to allocate a local variable.

; CHECK-LABEL: test2
; CHECK: allocframe

; CHECK-FLAG-LABEL: test2
; CHECK-FLAG: allocframe

define dso_local void @test2() local_unnamed_addr #0 {
entry:
  %a = alloca i32, align 4
  %0 = bitcast i32* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #4
  call void @f3(i32* nonnull %a) #4
  unreachable
}

; Test that noreturn-stack-elim can elimnate the allocframe when no locals
; are allocated on the stack.

; CHECK-LABEL: test3
; CHECK: allocframe

; CHECK-FLAG-LABEL: test3
; CHECK-FLAG-NOT: allocframe

define dso_local void @test3(i32 %a) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %a, 5
  call void @f2(i32 %add)
  unreachable
}

; Test that nothing is optimized when an alloca is needed for local stack.

; CHECK-LABEL: test4
; CHECK: allocframe

; CHECK-FLAG-LABEL: test4
; CHECK-FLAG: allocframe

define dso_local void @test4(i32 %n) local_unnamed_addr #0 {
entry:
  %vla = alloca i32, i32 %n, align 8
  call void @f3(i32* nonnull %vla) #4
  unreachable
}


declare dso_local void @f1() local_unnamed_addr
declare dso_local void @f2(i32) local_unnamed_addr
declare dso_local void @f3(i32*) local_unnamed_addr

declare dso_local void @nrf1(%struct.A*) local_unnamed_addr #2

declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #5

attributes #0 = { noreturn nounwind }
attributes #2 = { noreturn }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }
attributes #5 = { argmemonly nounwind }

