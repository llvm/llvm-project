; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/indirect-call.prof -S | FileCheck %s

; CHECK-LABEL: @test
define void @test(void ()*) !dbg !3 {
  %2 = alloca void ()*
  store void ()* %0, void ()** %2
  %3 = load void ()*, void ()** %2
  ; CHECK: call {{.*}}, !prof ![[PROF:[0-9]+]]
  call void %3(), !dbg !4
  ret void
}

; CHECK-LABEL: @test_inline
; If the indirect call is promoted and inlined in profile, we should promote and inline it.
define void @test_inline(i64* (i32*)*, i32* %x) !dbg !3 {
  %2 = alloca i64* (i32*)*
  store i64* (i32*)* %0, i64* (i32*)** %2
  %3 = load i64* (i32*)*, i64* (i32*)** %2
; CHECK: icmp {{.*}} @foo_inline2
; CHECK: if.true.direct_targ:
; CHECK-NOT: call
; CHECK: if.false.orig_indirect:
; CHECK: icmp {{.*}} @foo_inline1
; CHECK: if.true.direct_targ1:
; CHECK-NOT: call
; CHECK: if.false.orig_indirect2:
; CHECK: call
  call i64* %3(i32* %x), !dbg !5
  ret void
}

; CHECK-LABEL: @test_noinline
; If the indirect call target is not available, we should not promote it.
define void @test_noinline(void ()*) !dbg !3 {
  %2 = alloca void ()*
  store void ()* %0, void ()** %2
  %3 = load void ()*, void ()** %2
; CHECK-NOT: icmp
; CHECK: call
  call void %3(), !dbg !5
  ret void
}

@x = global i32 0, align 4

define i32* @foo_inline1(i32* %x) !dbg !3 {
  ret i32* %x
}

define i32* @foo_inline2(i32* %x) !dbg !3 {
  ret i32* %x
}

define i32 @foo_noinline(i32 %x) !dbg !3 {
  ret i32 %x
}

define void @foo_direct() !dbg !3 {
  ret void
}

; CHECK-LABEL: @test_direct
; We should not promote a direct call.
define void @test_direct() !dbg !3 {
; CHECK-NOT: icmp
; CHECK: call
  call void @foo_alias(), !dbg !5
  ret void
}

@foo_alias = alias void (), void ()* @foo_direct

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "test.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, unit: !0)
!4 = !DILocation(line: 5, scope: !3)
!5 = !DILocation(line: 6, scope: !3)
; CHECK: ![[PROF]] = !{!"VP", i32 0, i64 3457, i64 9191153033785521275, i64 2059, i64 -1069303473483922844, i64 1398}
