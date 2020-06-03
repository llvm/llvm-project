; RUN: opt < %s -sample-profile -sample-profile-file=%S/Inputs/indirect-call.afdo -S | FileCheck %s

; Checks if indirect call targets are read correctly when reading from gcc
; format profile.
; It is expected to fail on certain architectures as gcc profile reader does
; not work.
; XFAIL: host-byteorder-big-endian

define void @test(void ()*) #0 !dbg !3 {
  %2 = alloca void ()*
  store void ()* %0, void ()** %2
  %3 = load void ()*, void ()** %2
  ; CHECK: call {{.*}}, !prof ![[PROF:[0-9]+]]
  call void %3(), !dbg !4
  ret void
}

attributes #0 = {"use-sample-profile"}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1)
!1 = !DIFile(filename: "test.cc", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, unit: !0)
!4 = !DILocation(line: 5, scope: !3)
; CHECK: ![[PROF]] = !{!"VP", i32 0, i64 3457, i64 9191153033785521275, i64 2059, i64 -1069303473483922844, i64 1398}
