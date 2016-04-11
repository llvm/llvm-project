; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashcalls -silence-passes > /dev/null
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; REQUIRES: loadable_module

; Bugpoint should keep the call's metadata attached to the call.

; CHECK: call void @foo(), !dbg ![[LOC:[0-9]+]], !attach ![[CALL:[0-9]+]]
; CHECK: ![[LOC]] = !DILocation(line: 104, column: 105, scope: ![[SCOPE:[0-9]+]])
; CHECK: ![[SCOPE]] = distinct !DISubprogram(name: "test"
; CHECK-SAME:                                file: ![[FILE:[0-9]+]]
; CHECK: ![[FILE]] = !DIFile(filename: "source.c", directory: "/dir")
; CHECK: ![[CALL]] = !{!"the call to foo"}

%rust_task = type {}
define void @test(i32* %a, i8* %b) {
    %s = mul i8 22, 9, !attach !0, !dbg !10
    store i8 %s, i8* %b, !attach !1, !dbg !11
    call void @foo(), !attach !2, !dbg !12
    store i32 7, i32* %a, !attach !3, !dbg !13
    %t = add i32 0, 5, !attach !4, !dbg !14
    ret void
}

declare void @foo()

!llvm.module.flags = !{!17}

!0 = !{!"boring"}
!1 = !{!"uninteresting"}
!2 = !{!"the call to foo"}
!3 = !{!"noise"}
!4 = !{!"filler"}

!9 = distinct !DISubprogram(name: "test", file: !15)
!10 = !DILocation(line: 100, column: 101, scope: !9)
!11 = !DILocation(line: 102, column: 103, scope: !9)
!12 = !DILocation(line: 104, column: 105, scope: !9)
!13 = !DILocation(line: 106, column: 107, scope: !9)
!14 = !DILocation(line: 108, column: 109, scope: !9)
!15 = !DIFile(filename: "source.c", directory: "/dir")
!16 = !{}
!17 = !{i32 1, !"Debug Info Version", i32 3}
