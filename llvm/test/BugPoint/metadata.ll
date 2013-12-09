; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashcalls -silence-passes > /dev/null
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; REQUIRES: loadable_module

; Bugpoint should keep the call's metadata attached to the call.

; CHECK: call void @foo(), !dbg ![[LOC:[0-9]+]], !attach ![[CALL:[0-9]+]]
; CHECK: ![[LOC]] = metadata !{i32 104, i32 105, metadata ![[SCOPE:[0-9]+]], metadata ![[SCOPE]]}
; CHECK: ![[SCOPE]] = metadata !{i32 458769, metadata ![[FILE:[0-9]+]], i32 0, metadata !"me", i1 true, metadata !"", i32 0, metadata ![[LIST:[0-9]+]], metadata ![[LIST]], null, null, null, metadata !""}
; CHECK: ![[FILE]] = metadata !{metadata !"source.c", metadata !"/dir"}
; CHECK: ![[LIST]] = metadata !{i32 0}
; CHECK: ![[CALL]] = metadata !{metadata !"the call to foo"}

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

!0 = metadata !{metadata !"boring"}
!1 = metadata !{metadata !"uninteresting"}
!2 = metadata !{metadata !"the call to foo"}
!3 = metadata !{metadata !"noise"}
!4 = metadata !{metadata !"filler"}

!9 = metadata !{i32 458769, metadata !15, i32 0, metadata !"me", i1 true, metadata !"", i32 0, metadata !16, metadata !16, null, null, null, metadata !""}
!10 = metadata !{i32 100, i32 101, metadata !9, metadata !9}
!11 = metadata !{i32 102, i32 103, metadata !9, metadata !9}
!12 = metadata !{i32 104, i32 105, metadata !9, metadata !9}
!13 = metadata !{i32 106, i32 107, metadata !9, metadata !9}
!14 = metadata !{i32 108, i32 109, metadata !9, metadata !9}
!15 = metadata !{metadata !"source.c", metadata !"/dir"}
!16 = metadata !{i32 0}
!17 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
