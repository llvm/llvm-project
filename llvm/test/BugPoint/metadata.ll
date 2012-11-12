; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashcalls -silence-passes > /dev/null
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; REQUIRES: loadable_module
; XFAIL: lto_on_osx

; Bugpoint should keep the call's metadata attached to the call.

; CHECK: call void @foo(), !dbg !0, !attach !2
; CHECK: !0 = metadata !{i32 104, i32 105, metadata !1, metadata !1}
; CHECK: !1 = metadata !{i32 0, i32 0, i32 0, metadata !"source.c", metadata !"/dir", metadata !"me", i1 true, i1 false, metadata !"", i32 0}
; CHECK: !2 = metadata !{metadata !"the call to foo"}

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

!0 = metadata !{metadata !"boring"}
!1 = metadata !{metadata !"uninteresting"}
!2 = metadata !{metadata !"the call to foo"}
!3 = metadata !{metadata !"noise"}
!4 = metadata !{metadata !"filler"}

!9 = metadata !{i32 0, i32 0, i32 0, metadata !"source.c", metadata !"/dir", metadata !"me", i1 true, i1 false, metadata !"", i32 0}
!10 = metadata !{i32 100, i32 101, metadata !9, metadata !9}
!11 = metadata !{i32 102, i32 103, metadata !9, metadata !9}
!12 = metadata !{i32 104, i32 105, metadata !9, metadata !9}
!13 = metadata !{i32 106, i32 107, metadata !9, metadata !9}
!14 = metadata !{i32 108, i32 109, metadata !9, metadata !9}
