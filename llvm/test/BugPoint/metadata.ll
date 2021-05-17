; REQUIRES: plugins
; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t -bugpoint-crashcalls -silence-passes -disable-namedmd-remove -disable-strip-debuginfo -disable-strip-debug-types > /dev/null
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
;
; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t-nodebug -bugpoint-crashcalls -silence-passes -disable-namedmd-remove > /dev/null
; RUN: llvm-dis %t-nodebug-reduced-simplified.bc -o - | FileCheck %s --check-prefix=NODEBUG
;
; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t-notype -bugpoint-crashcalls -silence-passes -disable-namedmd-remove -disable-strip-debuginfo > /dev/null
; RUN: llvm-dis %t-notype-reduced-simplified.bc -o - | FileCheck %s --check-prefix=NOTYPE
;
; Bugpoint can drop the metadata on the call, as it does not contrinute to the crash.

; CHECK: call void @foo()
; NODEBUG: call void @foo()
; NOTYPE: call void @foo()
; NODEBUG-NOT: call void @foo()
; NOTYPE-NOT: !DIBasicType
; NOTYPE: !DICompileUnit
; NOTYPE-NOT: !DIBasicType

%rust_task = type {}
define void @test(i32* %a, i8* %b) !dbg !9 {
    %s = mul i8 22, 9, !attach !0, !dbg !10
    store i8 %s, i8* %b, !attach !1, !dbg !11
    call void @foo(), !attach !2, !dbg !12
    store i32 7, i32* %a, !attach !3, !dbg !13
    %t = add i32 0, 5, !attach !4, !dbg !14
    ret void
}

declare void @foo()

!llvm.module.flags = !{!17}
!llvm.dbg.cu = !{!8}

!0 = !{!"boring"}
!1 = !{!"uninteresting"}
!2 = !{!"the call to foo"}
!3 = !{!"noise"}
!4 = !{!"filler"}

!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !15)
!9 = distinct !DISubprogram(name: "test", file: !15, type: !18, unit: !8)
!10 = !DILocation(line: 100, column: 101, scope: !9)
!11 = !DILocation(line: 102, column: 103, scope: !9)
!12 = !DILocation(line: 104, column: 105, scope: !9)
!13 = !DILocation(line: 106, column: 107, scope: !9)
!14 = !DILocation(line: 108, column: 109, scope: !9)
!15 = !DIFile(filename: "source.c", directory: "/dir")
!16 = !{}
!17 = !{i32 1, !"Debug Info Version", i32 3}
!18 = !DISubroutineType(types: !19)
!19 = !{!20, !20}
!20 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
