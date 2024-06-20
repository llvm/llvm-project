; RUN: opt -passes=bpf-preserve-static-offset -mtriple=bpf-pc-linux -S -o - %s 2>&1 | FileCheck %s
;
; If load offset is not a constant bpf-preserve-static-offset should report a
; warning and remove preserve.static.offset call.
;
; Source:
;    #define __ctx __attribute__((preserve_static_offset))
;    
;    struct foo {
;      int a[7];
;    } __ctx;
;    
;    extern void consume(int);
;    
;    void bar(struct foo *p, unsigned long i) {
;      consume(p->a[i]);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -debug-info-kind=line-tables-only -triple bpf \
;         -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

; CHECK:      warning: some-file.c:10:11: in function bar void (ptr, i64):
; CHECK-SAME: Non-constant offset in access to a field of a type marked with
; CHECK-SAME: preserve_static_offset might be rejected by BPF verifier

%struct.foo = type { [7 x i32] }

; Function Attrs: nounwind
define dso_local void @bar(ptr noundef %p, i64 noundef %i) #0 !dbg !5 {
entry:
  %0 = call ptr @llvm.preserve.static.offset(ptr %p), !dbg !8
  %a = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 0, !dbg !8
  %arrayidx = getelementptr inbounds [7 x i32], ptr %a, i64 0, i64 %i, !dbg !9
  %1 = load i32, ptr %arrayidx, align 4, !dbg !9, !tbaa !10
  call void @consume(i32 noundef %1), !dbg !14
  ret void, !dbg !15
}

; CHECK:      define dso_local void @bar(ptr noundef %[[p:.*]], i64 noundef %[[i:.*]])
; CHECK:        %[[a:.*]] = getelementptr inbounds %struct.foo, ptr %[[p]], i32 0, i32 0, !dbg
; CHECK-NEXT:   %[[arrayidx:.*]] = getelementptr inbounds [7 x i32], ptr %[[a]], i64 0, i64 %[[i]], !dbg
; CHECK-NEXT:   %[[v5:.*]] = load i32, ptr %[[arrayidx]], align 4, !dbg {{.*}}, !tbaa
; CHECK-NEXT:   call void @consume(i32 noundef %[[v5]]), !dbg

declare void @consume(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.preserve.static.offset(ptr readnone) #2

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"clang"}
!5 = distinct !DISubprogram(name: "bar", scope: !1, file: !1, line: 9, type: !6, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DISubroutineType(types: !7)
!7 = !{}
!8 = !DILocation(line: 10, column: 14, scope: !5)
!9 = !DILocation(line: 10, column: 11, scope: !5)
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !12, i64 0}
!12 = !{!"omnipotent char", !13, i64 0}
!13 = !{!"Simple C/C++ TBAA"}
!14 = !DILocation(line: 10, column: 3, scope: !5)
!15 = !DILocation(line: 11, column: 1, scope: !5)
