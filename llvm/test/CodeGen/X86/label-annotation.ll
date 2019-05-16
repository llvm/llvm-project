; RUN: llc -mtriple=x86_64-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-windows-msvc -O0 < %s | FileCheck %s
; RUN: llc -mtriple=i686-windows-msvc < %s | FileCheck %s
; RUN: llc -mtriple=i686-windows-msvc -O0 < %s | FileCheck %s

; Source to regenerate:
; $ clang --target=x86_64-windows-msvc -S annotation.c  -g -gcodeview -o t.ll \
;      -emit-llvm -O1 -Xclang -disable-llvm-passes -fms-extensions
; void g(void);
; void f(void) {
;   g();
;   __annotation(L"a1", L"a2");
;   g();
; }
; void trapit() {
;   __annotation(L"foo", L"bar");
;   asm volatile ("int $0x2C");
;   __builtin_unreachable();
; }

; Function Attrs: nounwind uwtable
define dso_local void @f() #0 !dbg !8 {
entry:
  call void @g(), !dbg !11
  call void @llvm.codeview.annotation(metadata !12), !dbg !13
  call void @g(), !dbg !14
  ret void, !dbg !15
}

declare dso_local void @g() #1


; CHECK-LABEL: {{_?f: # @f}}
; CHECK: {{call[ql] _?g}}
; CHECK: {{\.?}}Lannotation0:
; CHECK: {{call[ql] _?g}}
; CHECK: {{ret[ql]}}


; Function Attrs: inaccessiblememonly noduplicate nounwind
declare void @llvm.codeview.annotation(metadata) #2

; Function Attrs: nounwind uwtable
define dso_local void @trapit() #0 !dbg !16 {
entry:
  call void @llvm.codeview.annotation(metadata !17), !dbg !18
  call void asm sideeffect "int $$0x2C", "~{dirflag},~{fpsr},~{flags}"() #3, !dbg !19, !srcloc !20
  unreachable, !dbg !21
}


; CHECK-LABEL: {{_?trapit: # @trapit}}
; CHECK: {{\.?}}Lannotation1:
; CHECK: int $44


; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4121                    # Record kind: S_ANNOTATION
; CHECK-NEXT:  .secrel32       {{\.?}}Lannotation0
; CHECK-NEXT:  .secidx {{\.?}}Lannotation0
; CHECK-NEXT:  .short  2
; CHECK-NEXT:  .asciz  "a1"
; CHECK-NEXT:  .asciz  "a2"

; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END


; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4121                    # Record kind: S_ANNOTATION
; CHECK-NEXT:  .secrel32       {{\.?}}Lannotation1
; CHECK-NEXT:  .secidx {{\.?}}Lannotation1
; CHECK-NEXT:  .short  2
; CHECK-NEXT:  .asciz  "foo"
; CHECK-NEXT:  .asciz  "bar"

; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END



attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { inaccessiblememonly noduplicate nounwind }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (git@github.com:llvm/llvm-project.git 7f9a008a2db285aca57bfa0c09858c9527a7aa98)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "066b1dde2a08455e4d345baa8a920b56")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 9.0.0 (git@github.com:llvm/llvm-project.git 7f9a008a2db285aca57bfa0c09858c9527a7aa98)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !9, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 3, scope: !8)
!12 = !{!"a1", !"a2"}
!13 = !DILocation(line: 4, scope: !8)
!14 = !DILocation(line: 5, scope: !8)
!15 = !DILocation(line: 6, scope: !8)
!16 = distinct !DISubprogram(name: "trapit", scope: !1, file: !1, line: 7, type: !9, scopeLine: 7, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!17 = !{!"foo", !"bar"}
!18 = !DILocation(line: 8, scope: !16)
!19 = !DILocation(line: 9, scope: !16)
!20 = !{i32 149}
!21 = !DILocation(line: 10, scope: !16)
