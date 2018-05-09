; RUN: llc < %s | FileCheck %s
; FIXME: fastisel screws up the order here.
; RUNX: llc -O0 < %s | FileCheck %s

; Source to regenerate:
; $ clang --target=x86_64-windows-msvc -S annotation.c  -g -gcodeview -o t.ll \
;      -emit-llvm -O1 -Xclang -disable-llvm-passes -fms-extensions
; void g(void);
; void f(void) {
;   g();
;   __annotation(L"a1", L"a2");
;   g();
; }

; ModuleID = 'annotation.c'
source_filename = "annotation.c"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

; Function Attrs: nounwind uwtable
define void @f() #0 !dbg !8 {
entry:
  call void @g(), !dbg !11
  call void @llvm.codeview.annotation(metadata !12), !dbg !13
  call void @g(), !dbg !14
  ret void, !dbg !15
}

; CHECK-LABEL: f: # @f
; CHECK: callq g
; CHECK: .Lannotation0:
; CHECK: callq g
; CHECK: retq

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4121                    # Record kind: S_ANNOTATION
; CHECK-NEXT:  .secrel32       .Lannotation0
; CHECK-NEXT:  .secidx .Lannotation0
; CHECK-NEXT:  .short  2
; CHECK-NEXT:  .asciz  "a1"
; CHECK-NEXT:  .asciz  "a2"

; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END

declare void @g() #1

; Function Attrs: nounwind
declare void @llvm.codeview.annotation(metadata) #2

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "annotation.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "51164221112d8a5baa55a995027e4ba5")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 2, type: !9, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 3, column: 3, scope: !8)
!12 = !{!"a1", !"a2"}
!13 = !DILocation(line: 4, column: 3, scope: !8)
!14 = !DILocation(line: 5, column: 3, scope: !8)
!15 = !DILocation(line: 6, column: 1, scope: !8)
