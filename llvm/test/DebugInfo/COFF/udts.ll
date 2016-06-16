; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc18.0.0"

; C++ source to regenerate:
; $ cat t.cpp
; void f() {
;   typedef int FOO;
;   FOO f;
; }

; CHECK:      ProcStart {
; CHECK:        DisplayName: f
; CHECK:        LinkageName: ?f@@YAXXZ
; CHECK:      }
; CHECK:      UDT {
; CHECK-NEXT:   Type: int (0x74)
; CHECK-NEXT:   UDTName: f::FOO
; CHECK-NEXT: }
; CHECK-NEXT: ProcEnd {
; CHECK-NEXT: }


; Function Attrs: nounwind
define void @"\01?f@@YAXXZ"() #0 !dbg !6 {
entry:
  %f = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %f, metadata !10, metadata !13), !dbg !14
  ret void, !dbg !15
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 (trunk 272079) (llvm/trunk 271895)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "-", directory: "/usr/local/src")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 (trunk 272079) (llvm/trunk 271895)"}
!6 = distinct !DISubprogram(name: "f", linkageName: "\01?f@@YAXXZ", scope: !7, file: !7, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DIFile(filename: "<stdin>", directory: "/usr/local/src")
!8 = !DISubroutineType(types: !9)
!9 = !{null}
!10 = !DILocalVariable(name: "f", scope: !6, file: !7, line: 4, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_typedef, name: "FOO", scope: !6, file: !7, line: 3, baseType: !12)
!12 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!13 = !DIExpression()
!14 = !DILocation(line: 4, column: 5, scope: !6)
!15 = !DILocation(line: 5, column: 1, scope: !6)
