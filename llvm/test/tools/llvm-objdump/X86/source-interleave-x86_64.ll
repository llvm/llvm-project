;  REQUIRES: shell
;  RUN: sed -e "s,SRC_COMPDIR,%p/Inputs,g" %s > %t.ll
;  RUN: llc  -o %t.o -filetype=obj -mtriple=x86_64-pc-linux  %t.ll
;  RUN: llvm-objdump -d -l %t.o | FileCheck --check-prefix="LINES" %t.ll
;  RUN: llvm-objdump -d -S %t.o | FileCheck --check-prefix="SOURCE" %t.ll
; ModuleID = 'source-interleave-x86_64.bc'
source_filename = "source-interleave-x86_64.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 1, align 4, !dbg !4

; Function Attrs: nounwind uwtable
define i32 @foo() #0 !dbg !9 {
entry:
  %0 = load i32, i32* @a, align 4, !dbg !12
  ret i32 %0, !dbg !13
}

; Function Attrs: nounwind uwtable
define i32 @main() #0 !dbg !14 {
entry:
  %retval = alloca i32, align 4
  %b = alloca i32*, align 8
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata i32** %b, metadata !15, metadata !17), !dbg !18
  store i32* @a, i32** %b, align 8, !dbg !18
  %0 = load i32*, i32** %b, align 8, !dbg !19
  %1 = load i32, i32* %0, align 4, !dbg !20
  %call = call i32 @foo(), !dbg !21
  %add = add nsw i32 %1, %call, !dbg !22
  ret i32 %add, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7}
!llvm.ident = !{!8}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 4.0.0 (http://llvm.org/git/clang d19a95e94dc57c5a72fd25d64f26134aa7d25fa0) (http://llvm.org/git/llvm.git 313924e6ff8a332063f61d3fda03812c220762f6)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, globals: !3)
!1 = !DIFile(filename: "source-interleave-x86_64.c", directory: "SRC_COMPDIR")
!2 = !{}
!3 = !{!4}
!4 = !DIGlobalVariableExpression(var: !DIGlobalVariable(name: "a", scope: !0, file: !1, line: 1, type: !5, isLocal: false, isDefinition: true), expr: !DIExpression())
!5 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{!"clang version 4.0.0 (http://llvm.org/git/clang d19a95e94dc57c5a72fd25d64f26134aa7d25fa0) (http://llvm.org/git/llvm.git 313924e6ff8a332063f61d3fda03812c220762f6)"}
!9 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 2, type: !10, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: false, unit: !0, variables: !2)
!10 = !DISubroutineType(types: !11)
!11 = !{!5}
!12 = !DILocation(line: 3, column: 10, scope: !9)
!13 = !DILocation(line: 3, column: 3, scope: !9)
!14 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 6, type: !10, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: false, unit: !0, variables: !2)
!15 = !DILocalVariable(name: "b", scope: !14, file: !1, line: 7, type: !16)
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64, align: 64)
!17 = !DIExpression()
!18 = !DILocation(line: 7, column: 8, scope: !14)
!19 = !DILocation(line: 8, column: 11, scope: !14)
!20 = !DILocation(line: 8, column: 10, scope: !14)
!21 = !DILocation(line: 8, column: 15, scope: !14)
!22 = !DILocation(line: 8, column: 13, scope: !14)
!23 = !DILocation(line: 8, column: 3, scope: !14)
; LINES: main:
; LINES-NEXT: ; {{[ -\(\)_A-Za-z0-9.\\/:]+}}source-interleave-x86_64.c:6

; SOURCE: main:
; SOURCE-NEXT: ; int main() {
