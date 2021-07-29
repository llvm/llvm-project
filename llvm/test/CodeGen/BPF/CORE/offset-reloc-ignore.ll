; RUN: opt -O2 %s | llvm-dis > %t1
; RUN: llc -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; RUN: llc -mattr=+alu32 -filetype=asm -o - %t1 | FileCheck -check-prefixes=CHECK %s
; Source code:
;   #define _(x) (__builtin_preserve_access_index(x))
;   int get_value(const int *arg);
;   int test(int *arg) {
;     return get_value(_(&arg[4]));
;   }
; Compilation flag:
;   clang -target bpf -O2 -g -S -emit-llvm -Xclang -disable-llvm-passes test.c

target triple = "bpf"

; Function Attrs: nounwind
define dso_local i32 @test(i32* %arg) local_unnamed_addr #0 !dbg !10 {
entry:
  call void @llvm.dbg.value(metadata i32* %arg, metadata !14, metadata !DIExpression()), !dbg !15
  %0 = tail call i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32* elementtype(i32) %arg, i32 0, i32 4), !dbg !16, !llvm.preserve.access.index !4
  %call = tail call i32 @get_value(i32* %0) #4, !dbg !17
  ret i32 %call, !dbg !18
}

; CHECK:             r1 += 16
; CHECK:             call get_value
; CHECK:             .section        .BTF.ext,"",@progbits
; CHECK-NOT:         .long   16                      # FieldReloc

declare dso_local i32 @get_value(i32*) local_unnamed_addr #1

; Function Attrs: nounwind readnone
declare i32* @llvm.preserve.array.access.index.p0i32.p0i32(i32*, i32, i32) #2

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #3

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind readnone speculatable willreturn }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp/home/yhs/work/tests/llvm/cast")
!2 = !{}
!3 = !{!4}
!4 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 64)
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 2, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 10.0.0 (trunk 367256) (llvm/trunk 367266)"}
!10 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 3, type: !11, scopeLine: 3, flags: DIFlagPrototyped, isDefinition: true, isOptimized: true, unit: !0, retainedNodes: !13)
!11 = !DISubroutineType(types: !12)
!12 = !{!5, !4}
!13 = !{!14}
!14 = !DILocalVariable(name: "arg", arg: 1, scope: !10, file: !1, line: 3, type: !4)
!15 = !DILocation(line: 0, scope: !10)
!16 = !DILocation(line: 4, column: 20, scope: !10)
!17 = !DILocation(line: 4, column: 10, scope: !10)
!18 = !DILocation(line: 4, column: 3, scope: !10)
