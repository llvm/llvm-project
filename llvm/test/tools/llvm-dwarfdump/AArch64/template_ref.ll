; RUN: llc -O0 %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump --verify %t.o

; $ cat ref.cpp
; int glbl = 42;
; int &r = glbl;
; template<int &ref>
; int x;
; int main() {
;   x<r> = 3;
; }

source_filename = "ref.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx15.0.0"

@glbl = global i32 42, align 4, !dbg !0
@r = constant ptr @glbl, align 8, !dbg !5
@_Z1xIL_Z4glblEE = linkonce_odr global i32 0, align 4, !dbg !9

; Function Attrs: mustprogress noinline norecurse nounwind optnone ssp uwtable(sync)
define noundef i32 @main() #0 !dbg !19 {
entry:
  store i32 3, ptr @_Z1xIL_Z4glblEE, align 4, !dbg !22
  ret i32 0, !dbg !23
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!13, !14, !15, !16, !17, !18}
!llvm.dbg.cu = !{!2}
!llvm.linker.options = !{}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glbl", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.0.sdk", sdk: "MacOSX15.0.sdk")
!3 = !DIFile(filename: "ref.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "b7b99c5692b757c3b50d0e841713927c")
!4 = !{!0, !5, !9}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "r", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "x", linkageName: "_Z1xIL_Z4glblEE", scope: !2, file: !3, line: 4, type: !8, isLocal: false, isDefinition: true, templateParams: !11)
!11 = !{!12}
!12 = !DITemplateValueParameter(name: "ref", type: !7, value: ptr @glbl)
!13 = !{i32 7, !"Dwarf Version", i32 5}
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 1, !"wchar_size", i32 4}
!16 = !{i32 8, !"PIC Level", i32 2}
!17 = !{i32 7, !"uwtable", i32 1}
!18 = !{i32 7, !"frame-pointer", i32 1}
!19 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 5, type: !20, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!20 = !DISubroutineType(types: !21)
!21 = !{!8}
!22 = !DILocation(line: 6, column: 8, scope: !19)
!23 = !DILocation(line: 7, column: 1, scope: !19)
