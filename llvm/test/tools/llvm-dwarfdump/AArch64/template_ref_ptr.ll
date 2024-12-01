; RUN: llc -O0 %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump --verify %t.o

; $ cat test.cpp
; int glbl = 42;
; constexpr int &r = glbl;
; constexpr int *p = &glbl;
; template<int &ref>
; int x;
; template<int *ref>
; int y;
; int main() {
;   x<r> = 3;
;   y<p> = 3;
; }

source_filename = "test.cpp"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx15.0.0"

@glbl = global i32 42, align 4, !dbg !0
@r = constant ptr @glbl, align 8, !dbg !5
@_Z1xIL_Z4glblEE = linkonce_odr global i32 0, align 4, !dbg !9
@_Z1yIXadL_Z4glblEEE = linkonce_odr global i32 0, align 4, !dbg !13

; Function Attrs: mustprogress noinline norecurse nounwind optnone ssp uwtable(sync)
define noundef i32 @main() #0 !dbg !26 {
entry:
  store i32 3, ptr @_Z1xIL_Z4glblEE, align 4, !dbg !29
  store i32 3, ptr @_Z1yIXadL_Z4glblEEE, align 4, !dbg !30
  ret i32 0, !dbg !31
}

attributes #0 = { mustprogress noinline norecurse nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+complxnum,+crc,+dotprod,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!18, !19, !20, !21, !22, !23, !24}
!llvm.dbg.cu = !{!2}
!llvm.linker.options = !{}
!llvm.ident = !{!25}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "glbl", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 19.0.0git (git@github.com:llvm/llvm-project.git cf311a1131b9aef3e66b2a20ad49cfc77212754b)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: Apple, sysroot: "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.0.sdk", sdk: "MacOSX15.0.sdk")
!3 = !DIFile(filename: "test.cpp", directory: "/tmp", checksumkind: CSK_MD5, checksum: "1aea094a645b8409da5183136c5297f0")
!4 = !{!0, !5, !9, !13}
!5 = !DIGlobalVariableExpression(var: !6, expr: !DIExpression())
!6 = distinct !DIGlobalVariable(name: "r", scope: !2, file: !3, line: 2, type: !7, isLocal: false, isDefinition: true)
!7 = !DIDerivedType(tag: DW_TAG_reference_type, baseType: !8, size: 64)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "x", linkageName: "_Z1xIL_Z4glblEE", scope: !2, file: !3, line: 5, type: !8, isLocal: false, isDefinition: true, templateParams: !11)
!11 = !{!12}
!12 = !DITemplateValueParameter(name: "ref", type: !7, value: ptr @glbl)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "y", linkageName: "_Z1yIXadL_Z4glblEEE", scope: !2, file: !3, line: 7, type: !8, isLocal: false, isDefinition: true, templateParams: !15)
!15 = !{!16}
!16 = !DITemplateValueParameter(name: "ref", type: !17, value: ptr @glbl)
!17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 64)
!18 = !{i32 2, !"SDK Version", [2 x i32] [i32 15, i32 0]}
!19 = !{i32 7, !"Dwarf Version", i32 5}
!20 = !{i32 2, !"Debug Info Version", i32 3}
!21 = !{i32 1, !"wchar_size", i32 4}
!22 = !{i32 8, !"PIC Level", i32 2}
!23 = !{i32 7, !"uwtable", i32 1}
!24 = !{i32 7, !"frame-pointer", i32 1}
!25 = !{!"clang version 19.0.0git (git@github.com:llvm/llvm-project.git cf311a1131b9aef3e66b2a20ad49cfc77212754b)"}
!26 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 8, type: !27, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!27 = !DISubroutineType(types: !28)
!28 = !{!8}
!29 = !DILocation(line: 9, column: 8, scope: !26)
!30 = !DILocation(line: 10, column: 8, scope: !26)
!31 = !DILocation(line: 11, column: 1, scope: !26)
