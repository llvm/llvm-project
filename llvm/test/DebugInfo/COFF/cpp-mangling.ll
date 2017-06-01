; RUN: llc -mcpu=core2 -mtriple=i686-pc-win32 -o - -O0 -filetype=obj < %s \
; RUN:   | llvm-readobj -codeview | FileCheck %s

; C++ source to regenerate:
; namespace foo {
; int bar(int x) { return x * 2; }
; }
; template <typename T, int (*)(int)>
; void fn_tmpl() {}
; template void fn_tmpl<int, foo::bar>();
; void f() {
;   fn_tmpl<int, foo::bar>();
; }

; CHECK:        ProcStart {
; CHECK:         FunctionType: bar ({{.*}})
; CHECK:         DisplayName: foo::bar{{$}}
; CHECK-NEXT:    LinkageName: ?bar@foo@@YAHH@Z

; CHECK:        ProcStart {
; CHECK:         FunctionType: fn_tmpl ({{.*}})
; CHECK:         DisplayName: foo::fn_tmpl<int,&foo::bar>
; CHECK-NEXT:    LinkageName: ??$fn_tmpl@H$1?bar@foo@@YAHH@Z@foo@@YAXXZ

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.0.23918"

$"\01??$fn_tmpl@H$1?bar@foo@@YAHH@Z@foo@@YAXXZ" = comdat any

; Function Attrs: nounwind
define i32 @"\01?bar@foo@@YAHH@Z"(i32 %x) #0 !dbg !6 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !11, metadata !12), !dbg !13
  %0 = load i32, i32* %x.addr, align 4, !dbg !14
  %mul = mul nsw i32 %0, 2, !dbg !15
  ret i32 %mul, !dbg !16
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
define weak_odr void @"\01??$fn_tmpl@H$1?bar@foo@@YAHH@Z@foo@@YAXXZ"() #0 comdat !dbg !17 {
entry:
  ret void, !dbg !24
}

attributes #0 = { nounwind "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!llvm.ident = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.9.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.cpp", directory: "D:\5Csrc\5Cllvm\5Cbuild")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{!"clang version 3.9.0 "}
!6 = distinct !DISubprogram(name: "bar", linkageName: "\01?bar@foo@@YAHH@Z", scope: !7, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!7 = !DINamespace(name: "foo", scope: null)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !10}
!10 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !DILocalVariable(name: "x", arg: 1, scope: !6, file: !1, line: 2, type: !10)
!12 = !DIExpression()
!13 = !DILocation(line: 2, column: 13, scope: !6)
!14 = !DILocation(line: 2, column: 25, scope: !6)
!15 = !DILocation(line: 2, column: 27, scope: !6)
!16 = !DILocation(line: 2, column: 18, scope: !6)
!17 = distinct !DISubprogram(name: "fn_tmpl<int,&foo::bar>", linkageName: "\01??$fn_tmpl@H$1?bar@foo@@YAHH@Z@foo@@YAXXZ", scope: !7, file: !1, line: 4, type: !18, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, templateParams: !20, variables: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !{!21, !22}
!21 = !DITemplateTypeParameter(name: "T", type: !10)
!22 = !DITemplateValueParameter(type: !23, value: i32 (i32)* @"\01?bar@foo@@YAHH@Z")
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32, align: 32)
!24 = !DILocation(line: 4, column: 17, scope: !17)
