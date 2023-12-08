; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump -v - | FileCheck %s

; C++ source to regenerate:

;typedef char  __attribute__((__aligned__(64))) alchar;

;int main(){
;    alchar newChar;
;}
; $ clang++ -O0 -g -gdwarf-5 debug-info-template-align.cpp -c

; CHECK: .debug_abbrev contents:

; CHECK: [5] DW_TAG_typedef  DW_CHILDREN_no
; CHECK:  DW_AT_alignment DW_FORM_udata

; CHECK: .debug_info contents:

;CHECK: DW_TAG_typedef [5]
;CHECK: DW_AT_name {{.*}} "alchar"
;CHECK-NEXT: DW_AT_alignment [DW_FORM_udata] (64)


; ModuleID = '/dir/test.cpp'
source_filename = "/dir/test.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #0 !dbg !7 {
entry:
  %newChar = alloca i8, align 64
  call void @llvm.dbg.declare(metadata ptr %newChar, metadata !12, metadata !DIExpression()), !dbg !15
  ret i32 0, !dbg !16
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { noinline norecurse nounwind optnone uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 10.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "/dir/test.cpp", directory: "/dir/", checksumkind: CSK_MD5, checksum: "872e252efdfcb9480b4bfaf8437f58ab")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 10.0.0 "}
!7 = distinct !DISubprogram(name: "main", scope: !8, file: !8, line: 12, type: !9, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "test.cpp", directory: "/dir", checksumkind: CSK_MD5, checksum: "872e252efdfcb9480b4bfaf8437f58ab")
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "newChar", scope: !7, file: !8, line: 13, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_typedef, name: "alchar", file: !8, line: 10, baseType: !14, align: 512)
!14 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!15 = !DILocation(line: 13, column: 10, scope: !7)
!16 = !DILocation(line: 14, column: 1, scope: !7)
