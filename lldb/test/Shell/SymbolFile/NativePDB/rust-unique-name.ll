; REQUIRES: system-windows
; RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s
; RUN: lldb-test symbols %t.exe | FileCheck %s

; Output from
; rustc lib.rs -C debuginfo=2 --emit=llvm-ir --crate-type=rlib
; (using rlib to avoid including panic handlers in IR)
;
; lib.rs:
;
; #![no_std]
; mod my_module {
;     #[repr(C)]
;     pub struct MyStruct {
;         pub a: i32,
;         pub b: i32,
;     }
; }
; #[unsafe(no_mangle)]
; extern "C" fn mainCRTStartup() -> my_module::MyStruct {
;     my_module::MyStruct { a: 3, b: 4 }
; }
; #[unsafe(no_mangle)]
; extern "C" fn main() {}

; =======================================================================
; ModuleID = 'lib.b43fc69277defcf4-cgu.0'
source_filename = "lib.b43fc69277defcf4-cgu.0"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; Function Attrs: nounwind uwtable
define i64 @mainCRTStartup() unnamed_addr #0 !dbg !6 {
start:
  %_0 = alloca [8 x i8], align 4
  store i32 3, ptr %_0, align 4, !dbg !20
  %0 = getelementptr inbounds i8, ptr %_0, i64 4, !dbg !20
  store i32 4, ptr %0, align 4, !dbg !20
  %1 = load i64, ptr %_0, align 4, !dbg !21
  ret i64 %1, !dbg !21
}

; Function Attrs: nounwind uwtable
define void @main() unnamed_addr #0 !dbg !22 {
start:
  ret void, !dbg !25
}

attributes #0 = { nounwind uwtable "target-cpu"="x86-64" "target-features"="+cx16,+sse3,+sahf" }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}
!llvm.dbg.cu = !{!4}

!0 = !{i32 8, !"PIC Level", i32 2}
!1 = !{i32 2, !"CodeView", i32 1}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{!"rustc version 1.88.0 (6b00bc388 2025-06-23)"}
!4 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !5, producer: "clang LLVM (rustc version 1.88.0 (6b00bc388 2025-06-23))", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false)
!5 = !DIFile(filename: "src/lib.rs\\@\\lib.b43fc69277defcf4-cgu.0", directory: "F:\\Dev\\testing")
!6 = distinct !DISubprogram(name: "mainCRTStartup", scope: !8, file: !7, line: 12, type: !9, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, templateParams: !19)
!7 = !DIFile(filename: "src/lib.rs", directory: "F:\\Dev\\testing", checksumkind: CSK_SHA256, checksum: "586087c038922a8fa0183c4b20c075445761d545e02d06af80cd5a62dcadb3ec")
!8 = !DINamespace(name: "lib", scope: null)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DICompositeType(tag: DW_TAG_structure_type, name: "MyStruct", scope: !13, file: !12, size: 64, align: 32, flags: DIFlagPublic, elements: !14, templateParams: !19, identifier: "99a3f33b03974e4eaf7224f807a544bf")
!12 = !DIFile(filename: "<unknown>", directory: "")
!13 = !DINamespace(name: "my_module", scope: !8)
!14 = !{!15, !18}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !11, file: !12, baseType: !16, size: 32, align: 32, flags: DIFlagPublic)
!16 = !DIDerivedType(tag: DW_TAG_typedef, name: "i32", file: !12, baseType: !17)
!17 = !DIBasicType(name: "__int32", size: 32, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !11, file: !12, baseType: !16, size: 32, align: 32, offset: 32, flags: DIFlagPublic)
!19 = !{}
!20 = !DILocation(line: 13, scope: !6)
!21 = !DILocation(line: 14, scope: !6)
!22 = distinct !DISubprogram(name: "main", scope: !8, file: !7, line: 17, type: !23, scopeLine: 17, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !4, templateParams: !19)
!23 = !DISubroutineType(types: !24)
!24 = !{null}
!25 = !DILocation(line: 17, scope: !22)

; CHECK:      Type{{.*}} , name = "MyStruct", size = 8, compiler_type = {{.*}} struct MyStruct {
; CHECK-NEXT:         int a;
; CHECK-NEXT:         int b;
; CHECK-NEXT:     }
