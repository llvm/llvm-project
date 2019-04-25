; RUN: llc -O0 < %s | FileCheck %s
; FIXME: Add test for llc with optimizations once it is implemented.

; Source to regenerate:
; $ clang --target=x86_64-windows-msvc -S heapallocsite.c  -g -gcodeview -o t.ll \
;      -emit-llvm -O0 -Xclang -disable-llvm-passes -fms-extensions
; __declspec(allocator) char *myalloc(void);
; void f() {
;   myalloc()
; }
;
; struct Foo {
;   __declspec(allocator) virtual void *alloc();
; };
; void use_alloc(void*);
; void do_alloc(Foo *p) {
;   use_alloc(p->alloc());
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.Foo = type { i32 (...)** }

; Function Attrs: noinline optnone uwtable
define dso_local void @f() #0 !dbg !8 {
entry:
  %call = call i8* @myalloc(), !dbg !11, !heapallocsite !2
  ret void, !dbg !12
}

; CHECK-LABEL: f: # @f
; CHECK: .Lheapallocsite0:
; CHECK: callq myalloc
; CHECK: .Lheapallocsite1:
; CHECK: retq

declare dso_local i8* @myalloc() #1

; Function Attrs: noinline optnone uwtable
define dso_local void @do_alloc(%struct.Foo* %p) #0 !dbg !13 {
entry:
  %p.addr = alloca %struct.Foo*, align 8
  store %struct.Foo* %p, %struct.Foo** %p.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.Foo** %p.addr, metadata !18, metadata !DIExpression()), !dbg !19
  %0 = load %struct.Foo*, %struct.Foo** %p.addr, align 8, !dbg !20
  %1 = bitcast %struct.Foo* %0 to i8* (%struct.Foo*)***, !dbg !20
  %vtable = load i8* (%struct.Foo*)**, i8* (%struct.Foo*)*** %1, align 8, !dbg !20
  %vfn = getelementptr inbounds i8* (%struct.Foo*)*, i8* (%struct.Foo*)** %vtable, i64 0, !dbg !20
  %2 = load i8* (%struct.Foo*)*, i8* (%struct.Foo*)** %vfn, align 8, !dbg !20
  %call = call i8* %2(%struct.Foo* %0), !dbg !20, !heapallocsite !2
  call void @use_alloc(i8* %call), !dbg !20
  ret void, !dbg !21
}

; CHECK-LABEL: do_alloc: # @do_alloc
; CHECK: .Lheapallocsite2:
; CHECK: callq *(%rax)
; CHECK: .Lheapallocsite3:
; CHECK: retq

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 .Lheapallocsite0
; CHECK-NEXT:  .secidx .Lheapallocsite0
; CHECK-NEXT:  .short .Lheapallocsite1-.Lheapallocsite0
; CHECK-NEXT:  .long 3
; CHECK-NEXT:  .p2align 2
; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 .Lheapallocsite2
; CHECK-NEXT:  .secidx .Lheapallocsite2
; CHECK-NEXT:  .short .Lheapallocsite3-.Lheapallocsite2
; CHECK-NEXT:  .long 3
; CHECK-NEXT:  .p2align 2
; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #2

declare dso_local void @use_alloc(i8*) #1

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git 4eff3de99423a62fd6e833e29c71c1e62ba6140b)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "heapallocsite.cpp", directory: "C:\5Csrc\5Ctest", checksumkind: CSK_MD5, checksum: "6d758cfa3834154a04ce8a55102772a9")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 4eff3de99423a62fd6e833e29c71c1e62ba6140b)"}
!8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 3, type: !9, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 4, scope: !8)
!12 = !DILocation(line: 5, scope: !8)
!13 = distinct !DISubprogram(name: "do_alloc", scope: !1, file: !1, line: 11, type: !14, scopeLine: 11, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64)
!17 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 7, flags: DIFlagFwdDecl)
!18 = !DILocalVariable(name: "p", arg: 1, scope: !13, file: !1, line: 11, type: !16)
!19 = !DILocation(line: 11, scope: !13)
!20 = !DILocation(line: 12, scope: !13)
!21 = !DILocation(line: 13, scope: !13)

