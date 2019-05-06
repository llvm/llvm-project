; RUN: llc -O0 < %s | FileCheck %s
; FIXME: Add test for llc with optimizations once it is implemented.

; Source to regenerate:
; $ clang --target=x86_64-windows-msvc -S heapallocsite.cpp -g -gcodeview -o t.ll \
;      -emit-llvm -O0 -Xclang -disable-llvm-passes -fms-extensions
;
; struct Foo {
;   __declspec(allocator) virtual void *alloc();
; };
;
; extern "C" __declspec(allocator) Foo *alloc_foo();
;
; extern "C" void use_alloc(void*);
; extern "C" void call_virtual(Foo *p) {
;   use_alloc(p->alloc());
; }
;
; extern "C" void call_multiple() {
;   use_alloc(alloc_foo());
;   use_alloc(alloc_foo());
; }

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.Foo = type { i32 (...)** }

; Function Attrs: noinline optnone uwtable
define dso_local void @call_virtual(%struct.Foo* %p) #0 !dbg !8 {
entry:
  %p.addr = alloca %struct.Foo*, align 8
  store %struct.Foo* %p, %struct.Foo** %p.addr, align 8
  call void @llvm.dbg.declare(metadata %struct.Foo** %p.addr, metadata !13, metadata !DIExpression()), !dbg !14
  %0 = load %struct.Foo*, %struct.Foo** %p.addr, align 8, !dbg !15
  %1 = bitcast %struct.Foo* %0 to i8* (%struct.Foo*)***, !dbg !15
  %vtable = load i8* (%struct.Foo*)**, i8* (%struct.Foo*)*** %1, align 8, !dbg !15
  %vfn = getelementptr inbounds i8* (%struct.Foo*)*, i8* (%struct.Foo*)** %vtable, i64 0, !dbg !15
  %2 = load i8* (%struct.Foo*)*, i8* (%struct.Foo*)** %vfn, align 8, !dbg !15
  %call = call i8* %2(%struct.Foo* %0), !dbg !15, !heapallocsite !2
  call void @use_alloc(i8* %call), !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local void @use_alloc(i8*) #2

; Function Attrs: noinline optnone uwtable
define dso_local void @call_multiple() #0 !dbg !17 {
entry:
  %call = call %struct.Foo* @alloc_foo(), !dbg !20, !heapallocsite !12
  %0 = bitcast %struct.Foo* %call to i8*, !dbg !20
  call void @use_alloc(i8* %0), !dbg !20
  %call1 = call %struct.Foo* @alloc_foo(), !dbg !21, !heapallocsite !12
  %1 = bitcast %struct.Foo* %call1 to i8*, !dbg !21
  call void @use_alloc(i8* %1), !dbg !21
  ret void, !dbg !22
}

declare dso_local %struct.Foo* @alloc_foo() #2

; CHECK-LABEL: call_virtual: # @call_virtual
; CHECK: .Lheapallocsite0:
; CHECK: callq *(%rax)
; CHECK: .Lheapallocsite1:
; CHECK: retq

; CHECK-LABEL: call_multiple: # @call_multiple
; CHECK: .Lheapallocsite4:
; CHECK: callq alloc_foo
; CHECK: .Lheapallocsite5:
; CHECK: .Lheapallocsite2:
; CHECK: callq alloc_foo
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
; CHECK-NEXT:  .long 4096
; CHECK-NEXT:  .p2align 2

; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 .Lheapallocsite4
; CHECK-NEXT:  .secidx .Lheapallocsite4
; CHECK-NEXT:  .short .Lheapallocsite5-.Lheapallocsite4
; CHECK-NEXT:  .long 4096
; CHECK-NEXT:  .p2align 2
; CHECK-LABEL: .short  4431                    # Record kind: S_PROC_ID_END

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git 9c8073f44f786fbf47335e53f20abe64429e8e47)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)!1 = !DIFile(filename: "filename", directory: "directory", checksumkind: CSK_MD5, checksum: "096443b661a0af36da9006330c08f97e")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git 9c8073f44f786fbf47335e53f20abe64429e8e47)"}
!8 = distinct !DISubprogram(name: "call_virtual", scope: !1, file: !1, line: 8, type: !9, scopeLine: 8, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUFoo@@")
!13 = !DILocalVariable(name: "p", arg: 1, scope: !8, file: !1, line: 8, type: !11)
!14 = !DILocation(line: 8, scope: !8)
!15 = !DILocation(line: 9, scope: !8)
!16 = !DILocation(line: 10, scope: !8)
!17 = distinct !DISubprogram(name: "call_multiple", scope: !1, file: !1, line: 12, type: !18, scopeLine: 12, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{null}
!20 = !DILocation(line: 13, scope: !17)
!21 = !DILocation(line: 14, scope: !17)
!22 = !DILocation(line: 15, scope: !17)
