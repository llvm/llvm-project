; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -filetype=obj | llvm-readobj - --codeview | FileCheck %s --check-prefix=OBJ

; C++ source to regenerate:
; struct Foo {
;   Foo() = default;
;   Foo(Foo &&other) { x = other.x; }
;   int x;
; };
; void some_function(int);
; Foo getFoo() {
;   Foo foo;
;   foo.x = 41;
;   some_function(foo.x);
;   return foo;
; }
;
; int main() {
;   Foo bar = getFoo();
;   return bar.x;
; }
; $ clang t.cpp -S -emit-llvm -g -o t.ll

; ASM-LABEL:  .long  241                      # Symbol subsection for GetFoo
; ASM:        .short 4414                     # Record kind: S_LOCAL
; ASM-NEXT:   .long 4113                      # TypeIndex
; ASM-NEXT:   .short 0                        # Flags
; ASM-NEXT:   .asciz "foo"
; ASM-NEXT:   .p2align 2
; ASM-NEXT: .Ltmp
; ASM:        .cv_def_range  .Ltmp{{.*}} .Ltmp{{.*}}, frame_ptr_rel, 40

; OBJ: Subsection [
; OBJ:   SubSectionType: Symbols (0xF1)
; OBJ:   LocalSym {
; OBJ:     Kind: S_LOCAL (0x113E)
; OBJ:     Type: Foo& (0x1011)
; OBJ:     Flags [ (0x0)
; OBJ:     ]
; OBJ:     VarName: foo
; OBJ:   }
; OBJ:   DefRangeFramePointerRelSym {
; OBJ:     Kind: S_DEFRANGE_FRAMEPOINTER_REL (0x1142)
; OBJ:     Offset: 40
; OBJ:     LocalVariableAddrRange {
; OBJ:       OffsetStart: .text+0x1D
; OBJ:       ISectStart: 0x0
; OBJ:       Range: 0x16
; OBJ:   }

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.16.27030"

%struct.Foo = type { i32 }

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?some_function@@YAXH@Z"(i32) #0 !dbg !8 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %.addr, metadata !12, metadata !DIExpression()), !dbg !13
  ret void, !dbg !13
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @"?GetFoo@@YA?AUFoo@@XZ"(%struct.Foo* noalias sret %agg.result) #0 !dbg !14 {
entry:
  %result.ptr = alloca i8*, align 8
  %0 = bitcast %struct.Foo* %agg.result to i8*
  store i8* %0, i8** %result.ptr, align 8
  call void @llvm.dbg.declare(metadata i8** %result.ptr, metadata !28, metadata !DIExpression(DW_OP_deref)), !dbg !29
  %x = getelementptr inbounds %struct.Foo, %struct.Foo* %agg.result, i32 0, i32 0, !dbg !30
  store i32 41, i32* %x, align 4, !dbg !30
  %x1 = getelementptr inbounds %struct.Foo, %struct.Foo* %agg.result, i32 0, i32 0, !dbg !31
  %1 = load i32, i32* %x1, align 4, !dbg !31
  call void @"?some_function@@YAXH@Z"(i32 %1), !dbg !31
  ret void, !dbg !32
}

; Function Attrs: noinline norecurse nounwind optnone uwtable
define dso_local i32 @main() #2 !dbg !33 {
entry:
  %retval = alloca i32, align 4
  %bar = alloca %struct.Foo, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo* %bar, metadata !36, metadata !DIExpression()), !dbg !37
  call void @"?GetFoo@@YA?AUFoo@@XZ"(%struct.Foo* sret %bar), !dbg !37
  %x = getelementptr inbounds %struct.Foo, %struct.Foo* %bar, i32 0, i32 0, !dbg !38
  %0 = load i32, i32* %x, align 4, !dbg !38
  ret i32 %0, !dbg !38
}

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline norecurse nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 9.0.0 (https://github.com/llvm/llvm-project.git c19ebebac4bf853e77a69c74abe9f7fce98c1d17)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "t.cpp", directory: "C:\5Csrc\5Ctesting\5Cnrvo", checksumkind: CSK_MD5, checksum: "52a5a20c02c102dfd255d5615680a8bd")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 9.0.0 (https://github.com/llvm/llvm-project.git c19ebebac4bf853e77a69c74abe9f7fce98c1d17)"}
!8 = distinct !DISubprogram(name: "some_function", linkageName: "?some_function@@YAXH@Z", scope: !1, file: !1, line: 13, type: !9, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(arg: 1, scope: !8, file: !1, line: 13, type: !11)
!13 = !DILocation(line: 13, scope: !8)
!14 = distinct !DISubprogram(name: "GetFoo", linkageName: "?GetFoo@@YA?AUFoo@@XZ", scope: !1, file: !1, line: 15, type: !15, scopeLine: 15, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!15 = !DISubroutineType(types: !16)
!16 = !{!17}
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 1, size: 32, flags: DIFlagTypePassByReference | DIFlagNonTrivial, elements: !18, identifier: ".?AUFoo@@")
!18 = !{!19, !20, !24}
!19 = !DIDerivedType(tag: DW_TAG_member, name: "x", scope: !17, file: !1, line: 4, baseType: !11, size: 32)
!20 = !DISubprogram(name: "Foo", scope: !17, file: !1, line: 2, type: !21, scopeLine: 2, flags: DIFlagPrototyped, spFlags: 0)
!21 = !DISubroutineType(types: !22)
!22 = !{null, !23}
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 64, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DISubprogram(name: "Foo", scope: !17, file: !1, line: 3, type: !25, scopeLine: 3, flags: DIFlagPrototyped, spFlags: 0)
!25 = !DISubroutineType(types: !26)
!26 = !{null, !23, !27}
!27 = !DIDerivedType(tag: DW_TAG_rvalue_reference_type, baseType: !17, size: 64)
!28 = !DILocalVariable(name: "foo", scope: !14, file: !1, line: 17, type: !17)
!29 = !DILocation(line: 17, scope: !14)
!30 = !DILocation(line: 18, scope: !14)
!31 = !DILocation(line: 19, scope: !14)
!32 = !DILocation(line: 21, scope: !14)
!33 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 23, type: !34, scopeLine: 23, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!34 = !DISubroutineType(types: !35)
!35 = !{!11}
!36 = !DILocalVariable(name: "bar", scope: !33, file: !1, line: 24, type: !17)
!37 = !DILocation(line: 24, scope: !33)
!38 = !DILocation(line: 25, scope: !33)
