; RUN: llc < %s | FileCheck --check-prefixes=DAG,CHECK %s
; RUN: llc -O0 < %s | FileCheck --check-prefixes=FAST,CHECK %s

; Source to regenerate:
; $ clang -cc1 -triple x86_64-windows-msvc t.cpp -debug-info-kind=limited \
;      -gcodeview -O2 -fms-extensions -emit-llvm -o t.ll
;
; extern "C" struct Foo {
;   __declspec(allocator) virtual void *alloc();
; };
; extern "C" __declspec(allocator) Foo *alloc_foo();
; extern "C" void use_result(void *);
;
; extern "C" int call_tail() {
;   use_result(alloc_foo());
; }
; extern "C" int call_virtual(Foo *p) {
;   use_result(p->alloc());
;   return 0;
; }
; extern "C" int call_multiple() {
;   use_result(alloc_foo());
;   use_result(alloc_foo());
;   return 0;
; }

; ModuleID = 'label.cpp'
source_filename = "label.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.Foo = type { i32 (...)** }

; Function Attrs: nounwind
define dso_local void @call_tail() local_unnamed_addr #0 !dbg !7 {
entry:
  %call = tail call %struct.Foo* @alloc_foo() #3, !dbg !11, !heapallocsite !12
  %0 = bitcast %struct.Foo* %call to i8*, !dbg !11
  tail call void @use_result(i8* %0) #3, !dbg !11
  ret void, !dbg !13
}

declare dso_local void @use_result(i8*) local_unnamed_addr #1

declare dso_local %struct.Foo* @alloc_foo() local_unnamed_addr #1

; Function Attrs: nounwind
define dso_local i32 @call_virtual(%struct.Foo* %p) local_unnamed_addr #0 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata %struct.Foo* %p, metadata !20, metadata !DIExpression()), !dbg !21
  %0 = bitcast %struct.Foo* %p to i8* (%struct.Foo*)***, !dbg !22
  %vtable = load i8* (%struct.Foo*)**, i8* (%struct.Foo*)*** %0, align 8, !dbg !22, !tbaa !23
  %1 = load i8* (%struct.Foo*)*, i8* (%struct.Foo*)** %vtable, align 8, !dbg !22
  %call = tail call i8* %1(%struct.Foo* %p) #3, !dbg !22, !heapallocsite !2
  tail call void @use_result(i8* %call) #3, !dbg !22
  ret i32 0, !dbg !26
}

; Function Attrs: nounwind
define dso_local i32 @call_multiple() local_unnamed_addr #0 !dbg !27 {
entry:
  %call = tail call %struct.Foo* @alloc_foo() #3, !dbg !30, !heapallocsite !12
  %0 = bitcast %struct.Foo* %call to i8*, !dbg !30
  tail call void @use_result(i8* %0) #3, !dbg !30
  %call1 = tail call %struct.Foo* @alloc_foo() #3, !dbg !31, !heapallocsite !12
  %1 = bitcast %struct.Foo* %call1 to i8*, !dbg !31
  tail call void @use_result(i8* %1) #3, !dbg !31
  ret i32 0, !dbg !32
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

; CHECK-LABEL: call_tail:         # @call_tail
; CHECK: .Lheapallocsite0:
; CHECK: callq alloc_foo
; CHECK: .Lheapallocsite1:

; CHECK-LABEL: call_virtual:      # @call_virtual
; CHECK: .Lheapallocsite2:
; CHECK: callq *{{.*}}%rax{{.*}}
; CHECK: .Lheapallocsite3:

; CHECK-LABEL: call_multiple:     # @call_multiple
; FastISel emits instructions in a different order.
; DAG:   .Lheapallocsite4:
; FAST:  .Lheapallocsite6:
; CHECK: callq alloc_foo
; DAG:   .Lheapallocsite5:
; FAST:  .Lheapallocsite7:
; DAG:   .Lheapallocsite8:
; FAST:  .Lheapallocsite4:
; CHECK: callq alloc_foo
; DAG:   .Lheapallocsite9:
; FAST:  .Lheapallocsite5:

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 .Lheapallocsite0
; CHECK-NEXT:  .secidx .Lheapallocsite0
; CHECK-NEXT:  .short .Lheapallocsite1-.Lheapallocsite0
; CHECK-NEXT:  .long 4099

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 .Lheapallocsite2
; CHECK-NEXT:  .secidx .Lheapallocsite2
; CHECK-NEXT:  .short .Lheapallocsite3-.Lheapallocsite2
; CHECK-NEXT:  .long 3
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 .Lheapallocsite4
; CHECK-NEXT:  .secidx .Lheapallocsite4
; CHECK-NEXT:  .short .Lheapallocsite5-.Lheapallocsite4
; CHECK-NEXT:  .long 4099
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; FAST-NEXT:  .secrel32 .Lheapallocsite6
; FAST-NEXT:  .secidx .Lheapallocsite6
; FAST-NEXT:  .short .Lheapallocsite7-.Lheapallocsite6
; DAG-NEXT:  .secrel32 .Lheapallocsite8
; DAG-NEXT:  .secidx .Lheapallocsite8
; DAG-NEXT:  .short .Lheapallocsite9-.Lheapallocsite8
; CHECK-NEXT:  .long 4099

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git ebca9d67ffca71c9a996bd89844425ee13141f47)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/akhuang/testing/heapallocsite", checksumkind: CSK_MD5, checksum: "68a8ba93f37944165cfe76612a7073fd")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git ebca9d67ffca71c9a996bd89844425ee13141f47)"}
!7 = distinct !DISubprogram(name: "call_tail", scope: !8, file: !8, line: 7, type: !9, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "label.cpp", directory: "/usr/local/google/home/akhuang/testing/heapallocsite", checksumkind: CSK_MD5, checksum: "68a8ba93f37944165cfe76612a7073fd")
!9 = !DISubroutineType(types: !10)
!10 = !{null}
!11 = !DILocation(line: 8, scope: !7)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !8, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUFoo@@")
!13 = !DILocation(line: 9, scope: !7)
!14 = distinct !DISubprogram(name: "call_virtual", scope: !8, file: !8, line: 10, type: !15, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !19)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !18}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!19 = !{!20}
!20 = !DILocalVariable(name: "p", arg: 1, scope: !14, file: !8, line: 10, type: !18)
!21 = !DILocation(line: 0, scope: !14)
!22 = !DILocation(line: 11, scope: !14)
!23 = !{!24, !24, i64 0}
!24 = !{!"vtable pointer", !25, i64 0}
!25 = !{!"Simple C++ TBAA"}
!26 = !DILocation(line: 12, scope: !14)
!27 = distinct !DISubprogram(name: "call_multiple", scope: !8, file: !8, line: 14, type: !28, scopeLine: 14, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{!17}
!30 = !DILocation(line: 15, scope: !27)
!31 = !DILocation(line: 16, scope: !27)
!32 = !DILocation(line: 17, scope: !27)
