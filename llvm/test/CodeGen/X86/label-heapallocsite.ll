; RUN: llc < %s | FileCheck --check-prefixes=CHECK %s
; RUN: llc -O0 < %s | FileCheck --check-prefixes=CHECK %s

; Source to regenerate:
; $ clang -cc1 -triple x86_64-windows-msvc t.cpp -debug-info-kind=limited \
;      -gcodeview -O2 -fms-extensions -emit-llvm -o t.ll
;
; extern "C" struct Foo {
;   __declspec(allocator) virtual ptr alloc();
; };
; extern "C" __declspec(allocator) Foo *alloc_foo();
; extern "C" void use_result(ptr);
; extern "C" Foo *call_tail() {
;   return alloc_foo();
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

; ModuleID = 't.cpp'
source_filename = "t.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-windows-msvc"

%struct.Foo = type { ptr }

; Function Attrs: nounwind
define dso_local ptr @call_tail() local_unnamed_addr #0 !dbg !7 {
entry:
  %call = tail call ptr @alloc_foo() #3, !dbg !13, !heapallocsite !12
  ret ptr %call, !dbg !13
}

declare dso_local ptr @alloc_foo() local_unnamed_addr #1

; Function Attrs: nounwind
define dso_local i32 @call_virtual(ptr %p) local_unnamed_addr #0 !dbg !14 {
entry:
  call void @llvm.dbg.value(metadata ptr %p, metadata !19, metadata !DIExpression()), !dbg !20
  %vtable = load ptr, ptr %p, align 8, !dbg !21, !tbaa !22
  %0 = load ptr, ptr %vtable, align 8, !dbg !21
  %call = tail call ptr %0(ptr %p) #3, !dbg !21, !heapallocsite !2
  tail call void @use_result(ptr %call) #3, !dbg !21
  ret i32 0, !dbg !25
}

declare dso_local void @use_result(ptr) local_unnamed_addr #1

; Function Attrs: nounwind
define dso_local i32 @call_multiple() local_unnamed_addr #0 !dbg !26 {
entry:
  %call = tail call ptr @alloc_foo() #3, !dbg !29, !heapallocsite !12
  tail call void @use_result(ptr %call) #3, !dbg !29
  %call1 = tail call ptr @alloc_foo() #3, !dbg !30, !heapallocsite !12
  tail call void @use_result(ptr %call1) #3, !dbg !30
  ret i32 0, !dbg !31
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #2


; Don't emit metadata for tail calls.
; CHECK-LABEL: call_tail:         # @call_tail
; CHECK: jmp alloc_foo

; CHECK-LABEL: call_virtual:      # @call_virtual
; CHECK:       callq *{{.*}}%rax{{.*}}
; CHECK-NEXT:  [[LABEL1:.Ltmp[0-9]+]]:

; CHECK-LABEL: call_multiple:     # @call_multiple
; CHECK:       callq alloc_foo
; CHECK-NEXT:  [[LABEL3:.Ltmp[0-9]+]]:
; CHECK:       callq alloc_foo
; CHECK-NEXT:  [[LABEL5:.Ltmp[0-9]+]]:

; CHECK-LABEL: .short  4423                    # Record kind: S_GPROC32_ID
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 [[LABEL0:.Ltmp[0-9]+]]
; CHECK-NEXT:  .secidx [[LABEL0]]
; CHECK-NEXT:  .short [[LABEL1]]-[[LABEL0]]
; CHECK-NEXT:  .long 3
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 [[LABEL2:.Ltmp[0-9]+]]
; CHECK-NEXT:  .secidx [[LABEL2]]
; CHECK-NEXT:  .short [[LABEL3]]-[[LABEL2]]
; CHECK-NEXT:  .long 4096
; CHECK:       .short  4446                    # Record kind: S_HEAPALLOCSITE
; CHECK-NEXT:  .secrel32 [[LABEL4:.Ltmp[0-9]+]]
; CHECK-NEXT:  .secidx [[LABEL4]]
; CHECK-NEXT:  .short [[LABEL5]]-[[LABEL4]]
; CHECK-NEXT:  .long 4096

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+cx8,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable willreturn }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 10.0.0 (https://github.com/llvm/llvm-project.git fa686ea7650235c6dff988cc8cba49e130b3d5f8)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "<stdin>", directory: "/usr/local/google/home/akhuang/testing/heapallocsite", checksumkind: CSK_MD5, checksum: "e0a04508b4229fc4aee0baa364e25987")
!2 = !{}
!3 = !{i32 2, !"CodeView", i32 1}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 2}
!6 = !{!"clang version 10.0.0 (https://github.com/llvm/llvm-project.git fa686ea7650235c6dff988cc8cba49e130b3d5f8)"}
!7 = distinct !DISubprogram(name: "call_tail", scope: !8, file: !8, line: 6, type: !9, scopeLine: 6, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!8 = !DIFile(filename: "t.cpp", directory: "/usr/local/google/home/akhuang/testing/heapallocsite", checksumkind: CSK_MD5, checksum: "e0a04508b4229fc4aee0baa364e25987")
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 64)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !8, line: 1, flags: DIFlagFwdDecl, identifier: ".?AUFoo@@")
!13 = !DILocation(line: 7, scope: !7)
!14 = distinct !DISubprogram(name: "call_virtual", scope: !8, file: !8, line: 9, type: !15, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !18)
!15 = !DISubroutineType(types: !16)
!16 = !{!17, !11}
!17 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!18 = !{!19}
!19 = !DILocalVariable(name: "p", arg: 1, scope: !14, file: !8, line: 9, type: !11)
!20 = !DILocation(line: 0, scope: !14)
!21 = !DILocation(line: 10, scope: !14)
!22 = !{!23, !23, i64 0}
!23 = !{!"vtable pointer", !24, i64 0}
!24 = !{!"Simple C++ TBAA"}
!25 = !DILocation(line: 11, scope: !14)
!26 = distinct !DISubprogram(name: "call_multiple", scope: !8, file: !8, line: 13, type: !27, scopeLine: 13, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !2)
!27 = !DISubroutineType(types: !28)
!28 = !{!17}
!29 = !DILocation(line: 14, scope: !26)
!30 = !DILocation(line: 15, scope: !26)
!31 = !DILocation(line: 16, scope: !26)
