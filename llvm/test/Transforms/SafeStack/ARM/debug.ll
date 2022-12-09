; RUN: opt -safe-stack -safestack-use-pointer-address < %s -S | FileCheck %s
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7-pc-linux-android"

; Original C used to generate debug info:
; char*** addr;
; char** __safestack_pointer_address() {
;   return *addr;
; }
; void Capture(char*x);
; void f() { char c[16]; Capture(c); }

; CHECK: !36 = !DILocation(line: 3, column: 11, scope: !17, inlinedAt: !37)
; CHECK: !37 = distinct !DILocation(line: 6, scope: !27)

@addr = common local_unnamed_addr global ptr null, align 4, !dbg !0

; Function Attrs: norecurse nounwind readonly safestack
define ptr @__safestack_pointer_address() local_unnamed_addr #0 !dbg !17 {
entry:
  %0 = load ptr, ptr @addr, align 4, !dbg !20, !tbaa !21
  %1 = load ptr, ptr %0, align 4, !dbg !25, !tbaa !21
  ret ptr %1, !dbg !26
}

; Function Attrs: nounwind safestack
define void @f() local_unnamed_addr #1 !dbg !27 {
entry:
  %c = alloca [16 x i8], align 1
  call void @llvm.lifetime.start.p0(i64 16, ptr nonnull %c) #5, !dbg !35
  call void @llvm.dbg.declare(metadata ptr %c, metadata !31, metadata !DIExpression()), !dbg !36
  call void @Capture(ptr nonnull %c) #5, !dbg !37
  call void @llvm.lifetime.end.p0(i64 16, ptr nonnull %c) #5, !dbg !38
  ret void, !dbg !38
}

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #3

declare void @Capture(ptr) local_unnamed_addr #4

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #2

attributes #0 = { norecurse nounwind readonly safestack "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+armv7-a,+dsp,+neon,+vfp3,-thumb-mode" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind safestack "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+armv7-a,+dsp,+neon,+vfp3,-thumb-mode" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { argmemonly nounwind }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+armv7-a,+dsp,+neon,+vfp3,-thumb-mode" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!11, !12, !13, !14, !15}
!llvm.ident = !{!16}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "addr", scope: !2, file: !6, line: 1, type: !7, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5)
!3 = !DIFile(filename: "-", directory: "/")
!4 = !{}
!5 = !{!0}
!6 = !DIFile(filename: "<stdin>", directory: "/")
!7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !8, size: 32)
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 32)
!9 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32)
!10 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_unsigned_char)
!11 = !{i32 2, !"Dwarf Version", i32 4}
!12 = !{i32 2, !"Debug Info Version", i32 3}
!13 = !{i32 1, !"wchar_size", i32 4}
!14 = !{i32 1, !"min_enum_size", i32 4}
!15 = !{i32 7, !"PIC Level", i32 1}
!16 = !{!"clang"}
!17 = distinct !DISubprogram(name: "__safestack_pointer_address", scope: !6, file: !6, line: 2, type: !18, isLocal: false, isDefinition: true, scopeLine: 2, isOptimized: true, unit: !2, retainedNodes: !4)
!18 = !DISubroutineType(types: !19)
!19 = !{!8}
!20 = !DILocation(line: 3, column: 11, scope: !17)
!21 = !{!22, !22, i64 0}
!22 = !{!"any pointer", !23, i64 0}
!23 = !{!"omnipotent char", !24, i64 0}
!24 = !{!"Simple C/C++ TBAA"}
!25 = !DILocation(line: 3, column: 10, scope: !17)
!26 = !DILocation(line: 3, column: 3, scope: !17)
!27 = distinct !DISubprogram(name: "f", scope: !6, file: !6, line: 6, type: !28, isLocal: false, isDefinition: true, scopeLine: 6, isOptimized: true, unit: !2, retainedNodes: !30)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !{!31}
!31 = !DILocalVariable(name: "c", scope: !27, file: !6, line: 6, type: !32)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 128, elements: !33)
!33 = !{!34}
!34 = !DISubrange(count: 16)
!35 = !DILocation(line: 6, column: 12, scope: !27)
!36 = !DILocation(line: 6, column: 17, scope: !27)
!37 = !DILocation(line: 6, column: 24, scope: !27)
!38 = !DILocation(line: 6, column: 36, scope: !27)
