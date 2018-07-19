; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS1 %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS2 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Check origin instrumentation of stores.
; Check that debug info for origin propagation code is set correctly.

; Function Attrs: nounwind
define void @Store(i32* nocapture %p, i32 %x) #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32* %p, i64 0, metadata !11, metadata !DIExpression()), !dbg !16
  tail call void @llvm.dbg.value(metadata i32 %x, i64 0, metadata !12, metadata !DIExpression()), !dbg !16
  store i32 %x, i32* %p, align 4, !dbg !17, !tbaa !18
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind sanitize_memory "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, producer: "clang version 3.5.0 (204220)", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "../2.cc", directory: "/tmp/build0")
!2 = !{}
!4 = distinct !DISubprogram(name: "Store", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 1, file: !1, scope: !5, type: !6, variables: !10)
!5 = !DIFile(filename: "../2.cc", directory: "/tmp/build0")
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !9}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !9)
!9 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11, !12}
!11 = !DILocalVariable(name: "p", line: 1, arg: 1, scope: !4, file: !5, type: !8)
!12 = !DILocalVariable(name: "x", line: 1, arg: 2, scope: !4, file: !5, type: !9)
!13 = !{i32 2, !"Dwarf Version", i32 4}
!14 = !{i32 1, !"Debug Info Version", i32 3}
!15 = !{!"clang version 3.5.0 (204220)"}
!16 = !DILocation(line: 1, scope: !4)
!17 = !DILocation(line: 2, scope: !4)
!18 = !{!19, !19, i64 0}
!19 = !{!"int", !20, i64 0}
!20 = !{!"omnipotent char", !21, i64 0}
!21 = !{!"Simple C/C++ TBAA"}
!22 = !DILocation(line: 3, scope: !4)


; CHECK-LABEL: @Store
; CHECK: load {{.*}} @__msan_param_tls
; CHECK: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK: store {{.*}}!dbg ![[DBG:[01-9]+]]
; CHECK: icmp
; CHECK: br i1
; CHECK: <label>

; Origin tracking level 1: simply store the origin value
; CHECK-ORIGINS1: store i32 {{.*}}[[ORIGIN]],{{.*}}!dbg !{{.*}}[[DBG]]

; Origin tracking level 2: pass origin value through __msan_chain_origin and store the result.
; CHECK-ORIGINS2: [[ORIGIN2:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 {{.*}}[[ORIGIN]])
; CHECK-ORIGINS2: store i32 {{.*}}[[ORIGIN2]],{{.*}}!dbg !{{.*}}[[DBG]]

; CHECK: br label{{.*}}!dbg !{{.*}}[[DBG]]
; CHECK: <label>
; CHECK: store{{.*}}!dbg !{{.*}}[[DBG]]
; CHECK: ret void
