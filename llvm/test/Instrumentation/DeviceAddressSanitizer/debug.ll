; RUN: opt < %s -passes=dasan -S | FileCheck %s
; RUN: opt < %s -passes=dasan -dasan-recover -S | FileCheck %s --check-prefix=REC

target datalayout = "e-p:64:64:64-i8:8:8-i16:16:16-i32:32:32-i64:64:64-n8:16:32:64-S128-A5-G1"
target triple = "amdgcn-amd-amdhsa"

; The handler takes its return address as the location, so every check reports
; through a call of its own carrying the location of its access.

; CHECK-LABEL: define i32 @located
; CHECK: notail call void @__dasan_report_load({{.*}}) #[[SITE:[0-9]+]], !dbg [[LOC:![0-9]+]]
define i32 @located(ptr addrspace(1) %p) sanitize_device_address !dbg !4 {
  %v = load i32, ptr addrspace(1) %p, align 4, !dbg !8
  ret i32 %v, !dbg !8
}

; CHECK-LABEL: define i32 @unlocated
; CHECK: notail call void @__dasan_report_load({{.*}}) #[[SITE]], !dbg [[SCOPE:![0-9]+]]
define i32 @unlocated(ptr addrspace(1) %p) sanitize_device_address !dbg !9 {
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}

; CHECK-LABEL: define i32 @nodebug
; CHECK: notail call void @__dasan_report_load({{.*}}) #[[SITE]], {{.*}}!nosanitize
define i32 @nodebug(ptr addrspace(1) %p) sanitize_device_address {
  %v = load i32, ptr addrspace(1) %p, align 4
  ret i32 %v
}

; Checks sharing a bound, or inlined from elsewhere, still get their own call.
; CHECK-LABEL: define void @sites
; CHECK: notail call void @__dasan_report_store({{.*}}) #[[SITE]], !dbg [[S1:![0-9]+]]
; CHECK-NEXT: unreachable
; CHECK: store i32 0, ptr addrspace(1) %a, align 4, !dbg [[S1]]
; CHECK: notail call void @__dasan_report_store({{.*}}) #[[SITE]], !dbg [[S2:![0-9]+]]
; CHECK-NEXT: unreachable
; CHECK: store i32 1, ptr addrspace(1) %b, align 4, !dbg [[S2]]
; CHECK: notail call void @__dasan_report_load({{.*}}) #[[SITE]], !dbg [[S3:![0-9]+]]
; CHECK-NEXT: unreachable
; CHECK: load i32, ptr addrspace(1) %c, align 4, !dbg [[S3]]
; REC-LABEL: define void @sites
; REC: notail call void @__dasan_report_store_noabort({{.*}}) #[[SITE:[0-9]+]], !dbg [[S1:![0-9]+]]
; REC-NEXT: br label
; REC: store i32 0, ptr addrspace(1) %a, align 4, !dbg [[S1]]
; REC: notail call void @__dasan_report_store_noabort({{.*}}) #[[SITE]], !dbg [[S2:![0-9]+]]
; REC-NEXT: br label
; REC: store i32 1, ptr addrspace(1) %b, align 4, !dbg [[S2]]
; REC: notail call void @__dasan_report_load_noabort({{.*}}) #[[SITE]], !dbg [[S3:![0-9]+]]
; REC-NEXT: br label
; REC: load i32, ptr addrspace(1) %c, align 4, !dbg [[S3]]
define void @sites(ptr addrspace(1) %p, ptr addrspace(1) %q, i64 %i) sanitize_device_address !dbg !10 {
  %a = getelementptr i32, ptr addrspace(1) %p, i64 %i
  store i32 0, ptr addrspace(1) %a, align 4, !dbg !11
  %b = getelementptr i32, ptr addrspace(1) %q, i64 %i
  store i32 1, ptr addrspace(1) %b, align 4, !dbg !12
  %j = add i64 %i, 8
  %c = getelementptr i32, ptr addrspace(1) %p, i64 %j
  %v = load i32, ptr addrspace(1) %c, align 4, !dbg !14
  ret void
}

; CHECK: attributes #[[SITE]] = { noinline nomerge }
; REC: attributes #[[SITE]] = { noinline nomerge }

; CHECK-DAG: [[LOC]] = !DILocation(line: 7, column: 3, scope: !{{[0-9]+}})
; CHECK-DAG: [[SCOPE]] = !DILocation(line: 11, scope: !{{[0-9]+}})
; CHECK-DAG: [[S1]] = !DILocation(line: 21, column: 3, scope: !{{[0-9]+}})
; CHECK-DAG: [[S2]] = !DILocation(line: 22, column: 3, scope: !{{[0-9]+}})
; CHECK-DAG: [[S3]] = !DILocation(line: 31, column: 5, scope: !{{[0-9]+}}, inlinedAt: !{{[0-9]+}})

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = distinct !DISubprogram(name: "located", scope: !1, file: !1, line: 6, type: !5, scopeLine: 6, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !DILocation(line: 7, column: 3, scope: !4)
!9 = distinct !DISubprogram(name: "unlocated", scope: !1, file: !1, line: 10, type: !5, scopeLine: 11, spFlags: DISPFlagDefinition, unit: !0)
!10 = distinct !DISubprogram(name: "sites", scope: !1, file: !1, line: 20, type: !5, scopeLine: 20, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DILocation(line: 21, column: 3, scope: !10)
!12 = !DILocation(line: 22, column: 3, scope: !10)
!13 = distinct !DISubprogram(name: "helper", scope: !1, file: !1, line: 30, type: !5, scopeLine: 30, spFlags: DISPFlagDefinition, unit: !0)
!14 = !DILocation(line: 31, column: 5, scope: !13, inlinedAt: !15)
!15 = distinct !DILocation(line: 23, column: 3, scope: !10)
