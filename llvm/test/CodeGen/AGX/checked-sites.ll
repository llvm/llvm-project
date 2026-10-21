; RUN: llc -mtriple=agx -mcpu=agx-g14c -mattr=+agx-checked -stop-after=agx-checked-accesses < %s | FileCheck %s
; RUN: llc -mtriple=agx -mcpu=agx-g14c -stop-after=agx-checked-accesses < %s | FileCheck %s --check-prefix=FLAG

; Check sites carry the inlined call chain of their debug location, innermost
; first. The module flag selects the feature as well.

; CHECK: @__agx_checked_sites = addrspace(1) constant [48 x i8] c"1\09store 4\09put saxpy.cu:3:5\09saxpy saxpy.cu:12:3\0A\00"
; FLAG: @__agx_checked_sites

define internal void @put(ptr addrspace(1) %p, i32 %v) alwaysinline !dbg !10 {
  store i32 %v, ptr addrspace(1) %p, align 4, !dbg !11
  ret void
}

define agx_kernel void @saxpy(ptr addrspace(1) %out, i32 %i) !dbg !5 {
  %p = getelementptr i32, ptr addrspace(1) %out, i32 %i, !dbg !8
  call void @put(ptr addrspace(1) %p, i32 7), !dbg !8
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "saxpy.cu", directory: "/tmp")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"agx-checked", i32 1}
!5 = distinct !DISubprogram(name: "saxpy", scope: !1, file: !1, line: 10, type: !6, unit: !0, spFlags: DISPFlagDefinition)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !DILocation(line: 12, column: 3, scope: !5)
!10 = distinct !DISubprogram(name: "put", scope: !1, file: !1, line: 2, type: !6, unit: !0, spFlags: DISPFlagDefinition)
!11 = !DILocation(line: 3, column: 5, scope: !10)
