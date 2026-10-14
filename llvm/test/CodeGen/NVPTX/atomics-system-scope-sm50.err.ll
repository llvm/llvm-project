; RUN: split-file %s %t
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/cmpxchg.ll 2>&1 | FileCheck %s --check-prefix=LOC
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/cmpxchg-partword.ll 2>&1 | FileCheck %s
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/atomicrmw.ll 2>&1 | FileCheck %s
; RUN: not llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 -filetype=null %t/atomicrmw-expand.ll 2>&1 | FileCheck %s
; RUN: llc -mtriple=nvptx64 -mcpu=sm_50 -mattr=+ptx40 %t/device.ll -o - | FileCheck %s --check-prefix=DEVICE

; CHECK: error: <unknown>:0:0: in function {{.*}}: NVPTX system scope atomics require sm_60 or later
; LOC: error: system-atomic.c:7:9: in function cmpxchg_system i32 (ptr, i32, i32): NVPTX system scope atomics require sm_60 or later

;--- cmpxchg.ll
define i32 @cmpxchg_system(ptr %addr, i32 %cmp, i32 %new) !dbg !4 {
  %result = cmpxchg ptr %addr, i32 %cmp, i32 %new monotonic monotonic, !dbg !7
  %value = extractvalue { i32, i1 } %result, 0
  ret i32 %value
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "system-atomic.c", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "cmpxchg_system", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 7, column: 9, scope: !4)

;--- cmpxchg-partword.ll
define i8 @cmpxchg_system_i8(ptr %addr, i8 %cmp, i8 %new) {
  %result = cmpxchg ptr %addr, i8 %cmp, i8 %new monotonic monotonic
  %value = extractvalue { i8, i1 } %result, 0
  ret i8 %value
}

;--- atomicrmw.ll
define i32 @atomicrmw_system(ptr %addr, i32 %value) {
  %result = atomicrmw add ptr %addr, i32 %value monotonic
  ret i32 %result
}

;--- atomicrmw-expand.ll
define i32 @atomicrmw_system_nand(ptr %addr, i32 %value) {
  %result = atomicrmw nand ptr %addr, i32 %value monotonic
  ret i32 %result
}

;--- device.ll
; DEVICE-LABEL: cmpxchg_device(
; DEVICE: atom.cas.b32
define i32 @cmpxchg_device(ptr %addr, i32 %cmp, i32 %new) {
  %result = cmpxchg ptr %addr, i32 %cmp, i32 %new syncscope("device") monotonic monotonic
  %value = extractvalue { i32, i1 } %result, 0
  ret i32 %value
}

; DEVICE-LABEL: atomicrmw_device(
; DEVICE: atom.add.u32
define i32 @atomicrmw_device(ptr %addr, i32 %value) {
  %result = atomicrmw add ptr %addr, i32 %value syncscope("device") monotonic
  ret i32 %result
}
