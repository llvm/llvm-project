; RUN: rm -rf %t && split-file %s %t
; RUN: opt -S -passes=lowertypetests -lowertypetests-summary-action=export -lowertypetests-read-summary=%t/summary.ll %t/main.ll | FileCheck %s

; Tests that functions in a jump table are reordered according to call edge hotness
; recorded in the ThinLTO summary index:
; 1. (Highest priority) Within each strict type, hotter functions are placed later,
;    ordered strictly by: Cold (0) < Unknown (1) < None (2) < Hot (3) < Critical (4).
; 2. (High priority) Across fragments, the fragment with the globally hottest function
;    (@f_critical) is placed at the very end of the jump table.
; 3. (Best effort) Each strict type's members remain contiguous (lowered to range checks, no holes).

;--- summary.ll
^0 = module: (path: "cfi-jumptable-hotness-summary.o", hash: (0, 0, 0, 0, 0))
^1 = gv: (guid: 100, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1)))

; Functions in typeid1 covering all 5 call edge hotness types:
; f_cold (GUID: 3658589069114391263) -> Cold (tier 0)
^2 = gv: (guid: 3658589069114391263, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: cold)))))
; f_unknown (GUID: 1205929326482009600) -> Unknown (tier 1)
^3 = gv: (guid: 1205929326482009600, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: unknown)))))
; f_none (GUID: 1928809046209326017) -> None (tier 2)
^4 = gv: (guid: 1928809046209326017, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: none)))))
; f_hot (GUID: 9377218764429055595) -> Hot (tier 3)
^5 = gv: (guid: 9377218764429055595, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: hot)))))
; f_critical (GUID: 14457025706112322155) -> Critical (tier 4)
^6 = gv: (guid: 14457025706112322155, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: critical)))))

; Functions in typeid2:
; g_cold (GUID: 4485074774011110174) -> Cold (tier 0)
^7 = gv: (guid: 4485074774011110174, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: cold)))))
; g_unknown (GUID: 16691380702939550262) -> multiple calls (cold, unknown), max hotness Unknown (tier 1)
^8 = gv: (guid: 16691380702939550262, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: cold), (callee: ^1, hotness: unknown)))))
; g_hot (GUID: 18081025037889099144) -> multiple calls, max hotness Hot (tier 3)
^9 = gv: (guid: 18081025037889099144, summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 0), insts: 1, calls: ((callee: ^1, hotness: none), (callee: ^1, hotness: hot)))))

;--- main.ll
target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

@0 = private unnamed_addr constant [8 x ptr] [ptr @f_critical, ptr @g_unknown, ptr @f_cold, ptr @g_hot, ptr @f_none, ptr @g_cold, ptr @f_unknown, ptr @f_hot], align 16

; Strict typeid1 and typeid2 functions defined in scrambled order:
define void @f_critical() !type !0 !type !1 {
  ret void
}

define void @g_unknown() !type !2 !type !1 {
  ret void
}

define void @f_cold() !type !0 !type !1 {
  ret void
}

define void @g_hot() !type !2 !type !1 {
  ret void
}

define void @f_none() !type !0 !type !1 {
  ret void
}

define void @g_cold() !type !2 !type !1 {
  ret void
}

define void @f_unknown() !type !0 !type !1 {
  ret void
}

define void @f_hot() !type !0 !type !1 {
  ret void
}

declare i1 @llvm.type.test(ptr %ptr, metadata %bitset) nounwind readnone

define i1 @test_typeid1(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid1")
  ret i1 %x
}

define i1 @test_typeid2(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid2")
  ret i1 %x
}

define i1 @test_generalized(ptr %p) {
  %x = call i1 @llvm.type.test(ptr %p, metadata !"typeid.generalized")
  ret i1 %x
}

!0 = !{i32 0, !"typeid1"}
!1 = !{i32 0, !"typeid2"}
!2 = !{i32 0, !"typeid.generalized"}
; CHECK-LABEL: define hidden void @f_critical.cfi(
; CHECK-SAME: ) !type [[META0:![0-9]+]] !type [[META1:![0-9]+]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @g_unknown.cfi(
; CHECK-SAME: ) !type [[META2:![0-9]+]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @f_cold.cfi(
; CHECK-SAME: ) !type [[META0]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @g_hot.cfi(
; CHECK-SAME: ) !type [[META2]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @f_none.cfi(
; CHECK-SAME: ) !type [[META0]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @g_cold.cfi(
; CHECK-SAME: ) !type [[META2]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @f_unknown.cfi(
; CHECK-SAME: ) !type [[META0]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define hidden void @f_hot.cfi(
; CHECK-SAME: ) !type [[META0]] !type [[META1]] {
; CHECK-NEXT:    ret void
;
;
; CHECK-LABEL: define i1 @test_typeid1(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 ptrtoint (ptr getelementptr (i8, ptr @.cfi.jumptable, i64 56) to i64), [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.fshr.i64(i64 [[TMP2]], i64 [[TMP2]], i64 3)
; CHECK-NEXT:    [[TMP4:%.*]] = icmp ule i64 [[TMP3]], 4
; CHECK-NEXT:    ret i1 [[TMP4]]
;
;
; CHECK-LABEL: define i1 @test_typeid2(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 ptrtoint (ptr getelementptr (i8, ptr @.cfi.jumptable, i64 56) to i64), [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.fshr.i64(i64 [[TMP2]], i64 [[TMP2]], i64 3)
; CHECK-NEXT:    [[TMP4:%.*]] = icmp ule i64 [[TMP3]], 7
; CHECK-NEXT:    ret i1 [[TMP4]]
;
;
; CHECK-LABEL: define i1 @test_generalized(
; CHECK-SAME: ptr [[P:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = ptrtoint ptr [[P]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = sub i64 ptrtoint (ptr getelementptr (i8, ptr @.cfi.jumptable, i64 16) to i64), [[TMP1]]
; CHECK-NEXT:    [[TMP3:%.*]] = call i64 @llvm.fshr.i64(i64 [[TMP2]], i64 [[TMP2]], i64 3)
; CHECK-NEXT:    [[TMP4:%.*]] = icmp ule i64 [[TMP3]], 2
; CHECK-NEXT:    ret i1 [[TMP4]]
;
;
; CHECK-LABEL: define private void @.cfi.jumptable(
; CHECK-SAME: ) #[[ATTR1:[0-9]+]] prefalign(8) !elf_section_properties [[META3:![0-9]+]] {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @g_cold.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @g_unknown.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @g_hot.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @f_cold.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @f_unknown.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @f_none.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @f_hot.cfi)
; CHECK-NEXT:    call void asm sideeffect "jmp ${0:c}@plt\0Aint3\0Aint3\0Aint3\0A", "s"(ptr @f_critical.cfi)
; CHECK-NEXT:    unreachable
;
;
; CHECK: [[META0]] = !{i32 0, !"typeid1"}
; CHECK: [[META1]] = !{i32 0, !"typeid2"}
; CHECK: [[META2]] = !{i32 0, !"typeid.generalized"}
; CHECK: [[META3]] = !{i64 1879002126, i64 8}
