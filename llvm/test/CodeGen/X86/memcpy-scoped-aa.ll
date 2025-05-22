; RUN: llc -mtriple=x86_64-linux-gnu -stop-after=finalize-isel -o - %s | FileCheck --check-prefix=MIR %s

; Ensure that the scoped AA is attached on loads/stores lowered from mem ops.

; Re-evaluate the slot numbers of scopes as that numbering could be changed run-by-run.

; MIR-DAG: ![[DOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"bax"}
; MIR-DAG: ![[SCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"bax: %p"}
; MIR-DAG: ![[SCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"bax: %q"}
; MIR-DAG: ![[SET0:[0-9]+]] = !{![[SCOPE0]]}
; MIR-DAG: ![[SET1:[0-9]+]] = !{![[SCOPE1]]}

; MIR-LABEL: name: test_memcpy
; MIR:      %2:gr64 = MOV64rm %0, 1, $noreg, 16, $noreg :: (load (s64) from %ir.p1, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: %3:gr64 = MOV64rm %0, 1, $noreg, 24, $noreg :: (load (s64) from %ir.p1 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 8, $noreg, killed %3 :: (store (s64) into %ir.p0 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 0, $noreg, killed %2 :: (store (s64) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memcpy(ptr nocapture %p, ptr nocapture readonly %q) {
  %p0 = bitcast ptr %p to ptr
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 4
  %p1 = bitcast ptr %add.ptr to ptr
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %p0, ptr noundef nonnull align 4 dereferenceable(16) %p1, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr %q, i64 1
  %v1 = load i32, ptr %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memcpy_inline
; MIR:      %2:gr64 = MOV64rm %0, 1, $noreg, 16, $noreg :: (load (s64) from %ir.p1, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: %3:gr64 = MOV64rm %0, 1, $noreg, 24, $noreg :: (load (s64) from %ir.p1 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 8, $noreg, killed %3 :: (store (s64) into %ir.p0 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 0, $noreg, killed %2 :: (store (s64) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memcpy_inline(ptr nocapture %p, ptr nocapture readonly %q) {
  %p0 = bitcast ptr %p to ptr
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 4
  %p1 = bitcast ptr %add.ptr to ptr
  tail call void @llvm.memcpy.inline.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %p0, ptr noundef nonnull align 4 dereferenceable(16) %p1, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr %q, i64 1
  %v1 = load i32, ptr %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memmove
; MIR:      %2:gr64 = MOV64rm %0, 1, $noreg, 16, $noreg :: (load (s64) from %ir.p1, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: %3:gr64 = MOV64rm %0, 1, $noreg, 24, $noreg :: (load (s64) from %ir.p1 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 0, $noreg, killed %2 :: (store (s64) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 8, $noreg, killed %3 :: (store (s64) into %ir.p0 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memmove(ptr nocapture %p, ptr nocapture readonly %q) {
  %p0 = bitcast ptr %p to ptr
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 4
  %p1 = bitcast ptr %add.ptr to ptr
  tail call void @llvm.memmove.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %p0, ptr noundef nonnull align 4 dereferenceable(16) %p1, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr %q, i64 1
  %v1 = load i32, ptr %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_memset
; MIR:      %2:gr64 = MOV64ri -6148914691236517206
; MIR-NEXT: MOV64mr %0, 1, $noreg, 8, $noreg, %2 :: (store (s64) into %ir.p0 + 8, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 0, $noreg, %2 :: (store (s64) into %ir.p0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_memset(ptr nocapture %p, ptr nocapture readonly %q) {
  %p0 = bitcast ptr %p to ptr
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %p0, i8 170, i64 16, i1 false), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr %q, i64 1
  %v1 = load i32, ptr %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

; MIR-LABEL: name: test_mempcpy
; MIR:      %2:gr64 = MOV64rm %0, 1, $noreg, 16, $noreg :: (load (s64) from %ir.p1, align 1, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: %3:gr64 = MOV64rm %0, 1, $noreg, 24, $noreg :: (load (s64) from %ir.p1 + 8, align 1, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 8, $noreg, killed %3 :: (store (s64) into %ir.p0 + 8, align 1, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR-NEXT: MOV64mr %0, 1, $noreg, 0, $noreg, killed %2 :: (store (s64) into %ir.p0, align 1, !alias.scope ![[SET0]], !noalias ![[SET1]])
define i32 @test_mempcpy(ptr nocapture %p, ptr nocapture readonly %q) {
  %p0 = bitcast ptr %p to ptr
  %add.ptr = getelementptr inbounds i32, ptr %p, i64 4
  %p1 = bitcast ptr %add.ptr to ptr
  %call = tail call ptr @mempcpy(ptr noundef nonnull align 4 dereferenceable(16) %p0, ptr noundef nonnull align 4 dereferenceable(16) %p1, i64 16), !alias.scope !2, !noalias !4
  %v0 = load i32, ptr %q, align 4, !alias.scope !4, !noalias !2
  %q1 = getelementptr inbounds i32, ptr %q, i64 1
  %v1 = load i32, ptr %q1, align 4, !alias.scope !4, !noalias !2
  %add = add i32 %v0, %v1
  ret i32 %add
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memcpy.inline.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)

declare ptr @mempcpy(ptr, ptr, i64)

!0 = distinct !{!0, !"bax"}
!1 = distinct !{!1, !0, !"bax: %p"}
!2 = !{!1}
!3 = distinct !{!3, !0, !"bax: %q"}
!4 = !{!3}
